"""llm-bench command-line interface.

Subcommands: ``stub`` (deterministic local endpoint), ``run`` (benchmark an
endpoint and record a run directory), ``envinfo`` (hardware JSON from
nvidia-smi), and ``analyze`` (summarize and plot a recorded run).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

import llmbench
from llmbench import analysis, manifest, metrics, runio, workloads
from llmbench.loadgen import detect_engine_version, detect_model, run_load
from llmbench.stubserver import StubConfig, run_stub


def _positive_int(text: str) -> int:
    """argparse type: integer >= 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return value


def _nonnegative_int(text: str) -> int:
    """argparse type: integer >= 0."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be an integer >= 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="llm-bench",
        description="Benchmark OpenAI-compatible LLM inference endpoints.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stub = sub.add_parser("stub", help="run the deterministic OpenAI-compatible stub server")
    stub.add_argument("--port", type=int, default=8080)
    stub.add_argument("--ttft-ms", type=float, default=60.0)
    stub.add_argument("--itl-ms", type=float, default=15.0)
    stub.add_argument("--ttft-jitter-ms", type=float, default=0.0)
    stub.add_argument("--itl-jitter-ms", type=float, default=0.0)
    stub.add_argument("--error-rate", type=float, default=0.0)
    stub.add_argument("--seed", type=int, default=0)

    run = sub.add_parser("run", help="benchmark an endpoint and record a run directory")
    run.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    run.add_argument("--workload", default="chat-short", choices=sorted(workloads.WORKLOADS))
    run.add_argument("--concurrency", type=_positive_int, default=8)
    run.add_argument("--num-requests", type=_positive_int, default=256)
    run.add_argument("--warmup", type=_nonnegative_int, default=16)
    run.add_argument("--repeats", type=_positive_int, default=3)
    run.add_argument("--timeout-s", type=float, default=120.0)
    run.add_argument("--out", type=Path, default=Path("runs"))
    run.add_argument("--model", default=None, help="model id; auto-detected from /v1/models")
    run.add_argument("--model-revision", default=None)
    run.add_argument("--engine-name", default="unknown")
    run.add_argument("--engine-version", default=None, help="auto-detected from /version")
    run.add_argument("--quantization", default="fp16")
    run.add_argument("--image", default=None)
    run.add_argument("--image-digest", default=None)
    run.add_argument(
        "--serving-arg",
        action="append",
        default=None,
        dest="serving_args",
        help="repeatable; for values starting with '-' use --serving-arg=<value>",
    )
    run.add_argument("--hardware-json", type=Path, default=None)
    run.add_argument("--notes", default=None)

    sub.add_parser("envinfo", help="print hardware JSON detected via nvidia-smi")

    analyze = sub.add_parser("analyze", help="summarize (and optionally plot) a run directory")
    analyze.add_argument("run_dir", type=Path)
    analyze.add_argument("--plot", type=Path, default=None, help="write a PNG figure here")

    return parser


def _cmd_stub(args: argparse.Namespace) -> int:
    config = StubConfig(
        ttft_ms=args.ttft_ms,
        ttft_jitter_ms=args.ttft_jitter_ms,
        itl_ms=args.itl_ms,
        itl_jitter_ms=args.itl_jitter_ms,
        error_rate=args.error_rate,
        seed=args.seed,
    )
    print(f"stub listening on http://127.0.0.1:{args.port} (model_id={config.model_id})")
    run_stub(config, port=args.port)
    return 0


_HARDWARE_KEYS = ("gpu_name", "gpu_count", "driver_version", "cuda_version", "vram_mib")


def _load_hardware(path: Path | None) -> manifest.HardwareInfo:
    """Load HardwareInfo from a JSON file; unknown keys raise ValueError (typo guard)."""
    if path is None:
        return manifest.HardwareInfo()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if unknown := sorted(set(data) - set(_HARDWARE_KEYS)):
        raise ValueError(
            f"unknown hardware keys in {path}: {', '.join(unknown)} "
            f"(expected only: {', '.join(_HARDWARE_KEYS)})"
        )
    return manifest.HardwareInfo(
        gpu_name=data.get("gpu_name"),
        gpu_count=data.get("gpu_count"),
        driver_version=data.get("driver_version"),
        cuda_version=data.get("cuda_version"),
        vram_mib=data.get("vram_mib"),
    )


async def _run_async(args: argparse.Namespace) -> int:
    spec = workloads.WORKLOADS[args.workload]
    base_url: str = args.base_url

    # Hardware is validated before any network or filesystem side effects so a
    # typo'd file fails the run cleanly, and an all-null result is called out:
    # it would give runs on different GPUs identical fingerprints.
    try:
        hardware = _load_hardware(args.hardware_json)
    except (OSError, ValueError) as exc:
        print(f"error: could not load --hardware-json: {exc}", file=sys.stderr)
        return 2
    if all(v is None for v in hardware.to_dict().values()):
        print(
            "warning: no hardware info recorded (manifest.hardware will be all null, so "
            "runs on different GPUs share a fingerprint); pass --hardware-json, "
            "generated with `llm-bench envinfo`",
            file=sys.stderr,
        )

    model_id: str
    if args.model is not None:
        model_id = args.model
    else:
        try:
            model_id = await detect_model(base_url)
        except (httpx.HTTPError, ValueError) as exc:
            print(
                f"error: could not detect model from {base_url}/v1/models: {exc}",
                file=sys.stderr,
            )
            return 2
    engine_version = args.engine_version or await detect_engine_version(base_url)

    manifest_dict = manifest.build_manifest(
        model=manifest.ModelInfo(id=model_id, revision=args.model_revision),
        engine=manifest.EngineInfo(
            name=args.engine_name,
            version=engine_version,
            image=args.image,
            image_digest=args.image_digest,
            serving_args=tuple(args.serving_args or ()),
        ),
        quantization=args.quantization,
        hardware=hardware,
        workload=workloads.to_manifest_dict(spec),
        loadgen=manifest.LoadgenInfo(
            harness_version=llmbench.__version__,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            warmup_requests=args.warmup,
            repeats=args.repeats,
            timeout_s=args.timeout_s,
        ),
        created_at=datetime.now(UTC),
        notes=args.notes,
    )
    fp = str(manifest_dict["fingerprint"])
    run_dir = runio.create_run_dir(args.out, fp)
    runio.write_manifest(run_dir, manifest_dict)
    print(f"fingerprint: {fp}")
    print(f"run dir: {run_dir}")

    summaries: list[metrics.RepSummary] = []
    total_ok = 0
    for rep in range(1, args.repeats + 1):
        # Fresh (deterministic, identical) batch each rep: warmup + measured.
        messages = workloads.build_messages(spec, args.warmup + args.num_requests)
        records = await run_load(
            base_url,
            model_id,
            messages,
            max_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            ignore_eos=spec.ignore_eos,
            concurrency=args.concurrency,
            warmup_requests=args.warmup,
            timeout_s=args.timeout_s,
            rep=rep,
        )
        runio.append_records_jsonl(run_dir / f"rep{rep}.jsonl", records)
        summary = metrics.summarize_rep(records, rep)
        summaries.append(summary)
        total_ok += summary.requests_ok
        print(analysis.format_rep_line(summary))

    print()
    print(analysis.format_cross_table(metrics.summarize_run(summaries)))
    if total_ok == 0:
        print("error: every measured request failed", file=sys.stderr)
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return asyncio.run(_run_async(args))


def _collect_hardware() -> manifest.HardwareInfo:
    """Query nvidia-smi; all-null info (with a stderr warning) when unavailable."""
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        print("warning: nvidia-smi unavailable; emitting all-null hardware info", file=sys.stderr)
        return manifest.HardwareInfo()

    lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if not lines:
        print(
            "warning: nvidia-smi listed no GPUs; emitting all-null hardware info", file=sys.stderr
        )
        return manifest.HardwareInfo()
    fields = [part.strip() for part in lines[0].split(",")]
    gpu_name = fields[0] if fields[0] else None
    driver_version = fields[1] if len(fields) > 1 and fields[1] else None
    vram_mib: int | None = None
    if len(fields) > 2 and (match := re.match(r"(\d+)", fields[2])):
        vram_mib = int(match.group(1))

    cuda_version: str | None = None
    try:
        plain = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True, timeout=10
        )
        if match := re.search(r"CUDA Version:\s*([\d.]+)", plain.stdout):
            cuda_version = match.group(1)
    except (OSError, subprocess.SubprocessError):
        pass

    return manifest.HardwareInfo(
        gpu_name=gpu_name,
        gpu_count=len(lines),
        driver_version=driver_version,
        cuda_version=cuda_version,
        vram_mib=vram_mib,
    )


def _cmd_envinfo(args: argparse.Namespace) -> int:
    print(json.dumps(_collect_hardware().to_dict(), indent=2))
    return 0


def _rep_files(run_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in run_dir.glob("rep*.jsonl"):
        if match := re.fullmatch(r"rep(\d+)\.jsonl", path.name):
            files.append((int(match.group(1)), path))
    return sorted(files)


def _cmd_analyze(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"error: no manifest.json in {run_dir}", file=sys.stderr)
        return 2
    try:
        manifest_dict: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"error: corrupt manifest.json in {run_dir}: {exc}", file=sys.stderr)
        return 2
    rep_files = _rep_files(run_dir)
    if not rep_files:
        print(f"error: no rep<N>.jsonl files in {run_dir}", file=sys.stderr)
        return 2

    try:
        rep_records = [(rep, runio.read_records_jsonl(path)) for rep, path in rep_files]
    except ValueError as exc:
        # Typical cause: a rep file truncated mid-record by a killed run.
        print(f"error: corrupt rep file: {exc}", file=sys.stderr)
        return 2
    summaries = [metrics.summarize_rep(records, rep) for rep, records in rep_records]

    print(analysis.format_manifest_summary(manifest_dict))
    print()
    print(analysis.format_rep_table(summaries))
    print()
    print(analysis.format_cross_table(metrics.summarize_run(summaries)))

    if args.plot is not None:
        workload = manifest_dict.get("workload") or {}
        analysis.plot_run(
            rep_records,
            args.plot,
            workload_name=str(workload.get("name", "unknown")),
            fingerprint=str(manifest_dict.get("fingerprint", "")),
        )
        print(f"\nplot written to {args.plot}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    handlers = {
        "stub": _cmd_stub,
        "run": _cmd_run,
        "envinfo": _cmd_envinfo,
        "analyze": _cmd_analyze,
    }
    return handlers[str(args.command)](args)


if __name__ == "__main__":
    raise SystemExit(main())
