"""End-to-end CLI tests: stub server on a real socket, cli.main() called directly."""

import json
import socket
import statistics
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from llmbench import cli
from llmbench.manifest import fingerprint as compute_fingerprint
from llmbench.metrics import summarize_rep
from llmbench.runio import read_records_jsonl
from llmbench.stubserver import StubConfig, create_app

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _StubServer:
    """The real stub app under uvicorn on an ephemeral port, in a daemon thread."""

    def __init__(self, config: StubConfig) -> None:
        self._server = uvicorn.Server(
            uvicorn.Config(create_app(config), host="127.0.0.1", port=0, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> str:
        self._thread.start()
        deadline = time.monotonic() + 15
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("stub server thread died during startup")
            if time.monotonic() > deadline:
                raise RuntimeError("stub server did not start within 15s")
            time.sleep(0.01)
        port = int(self._server.servers[0].sockets[0].getsockname()[1])
        return f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest.fixture(scope="module")
def stub_url() -> Iterator[str]:
    server = _StubServer(StubConfig(ttft_ms=40, itl_ms=2, seed=7))
    yield server.start()
    server.stop()


@pytest.fixture(scope="module")
def run_dir(stub_url: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full `llm-bench run` against the stub, shared by the assertion tests."""
    out = tmp_path_factory.mktemp("runs")
    rc = cli.main(
        [
            "run",
            "--base-url",
            stub_url,
            "--workload",
            "chat-short",
            "--num-requests",
            "12",
            "--warmup",
            "2",
            "--repeats",
            "2",
            "--concurrency",
            "4",
            "--timeout-s",
            "30",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("stub", "run", "envinfo", "analyze"):
        assert command in out


def test_run_writes_manifest_and_rep_files(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (run_dir / "rep1.jsonl").is_file()
    assert (run_dir / "rep2.jsonl").is_file()

    # Fingerprint must be recomputable from the manifest itself.
    assert manifest["fingerprint"] == compute_fingerprint(manifest)
    assert run_dir.name.startswith(manifest["fingerprint"][:12] + "-")

    assert manifest["model"]["id"] == "stub-model"  # auto-detected from /v1/models
    assert manifest["engine"]["version"].startswith("stub-")  # auto-detected from /version
    assert manifest["workload"]["name"] == "chat-short"
    assert manifest["workload"]["max_output_tokens"] == 128
    assert manifest["loadgen"]["num_requests"] == 12
    assert manifest["loadgen"]["warmup_requests"] == 2
    assert manifest["loadgen"]["repeats"] == 2


def test_run_records_meet_calibration_bounds(run_dir: Path) -> None:
    measured: list[dict[str, Any]] = []
    for rep_file in ("rep1.jsonl", "rep2.jsonl"):
        records = read_records_jsonl(run_dir / rep_file)
        assert len(records) == 14  # 2 warmup + 12 measured
        assert sum(1 for r in records if r["phase"] == "warmup") == 2
        assert all(r["ok"] for r in records)  # stub error_rate is 0
        for r in records:
            assert r["output_tokens"] == 128  # exactly the workload's max_output_tokens
            assert r["tokens_source"] == "usage"
        measured.extend(r for r in records if r["phase"] == "measure")

    ttft_p50 = statistics.median(r["ttft_s"] for r in measured)
    assert 0.03 <= ttft_p50 <= 0.25  # stub ttft is 40ms

    gaps = [gap for r in measured for gap in r["itl_s"]]
    assert len(gaps) == len(measured) * 127
    mean_itl = sum(gaps) / len(gaps)
    assert 0.001 <= mean_itl <= 0.02  # stub itl is 2ms


def test_analyze_with_plot(run_dir: Path, tmp_path: Path) -> None:
    plot_path = tmp_path / "latency.png"
    rc = cli.main(["analyze", str(run_dir), "--plot", str(plot_path)])
    assert rc == 0
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
    assert plot_path.read_bytes()[:8] == PNG_MAGIC


def test_analyze_missing_dir_exits_nonzero(tmp_path: Path) -> None:
    assert cli.main(["analyze", str(tmp_path / "nope")]) != 0


def test_analyze_prints_hardware_line(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A run recorded without --hardware-json must be visibly flagged at review time."""
    rc = cli.main(["analyze", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hardware:" in out
    assert "no hardware info recorded" in out


def test_analyze_corrupt_rep_file_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rep file truncated mid-record (killed run) must give a clean error, not a traceback."""
    run_dir = tmp_path / "deadbeefdead-20260101T000000Z"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"fingerprint": "deadbeef"}), encoding="utf-8"
    )
    (run_dir / "rep1.jsonl").write_text('{"phase": "measure", "ok": tru', encoding="utf-8")
    rc = cli.main(["analyze", str(run_dir)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "rep1.jsonl" in err
    assert "line 1" in err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--concurrency", "0"),
        ("--num-requests", "0"),
        ("--repeats", "0"),
        ("--warmup", "-1"),
    ],
)
def test_run_rejects_out_of_range_counts(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """concurrency=0 used to hang forever; repeats=0 crashed after writing the manifest."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--base-url", "http://127.0.0.1:1", flag, value])
    assert exc_info.value.code == 2
    assert "must be an integer" in capsys.readouterr().err


def test_run_without_hardware_json_warns_on_stderr(
    stub_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """All-null hardware is permitted but must never be silent."""
    rc = cli.main(
        [
            "run",
            "--base-url",
            stub_url,
            "--num-requests",
            "1",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--concurrency",
            "1",
            "--timeout-s",
            "30",
            "--out",
            str(tmp_path / "runs"),
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "hardware" in err
    assert "envinfo" in err


def test_run_rejects_typoed_hardware_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo'd hardware key must fail the run before any run directory exists."""
    hw = tmp_path / "hw.json"
    hw.write_text(json.dumps({"gpuname": "NVIDIA RTX 4090"}), encoding="utf-8")
    rc = cli.main(
        [
            "run",
            "--base-url",
            "http://127.0.0.1:1",
            "--model",
            "m",
            "--engine-version",
            "v",
            "--out",
            str(tmp_path / "runs"),
            "--hardware-json",
            str(hw),
        ]
    )
    assert rc == 2
    assert "gpuname" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def _quick_run_fingerprint(stub_url: str, out: Path, extra: list[str]) -> str:
    rc = cli.main(
        [
            "run",
            "--base-url",
            stub_url,
            "--num-requests",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--concurrency",
            "2",
            "--timeout-s",
            "30",
            "--out",
            str(out),
            *extra,
        ]
    )
    assert rc == 0
    (run_dir,) = (p for p in out.iterdir() if p.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest["fingerprint"])


def test_fingerprint_sensitivity(stub_url: str, tmp_path: Path) -> None:
    fp_a = _quick_run_fingerprint(stub_url, tmp_path / "a", [])
    fp_b = _quick_run_fingerprint(stub_url, tmp_path / "b", [])
    fp_arg = _quick_run_fingerprint(
        stub_url, tmp_path / "c", ["--serving-arg=--max-model-len=4096"]
    )
    fp_conc = _quick_run_fingerprint(stub_url, tmp_path / "d", ["--concurrency", "3"])

    assert len(fp_a) == 64
    assert fp_a == fp_b  # identical config -> identical fingerprint
    assert fp_arg != fp_a  # serving args are fingerprinted
    assert fp_conc != fp_a  # loadgen concurrency is fingerprinted


def test_error_injection_end_to_end(tmp_path: Path) -> None:
    server = _StubServer(StubConfig(ttft_ms=1, itl_ms=0, error_rate=0.5, seed=11))
    url = server.start()
    try:
        out = tmp_path / "runs"
        rc = cli.main(
            [
                "run",
                "--base-url",
                url,
                "--num-requests",
                "20",
                "--warmup",
                "0",
                "--repeats",
                "1",
                "--concurrency",
                "4",
                "--timeout-s",
                "30",
                "--out",
                str(out),
            ]
        )
        assert rc == 0  # some requests succeeded, so the run is usable
    finally:
        server.stop()

    (run_dir,) = (p for p in out.iterdir() if p.is_dir())
    records = read_records_jsonl(run_dir / "rep1.jsonl")
    assert len(records) == 20
    summary = summarize_rep(records, rep=1)
    assert 0.2 <= summary.error_rate <= 0.8

    failures = [r for r in records if not r["ok"]]
    assert failures
    for r in failures:
        assert isinstance(r["error"], str)
        assert r["error"]
    # Distributions come from OK records only.
    assert summary.ttft is not None
    assert summary.ttft.n == summary.requests_ok
    assert summary.e2e is not None
    assert summary.e2e.n == summary.requests_ok


def test_run_exits_nonzero_when_every_request_fails(tmp_path: Path) -> None:
    # Grab a free port and close it again: nothing is listening there.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    rc = cli.main(
        [
            "run",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--model",
            "unreachable-model",
            "--engine-version",
            "none",
            "--num-requests",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--concurrency",
            "1",
            "--timeout-s",
            "5",
            "--out",
            str(tmp_path / "runs"),
        ]
    )
    assert rc != 0


def test_envinfo_prints_hardware_shape_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["envinfo"])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert set(info) == {"gpu_name", "gpu_count", "driver_version", "cuda_version", "vram_mib"}
