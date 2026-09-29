"""Tests for the manifest contract: fingerprint determinism and run_id format.

Deliberately does not ``import pytest``: mypy following pytest's internals pulls
in numpy's stubs, whose PEP 695 ``type`` statements fail under the project's
pinned ``python_version = "3.11"``. Plain asserts keep mypy --strict green.
"""

import json
import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from llmbench.manifest import (
    EngineInfo,
    HardwareInfo,
    LoadgenInfo,
    ModelInfo,
    build_manifest,
    fingerprint,
    make_run_id,
)

_WORKLOAD: dict[str, object] = {
    "name": "chat-short",
    "description": "Short chat prompts, fixed output length.",
    "prompt_source": "chat_prompts.txt",
    "max_output_tokens": 128,
    "temperature": 0.0,
    "ignore_eos": True,
    "seed": 1337,
}


def _build(**overrides: Any) -> dict[str, object]:
    """Build a baseline manifest, applying keyword overrides to the config."""
    kwargs: dict[str, Any] = {
        "model": ModelInfo(id="meta-llama/Llama-3.1-8B-Instruct", revision="abc123"),
        "engine": EngineInfo(
            name="vllm",
            version="0.6.3",
            image="vllm/vllm-openai:v0.6.3",
            image_digest="sha256:deadbeef",
            serving_args=("--max-num-seqs", "256"),
        ),
        "quantization": "fp16",
        "hardware": HardwareInfo(
            gpu_name="NVIDIA A100-SXM4-80GB",
            gpu_count=1,
            driver_version="550.54.15",
            cuda_version="12.4",
            vram_mib=81920,
        ),
        "workload": dict(_WORKLOAD),
        "loadgen": LoadgenInfo(
            harness_version="0.1.0",
            concurrency=8,
            num_requests=100,
            warmup_requests=5,
            repeats=1,
            timeout_s=120.0,
        ),
        "created_at": datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC),
        "notes": None,
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def _expect_value_error(**overrides: Any) -> None:
    """Assert that _build(**overrides) raises ValueError mentioning tz-awareness."""
    message = ""
    try:
        _build(**overrides)
    except ValueError as exc:
        message = str(exc)
    assert "timezone-aware" in message


def test_identical_config_identical_fingerprint() -> None:
    a = _build()
    b = _build()
    assert a is not b
    assert a["fingerprint"] == b["fingerprint"]


def test_config_change_changes_fingerprint() -> None:
    cases: dict[str, dict[str, Any]] = {
        "serving-arg": {
            "engine": EngineInfo(
                name="vllm",
                version="0.6.3",
                image="vllm/vllm-openai:v0.6.3",
                image_digest="sha256:deadbeef",
                serving_args=("--max-num-seqs", "128"),
            )
        },
        "model-revision": {
            "model": ModelInfo(id="meta-llama/Llama-3.1-8B-Instruct", revision="def456")
        },
        "quantization": {"quantization": "awq"},
        "loadgen-concurrency": {
            "loadgen": LoadgenInfo(
                harness_version="0.1.0",
                concurrency=16,
                num_requests=100,
                warmup_requests=5,
                repeats=1,
                timeout_s=120.0,
            )
        },
        "workload-seed": {"workload": {**_WORKLOAD, "seed": 42}},
    }
    baseline = _build()["fingerprint"]
    for case_id, overrides in cases.items():
        changed = _build(**overrides)["fingerprint"]
        assert changed != baseline, f"fingerprint unchanged for case {case_id!r}"


def test_created_at_and_notes_excluded_from_fingerprint() -> None:
    a = _build(created_at=datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC), notes=None)
    b = _build(created_at=datetime(2027, 1, 1, 3, 4, 5, tzinfo=UTC), notes="rerun on same box")
    assert a["created_at"] != b["created_at"]
    assert a["notes"] != b["notes"]
    assert a["fingerprint"] == b["fingerprint"]


def test_fingerprint_key_itself_excluded() -> None:
    # Recomputing over the full manifest (which now contains "fingerprint")
    # must reproduce the stored value, proving the key is excluded.
    manifest = _build()
    assert fingerprint(manifest) == manifest["fingerprint"]


def test_fingerprint_is_64_lowercase_hex() -> None:
    fp = _build()["fingerprint"]
    assert isinstance(fp, str)
    assert re.fullmatch(r"[0-9a-f]{64}", fp)


def test_manifest_shape_and_json_round_trip() -> None:
    manifest = _build()
    assert manifest["schema_version"] == 1
    assert set(manifest) == {
        "schema_version",
        "fingerprint",
        "created_at",
        "model",
        "engine",
        "quantization",
        "hardware",
        "workload",
        "loadgen",
        "notes",
    }
    engine = manifest["engine"]
    assert isinstance(engine, dict)
    assert engine["serving_args"] == ["--max-num-seqs", "256"]
    assert json.loads(json.dumps(manifest)) == manifest


def test_created_at_recorded_as_utc_iso8601() -> None:
    est = timezone(timedelta(hours=-5))
    manifest = _build(created_at=datetime(2026, 9, 28, 7, 0, 0, tzinfo=est))
    assert manifest["created_at"] == "2026-09-28T12:00:00+00:00"


def test_naive_created_at_rejected() -> None:
    _expect_value_error(created_at=datetime(2026, 9, 28, 12, 0, 0))  # intentionally naive


def test_run_id_format() -> None:
    manifest = _build()
    fp = manifest["fingerprint"]
    assert isinstance(fp, str)
    run_id = make_run_id(fp, datetime(2026, 9, 28, 12, 34, 56, tzinfo=UTC))
    assert run_id == f"{fp[:12]}-20260928T123456Z"
    assert re.fullmatch(r"[0-9a-f]{12}-\d{8}T\d{6}Z", run_id)


def test_run_id_converts_to_utc_and_rejects_naive() -> None:
    est = timezone(timedelta(hours=-5))
    assert make_run_id("a" * 64, datetime(2026, 9, 28, 7, 0, 0, tzinfo=est)).endswith(
        "-20260928T120000Z"
    )
    message = ""
    try:
        make_run_id("a" * 64, datetime(2026, 9, 28, 12, 0, 0))  # intentionally naive
    except ValueError as exc:
        message = str(exc)
    assert "timezone-aware" in message
