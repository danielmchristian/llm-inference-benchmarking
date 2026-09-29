"""Tests for llmbench.analysis formatting: per-row rep counts and the hardware line."""

from typing import Any

from llmbench.analysis import format_cross_table, format_manifest_summary
from llmbench.metrics import summarize_rep, summarize_run

_HW_KEYS = ("gpu_name", "gpu_count", "driver_version", "cuda_version", "vram_mib")


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "phase": "measure",
        "ok": True,
        "start_unix": 0.0,
        "start_mono_s": 0.0,
        "e2e_s": 1.0,
        "ttft_s": 0.1,
        "itl_s": [0.01],
        "output_tokens": 8,
        "tpot_s": 0.02,
    }
    record.update(overrides)
    return record


def test_cross_table_shows_per_row_rep_count() -> None:
    """A rep with zero successes must not be silently absent from a row's claimed n."""
    failed_rep = summarize_rep([_record(ok=False, error="boom")], rep=1)
    ok_rep = summarize_rep([_record()], rep=2)
    table = format_cross_table(summarize_run([failed_rep, ok_rep]))
    lines = table.splitlines()
    assert lines[0] == "cross-rep summary (2 reps)"
    assert lines[1].split()[-1] == "n_reps"
    by_metric = {line.split()[0]: line.split() for line in lines[2:]}
    assert by_metric["ttft_p50_s"][-1] == "1"  # only rep 2 supplied ttft
    assert by_metric["e2e_p50_s"][-1] == "1"
    assert by_metric["error_rate"][-1] == "2"  # every rep has an error rate
    assert by_metric["requests_per_s"][-1] == "2"


def test_manifest_summary_shows_hardware() -> None:
    manifest = {
        "fingerprint": "f" * 64,
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "driver_version": "550.54.15",
            "cuda_version": "12.4",
            "vram_mib": 24564,
        },
    }
    text = format_manifest_summary(manifest)
    assert "hardware:" in text
    assert "NVIDIA GeForce RTX 4090 x1 driver=550.54.15 cuda=12.4 vram_mib=24564" in text


def test_manifest_summary_flags_missing_hardware() -> None:
    """All-null hardware must be visibly flagged, not omitted, at review time."""
    all_null = format_manifest_summary({"hardware": dict.fromkeys(_HW_KEYS)})
    no_key = format_manifest_summary({})
    for text in (all_null, no_key):
        assert "hardware:     unknown (no hardware info recorded)" in text
