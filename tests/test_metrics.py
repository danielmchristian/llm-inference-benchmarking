"""Tests for llmbench.metrics against the record and metrics contracts."""

import json
import math
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from llmbench.metrics import RepSummary, summarize_rep, summarize_run


def make_record(
    *,
    rep: int = 1,
    request_id: int = 0,
    phase: str = "measure",
    ok: bool = True,
    error: str | None = None,
    status_code: int | None = 200,
    start_unix: float = 0.0,
    start_mono_s: float | None = None,
    e2e_s: float | None = None,
    ttft_s: float | None = None,
    itl_s: list[float] | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    tokens_source: str | None = None,
    tpot_s: float | None = None,
) -> dict[str, Any]:
    """Build a record dict matching the run record contract."""
    return {
        "schema_version": 1,
        "rep": rep,
        "request_id": request_id,
        "phase": phase,
        "ok": ok,
        "error": error,
        "status_code": status_code,
        "start_unix": start_unix,
        "start_mono_s": start_mono_s,
        "e2e_s": e2e_s,
        "ttft_s": ttft_s,
        "itl_s": itl_s if itl_s is not None else [],
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tokens_source": tokens_source,
        "tpot_s": tpot_s,
    }


def test_percentiles_match_numpy_quantile() -> None:
    """Dist percentiles must equal numpy.quantile (default linear interpolation)."""
    values = [0.10, 0.35, 0.02, 0.80, 0.44, 0.61, 0.29, 0.05, 0.93, 0.17, 0.52]
    records = [
        make_record(request_id=i, start_unix=float(i), e2e_s=v) for i, v in enumerate(values)
    ]
    summary = summarize_rep(records, rep=1)
    arr = np.asarray(values)
    assert summary.e2e is not None
    assert summary.e2e.p50 == pytest.approx(float(np.quantile(arr, 0.5)))
    assert summary.e2e.p90 == pytest.approx(float(np.quantile(arr, 0.9)))
    assert summary.e2e.p99 == pytest.approx(float(np.quantile(arr, 0.99)))
    assert summary.e2e.mean == pytest.approx(float(arr.mean()))
    assert summary.e2e.std == pytest.approx(float(arr.std()))
    assert summary.e2e.n == len(values)


def test_warmup_records_excluded() -> None:
    warmup = make_record(phase="warmup", start_unix=0.0, e2e_s=99.0, ttft_s=50.0, output_tokens=999)
    measure = make_record(request_id=1, start_unix=100.0, e2e_s=2.0, ttft_s=0.5, output_tokens=10)
    summary = summarize_rep([warmup, measure], rep=1)
    assert summary.requests_ok == 1
    assert summary.requests_failed == 0
    assert summary.e2e is not None
    assert summary.e2e.n == 1
    assert summary.e2e.mean == pytest.approx(2.0)
    assert summary.output_tokens_total == 10
    # Window spans only the measured record, not the warmup one.
    assert summary.duration_s == pytest.approx(2.0)


def test_failed_records_excluded_from_dists_but_counted_in_error_rate() -> None:
    records = [
        make_record(request_id=0, start_unix=0.0, e2e_s=1.0, ttft_s=0.2, output_tokens=5),
        make_record(request_id=1, start_unix=0.5, e2e_s=3.0, ttft_s=0.4, output_tokens=5),
        make_record(
            request_id=2,
            ok=False,
            error="timeout",
            status_code=None,
            start_unix=1.0,
            e2e_s=500.0,  # best-effort timing on a failure must not pollute dists
            ttft_s=400.0,
        ),
    ]
    summary = summarize_rep(records, rep=1)
    assert summary.requests_ok == 2
    assert summary.requests_failed == 1
    assert summary.error_rate == pytest.approx(1 / 3)
    assert summary.ttft is not None
    assert summary.ttft.n == 2
    assert summary.ttft.mean == pytest.approx(0.3)
    assert summary.e2e is not None
    assert summary.e2e.n == 2


def test_duration_window_from_ok_measured_records_only() -> None:
    records = [
        make_record(request_id=0, start_unix=10.0, e2e_s=2.0),
        make_record(request_id=1, start_unix=11.0, e2e_s=4.0),  # window end: 15.0
        # A failed record starting earlier and ending later must not widen the window.
        make_record(request_id=2, ok=False, error="boom", start_unix=1.0, e2e_s=100.0),
    ]
    summary = summarize_rep(records, rep=1)
    assert summary.duration_s == pytest.approx(5.0)  # 15.0 - 10.0
    assert summary.requests_per_s == pytest.approx(2 / 5.0)


def test_duration_window_prefers_monotonic_start_over_wall_clock() -> None:
    """A wall-clock step mid-rep must not corrupt the throughput window.

    e2e_s is perf_counter-based, so the window must anchor on the monotonic
    start_mono_s, not start_unix (which an NTP step can move mid-rep).
    """
    records = [
        make_record(request_id=0, start_unix=1000.0, start_mono_s=10.0, e2e_s=2.0, output_tokens=6),
        # Wall clock stepped back 30s between requests; the monotonic clock kept ticking.
        make_record(request_id=1, start_unix=971.0, start_mono_s=11.0, e2e_s=2.0, output_tokens=6),
    ]
    summary = summarize_rep(records, rep=1)
    assert summary.duration_s == pytest.approx(3.0)  # 13.0 - 10.0 in the monotonic domain
    assert summary.output_tokens_per_s == pytest.approx(12 / 3.0)
    assert summary.requests_per_s == pytest.approx(2 / 3.0)


def test_duration_window_falls_back_to_start_unix_for_old_records() -> None:
    records = [
        make_record(request_id=0, start_unix=10.0, e2e_s=2.0),
        make_record(request_id=1, start_unix=11.0, e2e_s=4.0),
    ]
    summary = summarize_rep(records, rep=1)
    assert summary.duration_s == pytest.approx(5.0)


def test_throughput_uses_window_and_token_totals() -> None:
    records = [
        make_record(request_id=0, start_unix=0.0, e2e_s=2.0, output_tokens=30),
        make_record(request_id=1, start_unix=1.0, e2e_s=3.0, output_tokens=50),  # ends at 4.0
    ]
    summary = summarize_rep(records, rep=1)
    assert summary.output_tokens_total == 80
    assert summary.duration_s == pytest.approx(4.0)
    assert summary.output_tokens_per_s == pytest.approx(80 / 4.0)
    assert summary.requests_per_s == pytest.approx(2 / 4.0)


def test_itl_flattened_across_ok_requests() -> None:
    records = [
        make_record(request_id=0, start_unix=0.0, e2e_s=1.0, itl_s=[0.1, 0.2]),
        make_record(request_id=1, start_unix=0.0, e2e_s=1.0, itl_s=[0.3]),
        make_record(request_id=2, ok=False, error="x", start_unix=0.0, itl_s=[9.0]),
    ]
    summary = summarize_rep(records, rep=1)
    assert summary.itl is not None
    assert summary.itl.n == 3
    assert summary.itl.mean == pytest.approx((0.1 + 0.2 + 0.3) / 3)


def test_empty_input_returns_guarded_summary() -> None:
    summary = summarize_rep([], rep=1)
    assert summary == RepSummary(
        rep=1,
        requests_ok=0,
        requests_failed=0,
        error_rate=0.0,
        ttft=None,
        e2e=None,
        tpot=None,
        itl=None,
        output_tokens_total=0,
        duration_s=0.0,
        output_tokens_per_s=0.0,
        requests_per_s=0.0,
    )


def test_all_failed_input_returns_guarded_summary() -> None:
    records = [
        make_record(request_id=i, ok=False, error="refused", status_code=503, start_unix=float(i))
        for i in range(4)
    ]
    summary = summarize_rep(records, rep=2)
    assert summary.requests_ok == 0
    assert summary.requests_failed == 4
    assert summary.error_rate == pytest.approx(1.0)
    assert summary.ttft is None
    assert summary.e2e is None
    assert summary.tpot is None
    assert summary.itl is None
    assert summary.duration_s == 0.0
    assert summary.output_tokens_per_s == 0.0
    assert summary.requests_per_s == 0.0
    # An all-failed rep still aggregates across reps without raising.
    cross = summarize_run([summary])
    assert cross.n_reps == 1
    assert cross.ttft_p50 is None
    assert cross.error_rate is not None
    assert cross.error_rate.mean == pytest.approx(1.0)


def _rep_with(e2e_values: list[float], rep: int) -> RepSummary:
    records = [
        make_record(request_id=i, start_unix=float(i), e2e_s=v, ttft_s=v / 10, output_tokens=10)
        for i, v in enumerate(e2e_values)
    ]
    return summarize_rep(records, rep=rep)


def test_cross_rep_mean_std_min_max_hand_computed() -> None:
    # Per-rep e2e p50s are exactly 1.0, 2.0, 4.0 (odd-length arrays: median is exact).
    reps = [
        _rep_with([0.5, 1.0, 1.5], rep=1),
        _rep_with([1.0, 2.0, 3.0], rep=2),
        _rep_with([2.0, 4.0, 6.0], rep=3),
    ]
    cross = summarize_run(reps)
    assert cross.n_reps == 3
    assert cross.e2e_p50 is not None
    p50s = [1.0, 2.0, 4.0]
    mean = sum(p50s) / 3  # 7/3
    assert cross.e2e_p50.mean == pytest.approx(mean)
    assert cross.e2e_p50.std == pytest.approx(math.sqrt(sum((v - mean) ** 2 for v in p50s) / 3))
    assert cross.e2e_p50.min == pytest.approx(1.0)
    assert cross.e2e_p50.max == pytest.approx(4.0)
    assert cross.ttft_p50 is not None
    assert cross.ttft_p50.min == pytest.approx(0.1)
    assert cross.ttft_p50.max == pytest.approx(0.4)
    assert cross.error_rate is not None
    assert cross.error_rate.mean == 0.0


def test_cross_rep_stats_report_contributing_rep_count() -> None:
    """A rep with no data for a metric must be visible in that row's rep count."""
    failed_rep = summarize_rep([make_record(ok=False, error="boom", status_code=500)], rep=1)
    ok_rep = _rep_with([1.0, 2.0, 3.0], rep=2)
    cross = summarize_run([failed_rep, ok_rep])
    assert cross.n_reps == 2
    assert cross.e2e_p50 is not None
    assert cross.e2e_p50.n == 1  # only rep 2 supplied e2e
    assert cross.ttft_p50 is not None
    assert cross.ttft_p50.n == 1
    assert cross.error_rate is not None
    assert cross.error_rate.n == 2  # every rep has an error rate
    assert cross.requests_per_s is not None
    assert cross.requests_per_s.n == 2


def test_summarize_run_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_run([])


def test_asdict_round_trips_to_json() -> None:
    records = [
        make_record(
            request_id=i,
            start_unix=float(i),
            e2e_s=1.0 + i,
            ttft_s=0.1 * (i + 1),
            itl_s=[0.01, 0.02],
            output_tokens=8,
            tpot_s=0.05,
            tokens_source="usage",
        )
        for i in range(3)
    ]
    rep_summary = summarize_rep(records, rep=1)
    cross = summarize_run([rep_summary])

    rep_json = json.loads(json.dumps(asdict(rep_summary)))
    assert rep_json["rep"] == 1
    assert rep_json["requests_ok"] == 3
    assert rep_json["ttft"]["n"] == 3
    assert rep_json["ttft"]["p50"] == pytest.approx(0.2)

    cross_json = json.loads(json.dumps(asdict(cross)))
    assert cross_json["n_reps"] == 1
    assert cross_json["e2e_p50"]["mean"] == pytest.approx(2.0)
    assert cross_json["itl_p50"]["min"] == pytest.approx(0.015)
