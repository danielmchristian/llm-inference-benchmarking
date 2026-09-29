"""Aggregate per-request benchmark records into per-rep and cross-rep summaries.

Input records follow the run record contract (one dict per request; see the
runner). Only ``phase == "measure"`` records count toward any statistic.
Failed requests (``ok == false``) are excluded from latency/throughput
distributions but are included in ``error_rate``.

All durations are seconds (float).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Record = Mapping[str, Any]


@dataclass(frozen=True)
class Dist:
    """Summary of a sample distribution.

    Percentiles are computed with ``numpy.quantile`` using its default linear
    interpolation between order statistics; ``std`` is the population standard
    deviation (``ddof=0``).
    """

    p50: float
    p90: float
    p99: float
    mean: float
    std: float
    n: int


@dataclass(frozen=True)
class MetricStats:
    """Cross-rep aggregate of one scalar metric (population std, ``ddof=0``).

    ``n`` is the number of reps that supplied the metric; it can be smaller
    than the run's rep count when some reps produced no data (e.g. every
    request failed), and honest reporting must surface that.
    """

    mean: float
    std: float
    min: float
    max: float
    n: int


@dataclass(frozen=True)
class RepSummary:
    """Summary of a single repetition's measured requests.

    ``duration_s`` is the measurement window: ``max(start + e2e_s) -
    min(start)`` over OK measured records, where ``start`` is the monotonic
    ``start_mono_s`` (falling back to wall-clock ``start_unix`` for records
    written before that field existed). Throughputs are totals divided by that
    window (0.0 when the window is empty or zero-length).
    """

    rep: int
    requests_ok: int
    requests_failed: int
    error_rate: float
    ttft: Dist | None
    e2e: Dist | None
    tpot: Dist | None
    itl: Dist | None
    output_tokens_total: int
    duration_s: float
    output_tokens_per_s: float
    requests_per_s: float


@dataclass(frozen=True)
class CrossRepSummary:
    """Cross-rep mean/std/min/max for headline metrics across a run's reps.

    ``n_reps`` counts the reps in the run; each ``MetricStats`` aggregates
    only the reps that supplied that metric and records how many in its own
    ``n`` (which can be < ``n_reps``). A field is None when no rep supplied
    the underlying value (e.g. ``ttft_p50`` when every rep's ttft distribution
    is None).
    """

    n_reps: int
    ttft_p50: MetricStats | None
    ttft_p90: MetricStats | None
    ttft_p99: MetricStats | None
    e2e_p50: MetricStats | None
    e2e_p99: MetricStats | None
    tpot_p50: MetricStats | None
    itl_p50: MetricStats | None
    output_tokens_per_s: MetricStats | None
    requests_per_s: MetricStats | None
    error_rate: MetricStats | None


def _dist(values: Sequence[float]) -> Dist | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(arr, (0.5, 0.9, 0.99))
    return Dist(
        p50=float(qs[0]),
        p90=float(qs[1]),
        p99=float(qs[2]),
        mean=float(arr.mean()),
        std=float(arr.std()),
        n=len(values),
    )


def summarize_rep(records: Iterable[Record], rep: int) -> RepSummary:
    """Summarize one rep's records; warmup records and failed-request timings excluded.

    Never raises on empty or all-failed input: distributions come back None and
    rates come back 0.0 (``error_rate`` is failed/measured, so 1.0 when all
    measured requests failed and 0.0 when nothing was measured).
    """
    measured = [r for r in records if r.get("phase") == "measure"]
    ok = [r for r in measured if r.get("ok")]
    n_failed = len(measured) - len(ok)

    ttft = [float(r["ttft_s"]) for r in ok if r.get("ttft_s") is not None]
    e2e = [float(r["e2e_s"]) for r in ok if r.get("e2e_s") is not None]
    tpot = [float(r["tpot_s"]) for r in ok if r.get("tpot_s") is not None]
    itl = [float(gap) for r in ok for gap in r.get("itl_s") or []]

    output_tokens_total = sum(int(r["output_tokens"]) for r in ok if r.get("output_tokens"))

    # Window endpoints come only from OK measured records with a known end
    # time. The monotonic start is preferred: e2e_s is perf_counter-based, so
    # anchoring on wall-clock start_unix would corrupt the window (and both
    # throughputs) if the wall clock stepped mid-rep. start_unix is only a
    # fallback for records written before start_mono_s existed.
    spans: list[tuple[float, float]] = []
    for r in ok:
        start = r.get("start_mono_s")
        if start is None:
            start = r.get("start_unix")
        if start is None or r.get("e2e_s") is None:
            continue
        spans.append((float(start), float(start) + float(r["e2e_s"])))
    duration_s = max(end for _, end in spans) - min(start for start, _ in spans) if spans else 0.0

    tokens_per_s = output_tokens_total / duration_s if duration_s > 0 else 0.0
    requests_per_s = len(ok) / duration_s if duration_s > 0 else 0.0

    return RepSummary(
        rep=rep,
        requests_ok=len(ok),
        requests_failed=n_failed,
        error_rate=n_failed / len(measured) if measured else 0.0,
        ttft=_dist(ttft),
        e2e=_dist(e2e),
        tpot=_dist(tpot),
        itl=_dist(itl),
        output_tokens_total=output_tokens_total,
        duration_s=duration_s,
        output_tokens_per_s=tokens_per_s,
        requests_per_s=requests_per_s,
    )


def _stats(
    reps: Sequence[RepSummary], get: Callable[[RepSummary], float | None]
) -> MetricStats | None:
    values = [v for r in reps if (v := get(r)) is not None]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return MetricStats(
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(arr.min()),
        max=float(arr.max()),
        n=len(values),
    )


def summarize_run(reps: Iterable[RepSummary]) -> CrossRepSummary:
    """Aggregate mean/std/min/max of headline metrics across one or more reps.

    Raises ValueError on an empty input; the contract requires >= 1 rep.
    """
    rep_list = list(reps)
    if not rep_list:
        raise ValueError("summarize_run requires at least one RepSummary")

    def dist_p(name: str, field: str) -> Callable[[RepSummary], float | None]:
        def get(r: RepSummary) -> float | None:
            d: Dist | None = getattr(r, name)
            return None if d is None else float(getattr(d, field))

        return get

    return CrossRepSummary(
        n_reps=len(rep_list),
        ttft_p50=_stats(rep_list, dist_p("ttft", "p50")),
        ttft_p90=_stats(rep_list, dist_p("ttft", "p90")),
        ttft_p99=_stats(rep_list, dist_p("ttft", "p99")),
        e2e_p50=_stats(rep_list, dist_p("e2e", "p50")),
        e2e_p99=_stats(rep_list, dist_p("e2e", "p99")),
        tpot_p50=_stats(rep_list, dist_p("tpot", "p50")),
        itl_p50=_stats(rep_list, dist_p("itl", "p50")),
        output_tokens_per_s=_stats(rep_list, lambda r: r.output_tokens_per_s),
        requests_per_s=_stats(rep_list, lambda r: r.requests_per_s),
        error_rate=_stats(rep_list, lambda r: r.error_rate),
    )
