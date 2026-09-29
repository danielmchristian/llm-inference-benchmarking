"""Result presentation for llm-bench: text tables and the run plot.

All formatting and plotting lives here so the CLI stays thin. Durations are
reported in seconds everywhere, matching the record contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from llmbench.metrics import CrossRepSummary, Dist, MetricStats, RepSummary

# Categorical palette, colorblind-validated (worst adjacent CVD deltaE 24.2,
# above the >=12 target). Slots are assigned to reps in fixed order, never
# cycled; reps beyond the eighth fall back to the muted gray.
_PALETTE = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)
_MUTED = "#898781"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#fcfcfb"

_CROSS_METRICS: tuple[tuple[str, str], ...] = (
    ("ttft_p50_s", "ttft_p50"),
    ("ttft_p90_s", "ttft_p90"),
    ("ttft_p99_s", "ttft_p99"),
    ("e2e_p50_s", "e2e_p50"),
    ("e2e_p99_s", "e2e_p99"),
    ("tpot_p50_s", "tpot_p50"),
    ("itl_p50_s", "itl_p50"),
    ("output_tokens_per_s", "output_tokens_per_s"),
    ("requests_per_s", "requests_per_s"),
    ("error_rate", "error_rate"),
)


def _num(value: float) -> str:
    return f"{value:.4g}"


def _dist_field(dist: Dist | None, field: str) -> str:
    return "-" if dist is None else _num(float(getattr(dist, field)))


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Column-aligned plain-text table: first column left-, rest right-justified."""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        parts = [cells[0].ljust(widths[0])]
        parts.extend(c.rjust(w) for c, w in zip(cells[1:], widths[1:], strict=True))
        return "  ".join(parts).rstrip()

    return "\n".join([line(headers), *(line(row) for row in rows)])


def format_rep_line(summary: RepSummary) -> str:
    """One-line progress summary for a completed rep."""
    ttft = "-" if summary.ttft is None else f"{summary.ttft.p50:.4g}s"
    e2e = "-" if summary.e2e is None else f"{summary.e2e.p50:.4g}s"
    return (
        f"rep {summary.rep}: ok={summary.requests_ok} failed={summary.requests_failed} "
        f"error_rate={summary.error_rate:.3f} ttft_p50={ttft} e2e_p50={e2e} "
        f"out_tok/s={_num(summary.output_tokens_per_s)} "
        f"req/s={_num(summary.requests_per_s)} window={_num(summary.duration_s)}s"
    )


def format_rep_table(reps: Sequence[RepSummary]) -> str:
    """Per-rep table of headline metrics; '-' marks metrics with no data."""
    headers = [
        "rep",
        "ok",
        "fail",
        "err_rate",
        "ttft_p50_s",
        "ttft_p90_s",
        "ttft_p99_s",
        "e2e_p50_s",
        "e2e_p99_s",
        "tpot_p50_s",
        "itl_p50_s",
        "out_tokens",
        "out_tok/s",
        "req/s",
        "window_s",
    ]
    rows = [
        [
            str(s.rep),
            str(s.requests_ok),
            str(s.requests_failed),
            f"{s.error_rate:.3f}",
            _dist_field(s.ttft, "p50"),
            _dist_field(s.ttft, "p90"),
            _dist_field(s.ttft, "p99"),
            _dist_field(s.e2e, "p50"),
            _dist_field(s.e2e, "p99"),
            _dist_field(s.tpot, "p50"),
            _dist_field(s.itl, "p50"),
            str(s.output_tokens_total),
            _num(s.output_tokens_per_s),
            _num(s.requests_per_s),
            _num(s.duration_s),
        ]
        for s in reps
    ]
    return _table(headers, rows)


def format_cross_table(cross: CrossRepSummary) -> str:
    """Cross-rep mean/std/min/max table; '-' marks metrics no rep supplied.

    ``n_reps`` is per row: the number of reps that actually supplied the
    metric, which can be fewer than the run's rep count (e.g. a rep whose
    every request failed contributes no latency rows).
    """
    rows: list[list[str]] = []
    for label, attr in _CROSS_METRICS:
        stats: MetricStats | None = getattr(cross, attr)
        if stats is None:
            rows.append([label, "-", "-", "-", "-", "-"])
        else:
            rows.append(
                [
                    label,
                    _num(stats.mean),
                    _num(stats.std),
                    _num(stats.min),
                    _num(stats.max),
                    str(stats.n),
                ]
            )
    header = f"cross-rep summary ({cross.n_reps} reps)"
    return header + "\n" + _table(["metric", "mean", "std", "min", "max", "n_reps"], rows)


def format_manifest_summary(manifest: Mapping[str, Any]) -> str:
    """Human-readable run header: fingerprint, model, engine, hardware, workload, loadgen.

    All-null hardware is flagged explicitly rather than omitted, so a run
    recorded without ``--hardware-json`` is visible at review time.
    """
    model = manifest.get("model") or {}
    engine = manifest.get("engine") or {}
    hardware = manifest.get("hardware") or {}
    workload = manifest.get("workload") or {}
    loadgen = manifest.get("loadgen") or {}

    model_line = str(model.get("id", "?"))
    if model.get("revision"):
        model_line += f" @ {model['revision']}"
    engine_bits = [str(engine.get("name", "?"))]
    if engine.get("version"):
        engine_bits.append(str(engine["version"]))
    if engine.get("image"):
        engine_bits.append(f"image={engine['image']}")
    if engine.get("serving_args"):
        engine_bits.append("args=" + " ".join(str(a) for a in engine["serving_args"]))

    hw_keys = ("gpu_name", "gpu_count", "driver_version", "cuda_version", "vram_mib")
    if all(hardware.get(k) is None for k in hw_keys):
        hardware_line = "unknown (no hardware info recorded)"
    else:
        hw_bits: list[str] = []
        if hardware.get("gpu_name") is not None:
            hw_bits.append(str(hardware["gpu_name"]))
        if hardware.get("gpu_count") is not None:
            hw_bits.append(f"x{hardware['gpu_count']}")
        if hardware.get("driver_version") is not None:
            hw_bits.append(f"driver={hardware['driver_version']}")
        if hardware.get("cuda_version") is not None:
            hw_bits.append(f"cuda={hardware['cuda_version']}")
        if hardware.get("vram_mib") is not None:
            hw_bits.append(f"vram_mib={hardware['vram_mib']}")
        hardware_line = " ".join(hw_bits)

    return "\n".join(
        [
            f"fingerprint:  {manifest.get('fingerprint', '?')}",
            f"created_at:   {manifest.get('created_at', '?')}",
            f"model:        {model_line}",
            f"engine:       {' '.join(engine_bits)}",
            f"quantization: {manifest.get('quantization', '?')}",
            f"hardware:     {hardware_line}",
            (
                f"workload:     {workload.get('name', '?')} "
                f"(max_output_tokens={workload.get('max_output_tokens', '?')}, "
                f"temperature={workload.get('temperature', '?')}, "
                f"ignore_eos={workload.get('ignore_eos', '?')}, "
                f"seed={workload.get('seed', '?')})"
            ),
            (
                f"loadgen:      concurrency={loadgen.get('concurrency', '?')} "
                f"num_requests={loadgen.get('num_requests', '?')} "
                f"warmup={loadgen.get('warmup_requests', '?')} "
                f"repeats={loadgen.get('repeats', '?')} "
                f"timeout_s={loadgen.get('timeout_s', '?')}"
            ),
        ]
    )


def _panel(
    ax: Any,
    series: Sequence[tuple[int, Sequence[float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Draw one overlaid step-histogram panel with shared bins across reps."""
    combined = [v for _, values in series for v in values]
    n_drawn = 0
    if combined:
        edges = np.histogram_bin_edges(np.asarray(combined, dtype=np.float64), bins=30)
        for i, (rep, values) in enumerate(series):
            if not values:
                continue
            color = _PALETTE[i] if i < len(_PALETTE) else _MUTED
            ax.hist(
                values, bins=edges, histtype="step", linewidth=1.8, color=color, label=f"rep {rep}"
            )
            n_drawn += 1
    else:
        ax.text(
            0.5,
            0.5,
            "no successful measured requests",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=_INK_2,
        )

    ax.set_title(title, color=_INK)
    ax.set_xlabel(xlabel, color=_INK_2)
    ax.set_ylabel(ylabel, color=_INK_2)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_2)
    # A single series needs no legend box; the panel title carries it.
    if n_drawn >= 2:
        ax.legend(frameon=False, labelcolor=_INK_2)


def plot_run(
    rep_records: Sequence[tuple[int, Sequence[Mapping[str, Any]]]],
    out_path: Path | str,
    *,
    workload_name: str,
    fingerprint: str,
) -> None:
    """Write a PNG: TTFT and inter-token-latency histograms side by side, reps overlaid.

    Only OK, ``phase == "measure"`` records contribute, matching the metrics
    module. Bin edges are shared across reps within each panel so the overlays
    compare like with like. The figure renders 1200px wide.
    """
    # matplotlib is imported lazily so `llm-bench run` never pays its startup
    # cost; only `analyze --plot` reaches this function.
    import matplotlib

    matplotlib.use("Agg")  # off-screen rendering; this function only writes a file
    import matplotlib.pyplot as plt

    ttft_by_rep: list[tuple[int, list[float]]] = []
    itl_by_rep: list[tuple[int, list[float]]] = []
    for rep, records in rep_records:
        ok = [r for r in records if r.get("phase") == "measure" and r.get("ok")]
        ttft_by_rep.append((rep, [float(r["ttft_s"]) for r in ok if r.get("ttft_s") is not None]))
        itl_by_rep.append((rep, [float(g) for r in ok for g in r.get("itl_s") or []]))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=100, layout="constrained")
    fig.set_facecolor(_SURFACE)
    _panel(
        axes[0],
        ttft_by_rep,
        title="Time to first token",
        xlabel="TTFT (s)",
        ylabel="requests",
    )
    _panel(
        axes[1],
        itl_by_rep,
        title="Inter-token latency",
        xlabel="gap between content chunks (s)",
        ylabel="chunk gaps",
    )
    fig.suptitle(f"{workload_name} — {fingerprint[:12]}", color=_INK)
    fig.savefig(out_path, facecolor=_SURFACE)
    plt.close(fig)
