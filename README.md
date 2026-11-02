# llm-bench

<!-- CI badge: add after the repo is pushed to GitHub -->

Benchmark LLM serving engines across quantization formats and batching
configurations — and measure what those optimizations do to answer quality.

Most public inference benchmarks report throughput and latency but never check
whether the optimized model still gives the same answers. llm-bench treats
reproducibility as the product: every benchmark run emits a manifest (model
revision, engine version, quantization format, hardware, every serving flag)
fingerprinted with SHA-256; raw per-request results are written as JSONL and
published; serving engines are pinned by Docker image digest; and the
harness's own dependencies are locked (`uv.lock`, enforced in CI). Two runs
with the same fingerprint are directly comparable — anything else is visibly
different.

## Status

**Pre-alpha — Phase 1 complete.** The harness is calibration-tested against a
stub server with known timing, and the first real baseline is measured and
published: Llama-3.1-8B-Instruct (bf16) under vLLM 0.32.0 on an RTX 4090 —
see **[RESULTS.md](RESULTS.md)**. This table links to results as each phase
lands.

| Phase | Scope | Status |
|------:|-------|--------|
| 0 | Repo scaffolding, CI, decision log | done |
| 1 | Load generator + manifest fingerprinting; vLLM bf16 baseline, one workload | done — [results](RESULTS.md) |
| 2 | Engine comparison: vLLM vs SGLang vs a third engine, same model and hardware | planned |
| 3 | Quantization: FP16 vs FP8 vs AWQ vs GPTQ — throughput, latency, memory | planned |
| 4 | Quality regression: performance gains vs answer-quality cost | planned |
| 5 | Batching and concurrency sweep: the throughput-vs-latency curve | planned |
| 6 | Observability: Prometheus exporter + Grafana dashboard | planned |
| 7 | Kubernetes + multi-GPU tensor parallelism (optional) | planned |

## Architecture

The harness is a **client**: it drives any OpenAI-compatible HTTP endpoint and
never links against serving-engine internals. Engines run in Docker on the GPU
box (a rented RTX 4090); the harness runs anywhere — including against a local
stub server, so no GPU time is ever spent debugging the harness itself.

Planned module map (each module is created in the phase that needs it):

| Module | Purpose |
|--------|---------|
| `llmbench.loadgen` | Async load generator driving OpenAI-compatible endpoints |
| `llmbench.manifest` | Run manifest + SHA-256 config fingerprinting |
| `llmbench.metrics` | Per-request and aggregate stats (TTFT, TPOT, percentiles) |
| `llmbench.workloads` | Workload definitions and token distributions |
| `llmbench.stubserver` | Local OpenAI-compatible stub for GPU-free testing |
| `llmbench.engines` | Per-engine adapters (`vllm`, later `sglang`, …) |
| `llmbench.analysis` | Aggregation and plot generation from results data |
| `llmbench.cli` | The `llm-bench` command-line entry point |

## Development quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked   # create .venv and install pinned dev deps
make check         # lint + typecheck + tests — exactly what CI runs
```

Try the whole pipeline GPU-free against the bundled stub server (it streams
fake tokens with configurable, known timing — which is also how the harness
itself is calibration-tested):

```sh
uv run llm-bench stub --port 18080 --ttft-ms 120 --itl-ms 8 &
uv run llm-bench run --base-url http://127.0.0.1:18080 \
  --workload chat-short --concurrency 8 --num-requests 64 --repeats 3 --out runs
uv run llm-bench analyze runs/<RUN_ID> --plot runs/<RUN_ID>/latency.png
```

See `docs/metrics.md` for what each number means and how it is measured, and
`docs/phase1-runbook.md` for the real-GPU procedure.

## License

Code is licensed under [Apache-2.0](LICENSE). The benchmarked model
(`meta-llama/Llama-3.1-8B-Instruct`) is distributed under the Llama 3.1
Community License and is gated on Hugging Face; this repository contains no
model weights.
