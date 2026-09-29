# Metrics

What llm-bench measures, exactly how each number is computed, and where the
honest caveats are. All durations are seconds (float) unless a name ends in
`_ms`.

## The request timeline

Every benchmark request is a streaming `POST /v1/chat/completions`
(`stream: true`, `stream_options: {"include_usage": true}`). The load
generator records, per request:

- `t0` — `time.perf_counter()` taken immediately before the request is sent
  (after the concurrency slot is acquired, so client-side queueing never
  inflates latency). It is stored in the record as `start_mono_s`; wall-clock
  `time.time()` is captured at the same moment as `start_unix` (for humans —
  no computation uses it when `start_mono_s` is present).
- Arrival times of every **content chunk** — an SSE event whose
  `choices[0].delta.content` is a non-empty string. The role-only first chunk
  and the usage-only final chunk are not content chunks and never count.
- Stream end — when `data: [DONE]` is received.

## The four latency metrics

**TTFT (time to first token)** — first content chunk arrival minus `t0`.
This includes connection handling, server-side queueing, prompt processing
(prefill), and scheduling. Under load, queueing usually dominates TTFT; it is
the metric a user experiences as "how long until the reply starts".

**ITL (inter-token latency)** — the gaps between successive content chunk
arrivals after the first. A request with N content chunks contributes N-1
gaps; per-rep distributions flatten the gaps across all successful requests.
ITL is what a user experiences as the streaming rhythm; its tail (p99)
exposes decode stalls.

**TPOT (time per output token)** — `(e2e_s - ttft_s) / (output_tokens - 1)`,
computed only when the request succeeded, TTFT was observed, and
`output_tokens > 1`. TPOT is the *average* decode rate over the whole request
using the authoritative token count. Unlike mean ITL, it is immune to the
chunk-vs-token caveat below, which is why both are reported.

**E2E (end-to-end latency)** — stream end minus `t0`; the full request
duration.

## The chunk-vs-token caveat

ITL gaps are measured between **SSE chunks, not tokens**. vLLM streams one
token per chunk by default, so the two usually coincide — but nothing
guarantees it: an engine may coalesce several tokens into one chunk (common
under heavy load or with buffering proxies), which makes chunk gaps look
larger and fewer than the true per-token gaps.

Token *counts* therefore never come from counting chunks when better data
exists: `output_tokens` is taken from `usage.completion_tokens`
(`tokens_source: "usage"`), which the server computed with its own tokenizer.
Counting content chunks is only a fallback (`tokens_source: "chunk_count"`)
when the server does not report usage. Treat ITL from an endpoint that
coalesces chunks as a lower-bound rhythm measurement, and lean on TPOT for
per-token decode speed.

## Measurement window and throughput

Each rep's throughput is computed over the **measurement window**:

```
duration_s = max(start_mono_s + e2e_s) - min(start_mono_s)   over OK measured requests
output_tokens_per_s = sum(output_tokens) / duration_s
requests_per_s      = requests_ok / duration_s
```

The window runs from the first measured request's start to the last measured
request's end. Both endpoints live in the monotonic `perf_counter` domain —
the same clock `e2e_s` is measured with — so an NTP wall-clock step during a
rep cannot corrupt throughput. (`start_unix` is used as a fallback only for
records written before `start_mono_s` existed.) This is the standard way to report offered-load throughput: it
charges the system for idle tails and ramp-down, rather than summing
per-request rates (which overstates throughput whenever requests overlap).
Failed requests contribute nothing to the window or the totals.

## Failures and error_rate

A request is failed (`ok: false`) on any non-200 status, timeout, connection
failure, or malformed stream. `timeout_s` is enforced as an **overall
per-request deadline** — connect through `[DONE]` — not merely as httpx's
per-read timeout (which resets on every received chunk and so would never
fire against a slowly trickling or keepalive-emitting stream). Failed requests are **excluded** from every
latency and throughput statistic — a request that errors in 5ms would
otherwise make the system look faster — but are **included** in
`error_rate = failed / total measured`. A result with a nonzero error rate is
not comparable to a clean run and says so.

## Warmup

The first `warmup_requests` requests of every rep are flagged
`phase: "warmup"` and all complete before the first measured request starts.
They are excluded from every statistic. First requests pay one-time costs that
say nothing about steady state: connection establishment and TLS, server-side
lazy initialization, CUDA graph capture and kernel autotuning, cold KV-cache
allocators, and empty prefix caches. Discarding warmup keeps the measurement
window steady-state by construction rather than by hope.

## Fixed output length (ignore_eos)

Performance workloads set `ignore_eos: true` (a vLLM extension) so the engine
generates exactly `max_output_tokens` tokens instead of stopping at a natural
end-of-sequence. Without it, output length becomes a property of the model and
sampling settings — a quantized model that happens to write shorter answers
would "win" on every latency and throughput metric while doing less work.
Fixing decode length makes TPOT, ITL, and tokens/s comparable across engines,
quantizations, and configs. The trade-off is honest and documented: fixed
length measures serving performance, not answer quality (quality is a separate
concern with its own evaluation).

## Percentiles and spread

Distributions report p50/p90/p99, mean, population standard deviation
(`ddof=0`), and sample count `n`. Percentiles are computed with
`numpy.quantile` using its default **linear interpolation** between order
statistics; a reported p99 of an n=256 sample interpolates between the two
highest-ranked observations rather than snapping to one of them. Cross-rep
summaries report mean/std/min/max of each rep-level metric across repeats —
the honest way to show run-to-run variance instead of pooling everything into
one flattering distribution. Each cross-rep row also reports `n_reps`, the
number of reps that actually supplied that metric: a rep whose every request
failed contributes nothing to the latency rows (though it still counts in
`error_rate` and throughput), so different rows can cover different rep
subsets — and now say so explicitly instead of implying the full rep count.
