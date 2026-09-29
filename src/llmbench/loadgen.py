"""Async load generator for OpenAI-compatible chat completion endpoints.

Drives streaming ``/v1/chat/completions`` requests at a fixed concurrency and
emits one record dict per request (the run record contract). All latencies are
measured with ``time.perf_counter``; wall-clock time appears only as
``start_unix``. Individual request failures never raise out of ``run_load`` --
every request yields exactly one record.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import httpx

SCHEMA_VERSION = 1

Messages = list[dict[str, str]]


def _new_record(rep: int, request_id: int, phase: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rep": rep,
        "request_id": request_id,
        "phase": phase,
        "ok": False,
        "error": None,
        "status_code": None,
        "start_unix": 0.0,
        "start_mono_s": 0.0,
        "e2e_s": None,
        "ttft_s": None,
        "itl_s": [],
        "prompt_tokens": None,
        "output_tokens": None,
        "tokens_source": None,
        "tpot_s": None,
    }


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    *,
    rep: int,
    request_id: int,
    phase: str,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Send one streaming chat request and return its record; never raises.

    A CONTENT chunk is an SSE event whose ``choices[0].delta.content`` is a
    non-empty string; the role-only first chunk and the usage-only chunk are
    not content chunks, so they never count toward TTFT or ITL.

    ``timeout_s`` is an overall per-request deadline covering connect through
    ``[DONE]``. httpx's own timeout resets on every received chunk, so a
    trickling or keepalive-emitting stream would otherwise never time out.
    """
    record = _new_record(rep, request_id, phase)
    content_times: list[float] = []
    usage: dict[str, Any] | None = None
    saw_done = False
    error: str | None = None
    end: float | None = None

    t0 = time.perf_counter()
    record["start_unix"] = time.time()
    record["start_mono_s"] = t0
    try:
        async with asyncio.timeout(timeout_s):
            async with client.stream("POST", url, json=body) as resp:
                record["status_code"] = resp.status_code
                if resp.status_code != 200:
                    await resp.aread()
                    error = f"HTTP {resp.status_code}"
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line.removeprefix("data: ")
                        if data == "[DONE]":
                            saw_done = True
                            break
                        event = json.loads(data)
                        choices = event.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                content_times.append(time.perf_counter())
                        if event.get("usage") is not None:
                            usage = event["usage"]
                    if not saw_done:
                        error = "stream ended without [DONE]"
        end = time.perf_counter()
    except (TimeoutError, httpx.TimeoutException):
        error = "timeout"
    except json.JSONDecodeError:
        error = "malformed SSE event"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:200]

    # Best-effort timings: recorded even on failure when they were observed.
    if content_times:
        record["ttft_s"] = content_times[0] - t0
        record["itl_s"] = [b - a for a, b in pairwise(content_times)]
    if end is not None:
        record["e2e_s"] = end - t0

    if error is not None:
        record["error"] = error
        return record

    record["ok"] = True
    completion_tokens = usage.get("completion_tokens") if usage is not None else None
    if isinstance(completion_tokens, int):
        record["output_tokens"] = completion_tokens
        record["tokens_source"] = "usage"
    else:
        record["output_tokens"] = len(content_times)
        record["tokens_source"] = "chunk_count"
    prompt_tokens = usage.get("prompt_tokens") if usage is not None else None
    if isinstance(prompt_tokens, int):
        record["prompt_tokens"] = prompt_tokens

    ttft = record["ttft_s"]
    if ttft is not None and record["output_tokens"] > 1:
        record["tpot_s"] = (record["e2e_s"] - ttft) / (record["output_tokens"] - 1)
    return record


async def run_load(
    base_url: str,
    model: str,
    messages_batch: Sequence[Messages],
    *,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
    concurrency: int,
    warmup_requests: int,
    timeout_s: float,
    rep: int,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Drive one rep of streaming chat requests; return one record per request.

    The first ``min(warmup_requests, len(messages_batch))`` entries of
    ``messages_batch`` run as ``phase="warmup"`` and ALL complete before any
    ``phase="measure"`` request starts, keeping cold-start effects out of the
    measurement window (negative ``warmup_requests`` is clamped to 0).
    ``request_id`` is the 0-based position within the rep, warmup included.
    ``t0`` is taken after the concurrency slot is acquired, so client-side
    queueing never inflates latency. ``timeout_s`` is enforced as an overall
    per-request deadline (connect through ``[DONE]``), independent of httpx's
    per-read timeout. Owns (and closes) an httpx client sized for
    ``concurrency`` with ``timeout_s`` unless ``client`` is injected, in which
    case the injected client's timeout governs transport-level reads while the
    deadline still applies. Raises ValueError when ``concurrency < 1`` (a zero
    semaphore would deadlock forever).
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    owns_client = client is None
    active = (
        httpx.AsyncClient(
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        )
        if client is None
        else client
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def send(request_id: int, phase: str, messages: Messages) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if ignore_eos:
            # vLLM extension for fixed output length; the stub ignores it.
            body["ignore_eos"] = True
        async with semaphore:
            return await _one_request(
                active,
                url,
                body,
                rep=rep,
                request_id=request_id,
                phase=phase,
                timeout_s=timeout_s,
            )

    n_warmup = min(max(warmup_requests, 0), len(messages_batch))
    try:
        warmup = await asyncio.gather(
            *(send(i, "warmup", m) for i, m in enumerate(messages_batch[:n_warmup]))
        )
        measured = await asyncio.gather(
            *(send(n_warmup + i, "measure", m) for i, m in enumerate(messages_batch[n_warmup:]))
        )
    finally:
        if owns_client:
            await active.aclose()
    return [*warmup, *measured]


async def detect_model(base_url: str, client: httpx.AsyncClient | None = None) -> str:
    """Return the first model id listed by ``GET {base_url}/v1/models``.

    Raises (httpx errors or ValueError) when the endpoint is unreachable,
    non-200, or lists no models.
    """
    owns_client = client is None
    active = httpx.AsyncClient() if client is None else client
    try:
        resp = await active.get(f"{base_url.rstrip('/')}/v1/models")
        resp.raise_for_status()
        models = resp.json().get("data") or []
        if not models:
            raise ValueError(f"no models listed at {base_url}")
        return str(models[0]["id"])
    finally:
        if owns_client:
            await active.aclose()


async def detect_engine_version(
    base_url: str, client: httpx.AsyncClient | None = None
) -> str | None:
    """Return the ``version`` string from ``GET {base_url}/version``, or None on any failure."""
    owns_client = client is None
    active = httpx.AsyncClient() if client is None else client
    try:
        resp = await active.get(f"{base_url.rstrip('/')}/version")
        if resp.status_code != 200:
            return None
        version = resp.json().get("version")
        return version if isinstance(version, str) else None
    except Exception:
        return None
    finally:
        if owns_client:
            await active.aclose()
