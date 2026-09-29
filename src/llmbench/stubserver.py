"""Deterministic OpenAI-compatible stub server for GPU-free calibration testing.

Serves /v1/chat/completions (streaming and non-streaming) with configurable,
seeded timing and error injection so the load generator's measurements can be
checked against known ground truth without any real model behind it.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import llmbench

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.requests import Request

_WORDS = ("alpha", "bravo", "delta", "echo", "gamma", "kilo", "sigma", "omega")


@dataclass(frozen=True)
class StubConfig:
    """Timing and failure knobs for the stub endpoint. All *_ms are milliseconds."""

    ttft_ms: float = 60
    ttft_jitter_ms: float = 0
    itl_ms: float = 15
    itl_jitter_ms: float = 0
    error_rate: float = 0.0
    seed: int = 0
    model_id: str = "stub-model"


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _prompt_tokens(payload: dict[str, Any]) -> int:
    """Whitespace token count over message contents; a stand-in, not a tokenizer."""
    total = 0
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += len(content.split())
    return total


def _token_text(i: int) -> str:
    word = _WORDS[i % len(_WORDS)]
    return word if i == 0 else f" {word}"


def create_app(config: StubConfig) -> Starlette:
    """Build the Starlette app implementing the OpenAI-compatible stub contract."""
    request_index = count()

    def _request_rng(index: int) -> random.Random:
        # Per-request RNG derived from (seed, arrival index): deterministic for a
        # given request order even when request handling interleaves.
        return random.Random((config.seed << 32) ^ (index + 1))

    def _delay_s(rng: random.Random, base_ms: float, jitter_ms: float) -> float:
        return max(0.0, base_ms + rng.uniform(-jitter_ms, jitter_ms)) / 1000.0

    def _chunk(
        req_id: str,
        created: int,
        delta: dict[str, str],
        finish_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": config.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    async def chat_completions(request: Request) -> Response:
        # Arrival index is assigned before any await so RNG use is ordered.
        index = next(request_index)
        rng = _request_rng(index)
        errored = rng.random() < config.error_rate
        payload: dict[str, Any] = await request.json()

        if errored:
            return JSONResponse({"error": {"message": "stub injected error"}}, status_code=500)

        raw_max = payload.get("max_tokens")
        n_tokens = 64 if raw_max is None else max(0, int(raw_max))
        stream = bool(payload.get("stream", False))
        stream_options = payload.get("stream_options") or {}
        include_usage = bool(stream_options.get("include_usage", False))
        prompt_tokens = _prompt_tokens(payload)

        # All delays are drawn up front, synchronously, to keep the RNG stream
        # a pure function of (config, arrival index).
        delays = [_delay_s(rng, config.ttft_ms, config.ttft_jitter_ms)] + [
            _delay_s(rng, config.itl_ms, config.itl_jitter_ms) for _ in range(max(0, n_tokens - 1))
        ]

        req_id = f"chatcmpl-stub-{index}"
        created = int(time.time())

        if not stream:
            await asyncio.sleep(sum(delays))
            text = "".join(_token_text(i) for i in range(n_tokens))
            return JSONResponse(
                {
                    "id": req_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": config.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _usage(prompt_tokens, n_tokens),
                }
            )

        async def event_stream() -> AsyncIterator[bytes]:
            # First chunk carries only the role, no content key (vLLM/OpenAI shape).
            yield _sse(_chunk(req_id, created, {"role": "assistant"}, None))
            for i in range(n_tokens):
                await asyncio.sleep(delays[i])
                finish = "stop" if i == n_tokens - 1 else None
                yield _sse(_chunk(req_id, created, {"content": _token_text(i)}, finish))
            if include_usage:
                yield _sse(
                    {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": config.model_id,
                        "choices": [],
                        "usage": _usage(prompt_tokens, n_tokens),
                    }
                )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async def models(request: Request) -> Response:
        return JSONResponse(
            {"object": "list", "data": [{"id": config.model_id, "object": "model"}]}
        )

    async def version(request: Request) -> Response:
        return JSONResponse({"version": f"stub-{llmbench.__version__}"})

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/models", models),
            Route("/version", version),
        ]
    )


def run_stub(config: StubConfig, host: str = "127.0.0.1", *, port: int) -> None:
    """Run the stub server under uvicorn; blocks until interrupted."""
    uvicorn.run(create_app(config), host=host, port=port, log_level="warning")
