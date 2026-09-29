"""Behavioral and calibration tests for the deterministic stub server."""

import json
import time
from typing import Any

import httpx

import llmbench
from llmbench.stubserver import StubConfig, create_app

FAST = StubConfig(ttft_ms=0, itl_ms=0)


def _client(config: StubConfig) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(config))
    return httpx.AsyncClient(transport=transport, base_url="http://stub")


def _events(text: str) -> list[dict[str, Any]]:
    """Parse SSE body into JSON events, excluding the [DONE] sentinel."""
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "hello there stub"}],
        "max_tokens": 5,
        "stream": True,
    }
    body.update(overrides)
    return body


async def test_stream_shape_role_first_then_exact_content_chunks() -> None:
    async with _client(FAST) as client:
        resp = await client.post("/v1/chat/completions", json=_chat_body(max_tokens=7))
    assert resp.status_code == 200
    assert resp.text.endswith("data: [DONE]\n\n")

    events = _events(resp.text)
    first_delta = events[0]["choices"][0]["delta"]
    assert first_delta == {"role": "assistant"}
    assert "content" not in first_delta

    content_deltas = [
        e["choices"][0]["delta"]
        for e in events
        if e["choices"] and "content" in e["choices"][0]["delta"]
    ]
    assert len(content_deltas) == 7
    assert all(d["content"].strip() for d in content_deltas)


async def test_stream_default_max_tokens_is_64() -> None:
    body = _chat_body()
    del body["max_tokens"]
    async with _client(FAST) as client:
        resp = await client.post("/v1/chat/completions", json=body)
    events = _events(resp.text)
    n_content = sum(1 for e in events if e["choices"] and "content" in e["choices"][0]["delta"])
    assert n_content == 64


async def test_stream_usage_chunk_only_when_requested() -> None:
    async with _client(FAST) as client:
        without = await client.post("/v1/chat/completions", json=_chat_body())
        with_usage = await client.post(
            "/v1/chat/completions",
            json=_chat_body(stream_options={"include_usage": True}),
        )

    assert all("usage" not in e for e in _events(without.text))

    events = _events(with_usage.text)
    usage_events = [e for e in events if "usage" in e]
    assert len(usage_events) == 1
    # Usage chunk is last before [DONE] and carries no choices.
    assert events[-1] is usage_events[0]
    assert usage_events[0]["choices"] == []
    usage = usage_events[0]["usage"]
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == usage["prompt_tokens"] + 5


async def test_stream_timing_honors_ttft_and_itl() -> None:
    config = StubConfig(ttft_ms=80, itl_ms=5)
    n = 10
    first_content_at: float | None = None
    async with _client(config) as client:
        start = time.perf_counter()
        async with client.stream(
            "POST", "/v1/chat/completions", json=_chat_body(max_tokens=n)
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[len("data: ") :])
                choices = event["choices"]
                if choices and "content" in choices[0]["delta"] and first_content_at is None:
                    first_content_at = time.perf_counter() - start
        total = time.perf_counter() - start

    assert first_content_at is not None
    # Generous CI tolerances: only lower bounds are asserted.
    assert first_content_at >= 0.06
    assert total >= (0.080 + (n - 1) * 0.005) * 0.5


async def test_non_stream_returns_complete_response_with_usage() -> None:
    async with _client(FAST) as client:
        resp = await client.post("/v1/chat/completions", json=_chat_body(stream=False))
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "stub-model"
    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"].split()) == 5
    assert data["usage"]["completion_tokens"] == 5
    # Prompt has three whitespace-delimited words.
    assert data["usage"]["prompt_tokens"] == 3
    assert data["usage"]["total_tokens"] == 8


async def test_non_stream_timing_lower_bound() -> None:
    config = StubConfig(ttft_ms=50, itl_ms=5)
    async with _client(config) as client:
        start = time.perf_counter()
        resp = await client.post(
            "/v1/chat/completions", json=_chat_body(stream=False, max_tokens=5)
        )
        elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert elapsed >= (0.050 + 4 * 0.005) * 0.5


async def test_error_injection_deterministic_and_near_rate() -> None:
    config = StubConfig(ttft_ms=0, itl_ms=0, error_rate=0.25, seed=7)

    async def statuses() -> list[int]:
        results: list[int] = []
        async with _client(config) as client:
            for _ in range(40):
                resp = await client.post("/v1/chat/completions", json=_chat_body(max_tokens=1))
                results.append(resp.status_code)
                if resp.status_code == 500:
                    assert resp.json() == {"error": {"message": "stub injected error"}}
        return results

    first, second = await statuses(), await statuses()
    assert first == second  # same config + same request order -> same injected errors
    n_errors = first.count(500)
    # Expected 10 of 40 at rate 0.25; wide bounds since it is a seeded sample.
    assert 2 <= n_errors <= 20


async def test_zero_error_rate_never_errors() -> None:
    async with _client(FAST) as client:
        for _ in range(10):
            resp = await client.post("/v1/chat/completions", json=_chat_body(max_tokens=1))
            assert resp.status_code == 200


async def test_unknown_body_fields_are_ignored() -> None:
    body = _chat_body(stream=False, ignore_eos=True, best_of=1, extra={"nested": [1, 2]})
    async with _client(FAST) as client:
        resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.json()["usage"]["completion_tokens"] == 5


async def test_models_endpoint() -> None:
    config = StubConfig(model_id="my-test-model")
    async with _client(config) as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json() == {
        "object": "list",
        "data": [{"id": "my-test-model", "object": "model"}],
    }


async def test_version_endpoint() -> None:
    async with _client(FAST) as client:
        resp = await client.get("/version")
    assert resp.status_code == 200
    assert resp.json() == {"version": f"stub-{llmbench.__version__}"}
