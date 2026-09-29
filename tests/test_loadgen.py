"""Calibration and I/O tests: run_load against the real stub app, in-process via ASGITransport."""

import asyncio
import contextlib
import json
import math
import re
import statistics
import time
from collections.abc import AsyncIterator, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

import llmbench
from llmbench import runio
from llmbench.loadgen import detect_engine_version, detect_model, run_load
from llmbench.runio import (
    append_records_jsonl,
    create_run_dir,
    read_records_jsonl,
    write_manifest,
)
from llmbench.stubserver import StubConfig, create_app


class _QueueByteStream(httpx.AsyncByteStream):
    """Response stream fed chunk-by-chunk from the ASGI app task; None terminates."""

    def __init__(self, queue: "asyncio.Queue[bytes | None]", task: "asyncio.Task[None]") -> None:
        self._queue = queue
        self._task = task

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk
        await self._task  # surface app exceptions instead of losing them

    async def aclose(self) -> None:
        if not self._task.done():
            self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


class StreamingASGITransport(httpx.ASGITransport):
    """ASGITransport that preserves the app's chunk timing.

    httpx 0.28's stock ASGITransport runs the app to completion and joins the
    whole body into one chunk, which would make measured TTFT equal E2E and
    every ITL gap ~0 -- calibration against the stub's known timing would be
    meaningless. This subclass forwards each body chunk as the app sends it.
    Still in-process, no sockets.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.AsyncByteStream)
        body = b"".join([chunk async for chunk in request.stream])
        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": self.client,
            "root_path": self.root_path,
        }
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        response_started = asyncio.Event()
        response_complete = asyncio.Event()
        status_code = 500
        response_headers: list[tuple[bytes, bytes]] = []
        request_sent = False

        async def receive() -> MutableMapping[str, Any]:
            nonlocal request_sent
            if request_sent:
                await response_complete.wait()
                return {"type": "http.disconnect"}
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code, response_headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = message.get("headers", [])
                response_started.set()
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if chunk:
                    await queue.put(chunk)
                if not message.get("more_body", False):
                    response_complete.set()
                    await queue.put(None)

        async def call_app() -> None:
            await self.app(scope, receive, send)

        def unblock(_task: "asyncio.Task[None]") -> None:
            # Unblock everything if the app dies before/while responding.
            response_started.set()
            response_complete.set()
            queue.put_nowait(None)

        app_task = asyncio.create_task(call_app())
        app_task.add_done_callback(unblock)
        await response_started.wait()
        if app_task.done() and (exc := app_task.exception()) is not None:
            raise exc
        return httpx.Response(
            status_code,
            headers=response_headers,
            stream=_QueueByteStream(queue, app_task),
        )


RECORD_KEYS = {
    "schema_version",
    "rep",
    "request_id",
    "phase",
    "ok",
    "error",
    "status_code",
    "start_unix",
    "start_mono_s",
    "e2e_s",
    "ttft_s",
    "itl_s",
    "prompt_tokens",
    "output_tokens",
    "tokens_source",
    "tpot_s",
}


def _stub_client(config: StubConfig) -> httpx.AsyncClient:
    transport = StreamingASGITransport(app=create_app(config))
    return httpx.AsyncClient(transport=transport, base_url="http://stub")


async def _run(
    config: StubConfig,
    *,
    n_requests: int,
    max_tokens: int = 8,
    concurrency: int = 2,
    warmup: int = 0,
    timeout_s: float = 10.0,
) -> list[dict[str, Any]]:
    messages = [[{"role": "user", "content": f"prompt number {i}"}] for i in range(n_requests)]
    async with _stub_client(config) as client:
        return await run_load(
            "http://stub",
            config.model_id,
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            ignore_eos=True,
            concurrency=concurrency,
            warmup_requests=warmup,
            timeout_s=timeout_s,
            rep=1,
            client=client,
        )


async def test_calibration_against_known_stub_timing() -> None:
    """Stub with ttft 50ms / itl 5ms, zero jitter: measurements must land near truth."""
    config = StubConfig(ttft_ms=50, ttft_jitter_ms=0, itl_ms=5, itl_jitter_ms=0)
    records = await _run(config, n_requests=6, max_tokens=8, concurrency=3)

    assert len(records) == 6
    assert all(r["ok"] for r in records)
    assert all(r["phase"] == "measure" for r in records)
    assert all(r["status_code"] == 200 for r in records)
    assert all(set(r) == RECORD_KEYS for r in records)
    assert [r["request_id"] for r in records] == list(range(6))

    ttft_p50 = statistics.median(r["ttft_s"] for r in records)
    assert 0.03 <= ttft_p50 <= 0.2

    gaps = [gap for r in records for gap in r["itl_s"]]
    assert len(gaps) == 6 * (8 - 1)
    mean_itl = sum(gaps) / len(gaps)
    assert 0.002 <= mean_itl <= 0.02

    for r in records:
        assert r["error"] is None
        assert r["output_tokens"] == 8  # exactly the requested max_tokens
        assert r["tokens_source"] == "usage"
        # Stub counts prompt tokens as whitespace words: "prompt number <i>".
        assert r["prompt_tokens"] == 3
        assert r["e2e_s"] > r["ttft_s"] > 0
        assert r["start_mono_s"] > 0  # monotonic start recorded for window math
        assert math.isclose(r["tpot_s"], (r["e2e_s"] - r["ttft_s"]) / 7, rel_tol=1e-9)


async def test_warmup_completes_before_measurement_starts() -> None:
    config = StubConfig(ttft_ms=20, ttft_jitter_ms=0, itl_ms=1, itl_jitter_ms=0)
    records = await _run(config, n_requests=8, max_tokens=4, concurrency=4, warmup=3)

    warmup = [r for r in records if r["phase"] == "warmup"]
    measured = [r for r in records if r["phase"] == "measure"]
    assert len(warmup) == 3
    assert len(measured) == 5
    assert [r["request_id"] for r in records] == list(range(8))
    assert all(r["ok"] for r in records)

    first_measured_start = min(r["start_unix"] for r in measured)
    assert all(r["start_unix"] < first_measured_start for r in warmup)
    warmup_end = max(r["start_unix"] + r["e2e_s"] for r in warmup)
    # 5ms epsilon absorbs wall-clock vs perf_counter drift in start_unix + e2e_s.
    assert warmup_end <= first_measured_start + 0.005


async def test_error_injection_records_failures_without_raising() -> None:
    config = StubConfig(ttft_ms=0, itl_ms=0, error_rate=0.5, seed=11)
    records = await _run(config, n_requests=20, max_tokens=3, concurrency=1)

    assert len(records) == 20
    failures = [r for r in records if not r["ok"]]
    successes = [r for r in records if r["ok"]]
    assert failures
    assert successes

    for r in failures:
        assert r["status_code"] == 500
        assert isinstance(r["error"], str)
        assert "500" in r["error"]
        assert r["ttft_s"] is None
        assert r["itl_s"] == []
        assert r["output_tokens"] is None
        assert r["tokens_source"] is None
        assert r["tpot_s"] is None
    for r in successes:
        assert r["error"] is None
        assert r["output_tokens"] == 3
        assert r["tokens_source"] == "usage"
        assert r["e2e_s"] is not None


async def test_timeout_is_overall_request_deadline() -> None:
    """A steadily trickling stream must be bounded by timeout_s.

    Each inter-chunk gap (25ms) is far below the 0.3s timeout, so httpx's
    per-read timeout (which resets on every chunk) would never fire; only an
    overall per-request deadline stops the request.
    """
    config = StubConfig(ttft_ms=5, ttft_jitter_ms=0, itl_ms=25, itl_jitter_ms=0)
    t_start = time.perf_counter()
    records = await _run(config, n_requests=1, max_tokens=200, concurrency=1, timeout_s=0.3)
    elapsed = time.perf_counter() - t_start

    (record,) = records
    assert record["ok"] is False
    assert record["error"] == "timeout"
    assert record["e2e_s"] is None
    assert record["ttft_s"] is not None  # best-effort timing from chunks seen pre-deadline
    # Without a deadline the stream runs ~5s (200 chunks x 25ms); generous CI margin.
    assert elapsed < 3.0


async def test_negative_warmup_is_clamped_to_zero() -> None:
    """warmup_requests=-1 must not slice requests into warmup or emit request_id=-1."""
    config = StubConfig(ttft_ms=0, itl_ms=0)
    records = await _run(config, n_requests=2, max_tokens=2, concurrency=1, warmup=-1)
    assert [r["request_id"] for r in records] == [0, 1]
    assert all(r["phase"] == "measure" for r in records)


async def test_run_load_rejects_nonpositive_concurrency() -> None:
    """concurrency=0 must raise instead of deadlocking on an unacquirable semaphore."""
    with pytest.raises(ValueError, match="concurrency"):
        await run_load(
            "http://stub",
            "model",
            [],
            max_tokens=1,
            temperature=0.0,
            ignore_eos=False,
            concurrency=0,
            warmup_requests=0,
            timeout_s=1.0,
            rep=1,
        )


async def test_detect_model_and_engine_version() -> None:
    config = StubConfig(model_id="calibration-model")
    async with _stub_client(config) as client:
        model = await detect_model("http://stub", client)
        version = await detect_engine_version("http://stub", client)
        missing = await detect_engine_version("http://stub/nowhere", client)
    assert model == "calibration-model"
    assert version == f"stub-{llmbench.__version__}"
    assert missing is None


def test_runio_round_trip(tmp_path: Path) -> None:
    fingerprint = "ab12" * 16
    run_dir = create_run_dir(tmp_path / "runs", fingerprint)
    assert run_dir.parent == tmp_path / "runs"
    assert re.fullmatch(r"ab12ab12ab12-\d{8}T\d{6}Z", run_dir.name)

    manifest = {"schema_version": 1, "fingerprint": fingerprint, "notes": None}
    manifest_path = write_manifest(run_dir, manifest)
    assert manifest_path == run_dir / "manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    records: list[dict[str, Any]] = [
        {"schema_version": 1, "rep": 1, "request_id": 0, "ok": True, "itl_s": [0.1, 0.2]},
        {"schema_version": 1, "rep": 1, "request_id": 1, "ok": False, "error": "timeout"},
    ]
    rep_path = run_dir / "rep1.jsonl"
    append_records_jsonl(rep_path, records[:1])
    append_records_jsonl(rep_path, records[1:])
    assert read_records_jsonl(rep_path) == records


def test_create_run_dir_same_second_collision_gets_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs of one fingerprint in the same second must both get a directory."""

    class _FrozenDatetime:
        """Stands in for runio's datetime so every call lands in the same second."""

        @staticmethod
        def now(tz: timezone) -> datetime:
            return datetime(2026, 9, 28, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(runio, "datetime", _FrozenDatetime)
    fingerprint = "ab" * 32
    first = create_run_dir(tmp_path / "runs", fingerprint)
    second = create_run_dir(tmp_path / "runs", fingerprint)
    third = create_run_dir(tmp_path / "runs", fingerprint)
    assert first.name == "abababababab-20260928T120000Z"
    assert second.name == first.name + "-2"
    assert third.name == first.name + "-3"
    assert second.is_dir()
    assert third.is_dir()


def test_read_records_jsonl_truncated_line_raises_with_context(tmp_path: Path) -> None:
    """A rep file cut mid-record (killed writer) must fail with file and line, not a raw trace."""
    path = tmp_path / "rep1.jsonl"
    path.write_text('{"ok": true}\n{"ok": fal', encoding="utf-8")
    with pytest.raises(ValueError, match=r"rep1\.jsonl: line 2"):
        read_records_jsonl(path)


async def test_loadgen_records_round_trip_through_runio(tmp_path: Path) -> None:
    """Every field of a real record must survive a JSONL write/read unchanged."""
    config = StubConfig(ttft_ms=0, itl_ms=0)
    records = await _run(config, n_requests=2, max_tokens=2, concurrency=1)
    path = tmp_path / "rep1.jsonl"
    append_records_jsonl(path, records)
    assert read_records_jsonl(path) == records
