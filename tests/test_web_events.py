# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import math
import socket
import threading
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest
from pydantic import ValidationError
from xr_ai_runtime import AgentRuntime
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent, WebEventsAgent


def _request(url: str, *, host: str | None = None) -> tuple[int, dict[str, str], bytes]:
    request = Request(url, headers={"Host": host}) if host is not None else url
    with urlopen(request, timeout=2) as response:  # noqa: S310
        return response.status, dict(response.headers.items()), response.read()


async def _get(url: str) -> tuple[int, dict[str, str], bytes]:
    return await asyncio.to_thread(_request, url)


async def _get_with_host(url: str, host: str) -> tuple[int, dict[str, str], bytes]:
    return await asyncio.to_thread(_request, url, host=host)


def test_web_event_contract_is_strict_and_untraced() -> None:
    event = WebEvent(topic=" monitor.changes ", title=" Changes ", payload={"changed": True})

    assert event.topic == "monitor.changes"
    assert event.title == "Changes"
    assert WEB_EVENT_TOPIC.message_type is WebEvent
    assert WEB_EVENT_TOPIC.telemetry == "none"
    with pytest.raises(ValidationError):
        WebEvent(topic=" ")
    with pytest.raises(ValidationError):
        WebEvent(topic="ok", unexpected=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"value": math.nan},
        {"value": math.inf},
        {"nested": [{"value": -math.inf}]},
    ],
)
def test_web_event_rejects_non_finite_json_numbers(payload: dict) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        WebEvent(topic="measurements", payload=payload)


def test_web_event_payload_limit_counts_serialized_utf8_bytes() -> None:
    accepted = WebEvent(topic="measurements", payload={"text": "x" * 16_373})

    assert len(accepted.payload["text"]) == 16_373
    with pytest.raises(ValidationError, match="16384 UTF-8 bytes"):
        WebEvent(topic="measurements", payload={"text": "x" * 16_374})
    with pytest.raises(ValidationError, match="16384 UTF-8 bytes"):
        WebEvent(topic="measurements", payload={"text": "€" * 5_458})


async def test_viewer_serves_runtime_events_and_reports_rollover() -> None:
    runtime = AgentRuntime()
    viewer = runtime.register(
        "web-events",
        WebEventsAgent(port=0, max_events=2, title="Test events"),
    )

    async with viewer:
        async with runtime:
            for value in (1, 2, 3):
                await runtime.publish(
                    WEB_EVENT_TOPIC,
                    WebEvent(
                        topic="instruments.reading",
                        title="Readings",
                        payload={"value": value},
                    ),
                    participant_id="glasses-1",
                    source="instrument-monitor",
                )

            status, headers, body = await _get(f"{viewer.url}/api/events?after=0")
            result = json.loads(body)
            assert status == 200
            assert result["title"] == "Test events"
            assert result["cursor"] == 3
            assert result["oldest"] == 2
            assert result["reset"] is True
            assert [event["sequence"] for event in result["events"]] == [2, 3]
            assert [event["payload"] for event in result["events"]] == [
                {"value": 2},
                {"value": 3},
            ]
            last = result["events"][-1]
            assert last["topic"] == "instruments.reading"
            assert last["title"] == "Readings"
            assert last["participant_id"] == "glasses-1"
            assert last["source"] == "instrument-monitor"
            assert last["message_id"] == last["correlation_id"]
            assert last["parent_message_id"] is None
            assert last["timestamp_us"] > 0
            assert headers["Cache-Control"] == "no-store"
            assert "default-src 'self'" in headers["Content-Security-Policy"]

            _, _, body = await _get(f"{viewer.url}/api/events?after=2")
            incremental = json.loads(body)
            assert incremental["reset"] is False
            assert [event["sequence"] for event in incremental["events"]] == [3]

            _, _, body = await _get(f"{viewer.url}/api/events?after=99")
            future = json.loads(body)
            assert future["reset"] is True
            assert [event["sequence"] for event in future["events"]] == [2, 3]


async def test_viewer_serves_health_and_packaged_page() -> None:
    viewer = WebEventsAgent(port=0)

    async with viewer:
        assert viewer.running
        status, headers, body = await _get(f"{viewer.url}/healthz")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}
        assert headers["X-Content-Type-Options"] == "nosniff"

        status, headers, body = await _get(f"{viewer.url}/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b'id="topic-grid"' in body

        status, _, body = await _get(f"{viewer.url}/app.js")
        assert status == 200
        assert b"/api/events?after=" in body

        with pytest.raises(HTTPError) as invalid:
            await _get(f"{viewer.url}/api/events?after=invalid")
        assert invalid.value.code == 400

    assert not viewer.running


async def test_viewer_rejects_unrecognized_host_before_serving_routes() -> None:
    viewer = WebEventsAgent(port=0)

    async with viewer:
        port = urlparse(viewer.url).port
        assert port is not None
        for host in (
            "localhost",
            f"localhost:{port}",
            "127.0.0.1",
            f"127.0.0.1:{port}",
        ):
            status, _, _ = await _get_with_host(f"{viewer.url}/healthz", host)
            assert status == 200

        for path in ("/healthz", "/api/events?after=0", "/"):
            with pytest.raises(HTTPError) as rejected:
                await _get_with_host(f"{viewer.url}{path}", "rebind.example")
            assert rejected.value.code == 400

        for host in ("[::1]", f"[::1]:{port}"):
            with pytest.raises(HTTPError) as rejected:
                await _get_with_host(f"{viewer.url}/healthz", host)
            assert rejected.value.code == 400


async def test_viewer_lifecycle_is_idempotent_and_bind_failure_is_visible() -> None:
    viewer = WebEventsAgent(port=0)
    await viewer.start()
    first_url = viewer.url
    await viewer.start()
    assert viewer.url == first_url
    await viewer.stop()
    repeated_stop = await viewer.stop()
    assert repeated_stop is None

    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        blocked = WebEventsAgent(port=port)
        with pytest.raises(OSError):
            await blocked.start()
        assert not blocked.running


async def test_viewer_cancellation_finishes_listener_cleanup(monkeypatch) -> None:
    viewer = WebEventsAgent(port=0)
    await viewer.start()
    server = viewer._server  # noqa: SLF001
    assert server is not None
    host, port = server.server_address[:2]
    shutdown_started = threading.Event()
    allow_shutdown = threading.Event()
    original_shutdown = server.shutdown

    def delayed_shutdown() -> None:
        shutdown_started.set()
        allow_shutdown.wait()
        original_shutdown()

    monkeypatch.setattr(server, "shutdown", delayed_shutdown)
    stop = asyncio.create_task(viewer.stop())
    await asyncio.wait_for(asyncio.to_thread(shutdown_started.wait), 1.0)
    stop.cancel()
    allow_shutdown.set()

    with pytest.raises(asyncio.CancelledError):
        await stop

    assert not viewer.running
    assert viewer._server is None  # noqa: SLF001
    assert viewer._thread is None  # noqa: SLF001
    with socket.socket() as rebound:
        rebound.bind((host, port))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"host": " "}, "host"),
        ({"host": "::1"}, "IPv4"),
        ({"port": -1}, "port"),
        ({"port": 65_536}, "port"),
        ({"max_events": 0}, "max_events"),
        ({"title": " "}, "title"),
    ],
)
def test_viewer_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WebEventsAgent(**kwargs)
