# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upstream path selection in the /rtc reverse proxy."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import asynccontextmanager

import httpx
import pytest
from device_io_hub.transport.livekit import _lk_proxy
from device_io_hub.transport.livekit._lk_proxy import mount_rtc_proxy
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def upstream_urls() -> list[str]:
    return []


@pytest.fixture
def client(upstream_urls: list[str]) -> Iterator[TestClient]:
    def upstream(request: httpx.Request) -> httpx.Response:
        upstream_urls.append(str(request.url))
        return httpx.Response(200, text="ok")

    app = FastAPI()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    mount_rtc_proxy(
        app,
        client=http_client,
        lk_internal_http="http://lk.internal:7880",
        lk_internal_ws="ws://lk.internal:7880",
    )
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        asyncio.run(http_client.aclose())


@pytest.fixture
def ws_targets(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    targets: list[str] = []

    class _ClosedUpstream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self) -> None:
            pass

    @asynccontextmanager
    async def fake_connect(target: str, **_kwargs):
        targets.append(target)
        yield _ClosedUpstream()

    monkeypatch.setattr(_lk_proxy.websockets, "connect", fake_connect)
    return targets


@pytest.mark.parametrize(
    ("route", "upstream"),
    [
        ("/rtc/validate", "http://lk.internal:7880/rtc/validate?access_token=t"),
        ("/rtc/v1/validate", "http://lk.internal:7880/rtc/v1/validate?access_token=t"),
    ],
)
def test_validate_forwards_known_segments_with_query(
    client: TestClient, upstream_urls: list[str], route: str, upstream: str
) -> None:
    response = client.get(route, params={"access_token": "t"})

    assert response.status_code == 200
    assert upstream_urls == [upstream]


UNKNOWN_SEGMENTS = ["v9", "v1/../admin", "v1%2F..%2Fadmin"]


@pytest.mark.parametrize("segment", UNKNOWN_SEGMENTS)
def test_validate_rejects_unknown_segment(client: TestClient, upstream_urls: list[str], segment: str) -> None:
    response = client.get(f"/rtc/{segment}/validate")

    assert response.status_code == 404
    assert upstream_urls == []


@pytest.mark.parametrize(
    ("route", "upstream"),
    [
        ("/rtc", "ws://lk.internal:7880/rtc?access_token=t"),
        ("/rtc/v1", "ws://lk.internal:7880/rtc/v1?access_token=t"),
    ],
)
def test_websocket_forwards_known_segments_with_query(
    client: TestClient, ws_targets: list[str], route: str, upstream: str
) -> None:
    with client.websocket_connect(f"{route}?access_token=t"):
        pass

    assert ws_targets == [upstream]


@pytest.mark.parametrize("segment", UNKNOWN_SEGMENTS)
def test_websocket_refuses_unknown_segment(client: TestClient, ws_targets: list[str], segment: str) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/rtc/{segment}"):
            pass

    assert ws_targets == []
