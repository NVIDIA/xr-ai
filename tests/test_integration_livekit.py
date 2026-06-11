# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the LiveKit transport.

These tests boot a real ``livekit/livekit-server`` Docker container and
connect a real :class:`RoomClient` to it. They are tagged ``gpu`` so they
only run on a developer box where Docker is available — GitHub CI skips
the marker entirely.

The first test exercises :class:`LiveKitDocker` end-to-end (start, port
opens, stop, port closes, container gone). The second reuses a live
container via a module-scoped fixture and verifies that
``RoomClient.connect()`` reaches the ``CONN_CONNECTED`` state and that
``disconnect()`` tears down cleanly.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid
from dataclasses import replace
from typing import AsyncIterator

import pytest
from livekit import rtc

from _helpers_subprocess import pick_free_port
from xr_media_hub.ipc                       import ConnectorEndpoint
from xr_media_hub.transport.livekit         import _docker as _docker_mod
from xr_media_hub.transport.livekit._docker import LiveKitDocker
from xr_media_hub.transport.livekit._room_client import RoomClient
from xr_media_hub.transport.livekit.config       import LiveKitConnectorConfig

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]


_LIVEKIT_IMAGE   = "livekit/livekit-server:latest"
_PORT_OPEN_WAIT  = 30.0   # matches _docker._READY_TIMEOUT
_PORT_CLOSE_WAIT = 10.0


# ── helpers ──────────────────────────────────────────────────────────────────


async def _wait_port_open(port: int, timeout: float) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return True
        except OSError:
            await asyncio.sleep(0.2)
    return False


async def _wait_port_closed(port: int, timeout: float) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.2)
        except OSError:
            return True
    return False


def _container_running(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return name in out.decode().split()


def _build_cfg() -> LiveKitConnectorConfig:
    """Build a config bound to free ports and a unique room name."""
    lk_ws  = pick_free_port(7880)
    lk_tcp = pick_free_port(7881)
    lk_udp = pick_free_port(7882)
    return LiveKitConnectorConfig(
        lk_port_ws=lk_ws,
        lk_port_tcp=lk_tcp,
        lk_port_udp=lk_udp,
        lk_internal_url=f"ws://127.0.0.1:{lk_ws}",
        room_name=f"xr-test-{uuid.uuid4().hex[:8]}",
    )


# ── module setup ─────────────────────────────────────────────────────────────


pytest.importorskip("livekit")

if shutil.which("docker") is None:
    pytest.skip("docker not on PATH — LiveKit integration skipped",
                allow_module_level=True)


def _docker_daemon_alive() -> bool:
    try:
        subprocess.check_call(
            ["docker", "info"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


if not _docker_daemon_alive():
    pytest.skip("docker daemon not reachable — LiveKit integration skipped",
                allow_module_level=True)


# Pre-pull the LiveKit image once so the 30s ready-timeout in _docker.py
# is not eaten by an image pull on first invocation.
subprocess.run(
    ["docker", "pull", _LIVEKIT_IMAGE],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)


# ── tests ────────────────────────────────────────────────────────────────────


async def test_livekit_docker_lifecycle(monkeypatch):
    """Start the container, assert the WS port opens, stop, assert it closes.

    Uses a unique container name so concurrent test runs don't collide on the
    module-level ``_CONTAINER_NAME`` default.
    """
    name = f"xr-ai-livekit-test-{uuid.uuid4().hex[:10]}"
    monkeypatch.setattr(_docker_mod, "_CONTAINER_NAME", name)
    cfg = _build_cfg()
    docker = LiveKitDocker(cfg)

    await docker.start()
    try:
        assert await _wait_port_open(cfg.lk_port_ws, _PORT_OPEN_WAIT), (
            f"LiveKit WS port {cfg.lk_port_ws} never opened"
        )
        assert _container_running(name), f"container {name!r} not listed by docker ps"
    finally:
        await docker.stop()

    assert await _wait_port_closed(cfg.lk_port_ws, _PORT_CLOSE_WAIT), (
        f"LiveKit WS port {cfg.lk_port_ws} still open after stop"
    )
    assert not _container_running(name), (
        f"container {name!r} still listed by docker ps after stop"
    )


# ── room-client test (shares a docker container via fixture) ─────────────────


@pytest.fixture(scope="module")
async def live_docker() -> AsyncIterator[LiveKitConnectorConfig]:
    """Module-scoped LiveKit container shared across room-client tests.

    monkeypatch is function-scoped, so swap the module constant directly.
    """
    name = f"xr-ai-livekit-test-{uuid.uuid4().hex[:10]}"
    cfg  = _build_cfg()
    orig = _docker_mod._CONTAINER_NAME
    _docker_mod._CONTAINER_NAME = name
    docker = LiveKitDocker(cfg)
    await docker.start()
    try:
        yield cfg
    finally:
        await docker.stop()
        _docker_mod._CONTAINER_NAME = orig


async def test_room_client_connect(
    live_docker: LiveKitConnectorConfig,
    make_connector,
):
    cfg = replace(live_docker, identity=f"test-{uuid.uuid4().hex[:6]}")
    ep: ConnectorEndpoint = make_connector()
    client = RoomClient(cfg, ep)

    await client.connect()
    try:
        assert client._room.connection_state == rtc.ConnectionState.CONN_CONNECTED, (
            f"room not connected: state={client._room.connection_state}"
        )
    finally:
        await client.disconnect()

    assert client._room.connection_state != rtc.ConnectionState.CONN_CONNECTED, (
        "room still reports CONN_CONNECTED after disconnect()"
    )
