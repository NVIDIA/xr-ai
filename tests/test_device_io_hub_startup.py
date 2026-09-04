# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeviceIOHub startup ordering and readiness regressions."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import device_io_hub.__main__ as hub_main
import pytest
from device_io_hub._errors import StartupError
from device_io_hub.ipc._connector import _ConnectorRegistrationError
from device_io_hub.transport.livekit import connector as connector_module


@pytest.mark.asyncio
async def test_registration_failure_never_creates_ready_file(
    monkeypatch,
    tmp_path,
):
    receive_started = asyncio.Event()

    class FakeHub:
        def on_frame(self, _callback): pass
        def on_audio(self, _callback): pass
        def on_data(self, _callback): pass
        def on_participant(self, _callback): pass

        async def run(self):
            receive_started.set()
            await asyncio.Event().wait()

        def stop(self): pass
        def close(self): pass

    class FakeConnector:
        async def start(self):
            assert receive_started.is_set()
            raise StartupError("registration failed")

    connector = FakeConnector()
    monkeypatch.setattr(hub_main, "setup_logging", lambda _name: None)
    monkeypatch.setattr(
        hub_main,
        "load_config",
        lambda: SimpleNamespace(
            hub_push_addr="ipc://unused-in",
            hub_sub_addr="ipc://unused-out",
        ),
    )
    monkeypatch.setattr(hub_main, "HubEndpoint", lambda **_kwargs: FakeHub())
    monkeypatch.setattr(hub_main, "LiveKitConnector", lambda _cfg: connector)
    ready_file = tmp_path / "hub.ready"

    with pytest.raises(StartupError, match="registration failed"):
        await hub_main.main(ready_file=ready_file)

    assert receive_started.is_set()
    assert not ready_file.exists()


@pytest.mark.asyncio
async def test_livekit_room_is_not_connected_when_registration_fails(monkeypatch):
    events: list[str] = []

    class Service:
        async def start(self):
            events.append("service-start")

        async def stop(self):
            events.append("service-stop")

    class Endpoint:
        async def register(self):
            events.append("register")
            raise _ConnectorRegistrationError("shm_not_found", "segment disappeared")

        def stop(self): pass
        def close(self): events.append("endpoint-close")

    class Room:
        async def connect(self):
            events.append("room-connect")

        def stop(self): pass

    connector = connector_module.LiveKitConnector.__new__(
        connector_module.LiveKitConnector
    )
    connector._cfg = SimpleNamespace(room_name="test")
    connector._docker = Service()
    connector._token = Service()
    connector._web = Service()
    connector._ep = Endpoint()
    connector._room_client = Room()
    connector._room_connected = False
    monkeypatch.setattr(connector_module, "require_nvidia_video_codecs", lambda: None)

    with pytest.raises(StartupError, match="shm_not_found"):
        await connector.start()

    assert events[:4] == ["service-start", "service-start", "service-start", "register"]
    assert "room-connect" not in events
    assert "endpoint-close" in events
