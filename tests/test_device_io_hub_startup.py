# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeviceIOHub startup ordering and readiness regressions."""
from __future__ import annotations

import asyncio
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import device_io_hub.__main__ as hub_main
import pytest
from device_io_hub._errors import StartupError
from device_io_hub.ipc._connector import _ConnectorRegistrationError
from device_io_hub.transport.livekit import connector as connector_module


@pytest.mark.asyncio
async def test_registration_failure_never_creates_ready_file(main_runtime):
    runtime = main_runtime
    runtime.connector.start.side_effect = StartupError("registration failed")

    with pytest.raises(StartupError, match="registration failed"):
        await hub_main.main(ready_file=runtime.ready_file)

    assert runtime.hub_started.is_set()
    assert not runtime.ready_file.exists()
    runtime.hub.close.assert_called_once()
    assert all(task.done() for task in runtime.tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["docker", "token", "web", "register", "connect", "cancel-connect"])
async def test_connector_start_failure_cleans_up_all_resources(
    hub, make_connector, monkeypatch, stage,
):
    endpoint = make_connector()
    connector = connector_module.LiveKitConnector.__new__(connector_module.LiveKitConnector)
    connector._cfg = SimpleNamespace(room_name="test")
    connector._docker = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    connector._token = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    connector._web = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    connector._ep = endpoint
    connector._room_connect_started = False
    connector._room_client = SimpleNamespace(
        connect=AsyncMock(), disconnect=AsyncMock(), stop=Mock(),
    )
    monkeypatch.setattr(connector_module, "require_nvidia_video_codecs", lambda: None)
    failure = asyncio.CancelledError() if stage == "cancel-connect" else RuntimeError(stage)
    if stage == "register":
        failure = _ConnectorRegistrationError("shm_not_found", "segment disappeared")
        monkeypatch.setattr(endpoint, "register", AsyncMock(side_effect=failure))
    elif "connect" in stage:
        connector._room_client.connect.side_effect = failure
    else:
        getattr(connector, f"_{stage}").start.side_effect = failure

    with pytest.raises(StartupError if stage == "register" else type(failure)) as caught:
        await connector.start()

    if stage == "register":
        assert caught.value.__cause__ is failure
        assert str(caught.value).startswith("\n" + "━" * 56 + "\n")
    else:
        assert caught.value is failure
    connector._docker.stop.assert_awaited_once()
    connector._token.stop.assert_awaited_once()
    connector._web.stop.assert_awaited_once()
    if "connect" in stage:
        connector._room_client.disconnect.assert_awaited_once()
    else:
        connector._room_client.connect.assert_not_awaited()
        connector._room_client.disconnect.assert_not_awaited()
    assert endpoint._ring is None
    assert endpoint._push.closed
    assert endpoint._sub.closed
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=endpoint._shm_base_name, create=False)


@pytest.fixture
def main_runtime(monkeypatch, tmp_path):
    hub_started = asyncio.Event()
    runtime_started = asyncio.Event()
    running_tasks = []
    ready_file = tmp_path / "hub.ready"

    async def run_hub():
        running_tasks.append(asyncio.current_task())
        hub_started.set()
        await asyncio.Event().wait()

    async def start_connector():
        assert hub_started.is_set()

    async def run_connector():
        running_tasks.append(asyncio.current_task())
        assert ready_file.exists()
        runtime_started.set()
        await asyncio.Event().wait()

    hub = SimpleNamespace(
        on_frame=Mock(), on_audio=Mock(), on_data=Mock(), on_participant=Mock(),
        run=AsyncMock(side_effect=run_hub), stop=Mock(), close=Mock(),
    )
    connector = SimpleNamespace(
        start=AsyncMock(side_effect=start_connector),
        run=AsyncMock(side_effect=run_connector), stop=AsyncMock(),
    )
    config = SimpleNamespace(
        hub_push_addr="ipc://unused-in", hub_sub_addr="ipc://unused-out",
        video_recording={}, web_server_tls=False, enable_web_server=False,
        lk_port_ws=7880, room_name="test",
    )
    monkeypatch.setattr(hub_main, "setup_logging", lambda _name: None)
    monkeypatch.setattr(hub_main, "load_config", lambda: config)
    monkeypatch.setattr(hub_main, "HubEndpoint", lambda **_kwargs: hub)
    monkeypatch.setattr(hub_main, "LiveKitConnector", lambda _cfg: connector)
    monkeypatch.setattr(hub_main, "make_client_token", Mock(return_value="test-token"))
    monkeypatch.setattr(hub_main, "_recorder", None)
    return SimpleNamespace(
        hub=hub, connector=connector, config=config, hub_started=hub_started,
        runtime_started=runtime_started, tasks=running_tasks, ready_file=ready_file,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["token", "recorder"])
async def test_failure_after_connector_start_cleans_up_before_ready(main_runtime, monkeypatch, stage):
    runtime = main_runtime
    if stage == "token":
        monkeypatch.setattr(hub_main, "make_client_token", Mock(side_effect=ValueError("bad token")))
    else:
        runtime.config.video_recording = {"enabled": True, "chunk_frames": "not-an-integer"}

    with pytest.raises(ValueError):
        await asyncio.wait_for(hub_main.main(ready_file=runtime.ready_file), timeout=1.0)

    assert not runtime.ready_file.exists()
    runtime.connector.stop.assert_awaited_once()
    runtime.hub.stop.assert_called_once()
    runtime.hub.close.assert_called_once()
    assert all(task.done() for task in runtime.tasks)


@pytest.mark.asyncio
async def test_main_cancellation_after_ready_cleans_up(main_runtime):
    runtime = main_runtime
    main_task = asyncio.create_task(hub_main.main(ready_file=runtime.ready_file))
    try:
        await asyncio.wait_for(runtime.runtime_started.wait(), timeout=1.0)
    finally:
        main_task.cancel()
        await asyncio.gather(main_task, return_exceptions=True)

    assert main_task.cancelled()
    runtime.connector.stop.assert_awaited_once()
    runtime.hub.close.assert_called_once()
    assert all(task.done() for task in runtime.tasks)


@pytest.mark.asyncio
async def test_shutdown_signal_after_ready_exits_cleanly(main_runtime, monkeypatch):
    runtime = main_runtime
    loop = asyncio.get_running_loop()
    handlers = {}
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: handlers.update({sig: callback}))
    remove_handler = Mock()
    monkeypatch.setattr(loop, "remove_signal_handler", remove_handler)
    main_task = asyncio.create_task(hub_main.main(ready_file=runtime.ready_file))
    try:
        await asyncio.wait_for(runtime.runtime_started.wait(), timeout=1.0)
        handlers[hub_main.signal.SIGTERM]()
        await asyncio.wait_for(main_task, timeout=1.0)
    finally:
        main_task.cancel()
        await asyncio.gather(main_task, return_exceptions=True)

    runtime.connector.stop.assert_awaited_once()
    runtime.hub.close.assert_called_once()
    assert remove_handler.call_count == 2
    assert all(task.done() for task in runtime.tasks)
