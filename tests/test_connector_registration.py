# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory registration handshake and connector health regressions."""
from __future__ import annotations

import asyncio

import device_io_hub.ipc._connector as connector_module
import device_io_hub.ipc._hub as hub_module
import pytest
from device_io_hub.ipc import (
    ConnectorEndpoint,
)
from xr_ai_hub import PixelFormat
from xr_ai_hub._shm import _IncompatibleSharedMemoryError

pytestmark = pytest.mark.asyncio


async def test_ring_is_created_only_when_registration_begins(
    hub,
    make_connector,
    settle,
):
    connector = make_connector()
    assert connector._ring is None
    assert connector._shm_name == ""

    await settle()
    await connector.register()

    assert connector._ring is not None
    assert connector._registered is True
    assert hub._ring_registry[connector._connector_id] is not None


async def test_missing_ring_is_recreated_with_unique_name(
    hub,
    make_connector,
    settle,
    monkeypatch,
):
    connector = make_connector()
    real_ring = hub_module.ShmRingBuffer
    missing_name: str | None = None
    attempted_names: list[str] = []

    def fail_first_name(*, name: str, create: bool):
        nonlocal missing_name
        attempted_names.append(name)
        if missing_name is None:
            missing_name = name
        if name == missing_name:
            raise FileNotFoundError(name)
        return real_ring(name=name, create=create)

    monkeypatch.setattr(hub_module, "ShmRingBuffer", fail_first_name)
    await settle()

    await connector.register()

    assert connector._registered is True
    assert attempted_names[0] != attempted_names[-1]
    assert connector._shm_name == attempted_names[-1]
    assert connector._shm_name.startswith(f"{connector._shm_base_name}_")


async def test_incompatible_ring_fails_without_connecting_media(
    hub,
    make_connector,
    settle,
    monkeypatch,
):
    connector = make_connector()

    def reject_ring(*, name: str, create: bool):
        raise _IncompatibleSharedMemoryError(f"invalid layout in {name}")

    monkeypatch.setattr(hub_module, "ShmRingBuffer", reject_ring)
    await settle()

    with pytest.raises(connector_module._ConnectorRegistrationError, match="shm_incompatible"):
        await connector.register()

    assert connector._registered is False
    with pytest.raises(
        connector_module._ConnectorRegistrationError,
        match="before shared-memory registration",
    ):
        await connector.push_frame(b"ABCD", 1, 1, PixelFormat.RGBA, 1)


async def test_registration_acknowledgement_has_bounded_timeout(
    hub_addrs,
    monkeypatch,
):
    pull, pub = hub_addrs
    connector = ConnectorEndpoint(
        push_addr=pull,
        sub_addr=pub,
        connector_id="no-hub",
        shm_name="xr_test_no_hub",
        num_slots=1,
        max_frame_bytes=64,
    )
    monkeypatch.setattr(connector_module, "_DEFAULT_REGISTRATION_TIMEOUT_S", 0.05)
    monkeypatch.setattr(connector_module, "_DEFAULT_REGISTRATION_ATTEMPTS", 1)
    try:
        with pytest.raises(
            connector_module._ConnectorRegistrationError,
            match="registration_timeout",
        ):
            await asyncio.wait_for(
                connector.register(),
                timeout=0.5,
            )
        assert connector._registered is False
    finally:
        connector.close()


async def test_ring_creation_failure_is_structured(
    make_connector,
    monkeypatch,
):
    connector = make_connector()

    def fail_create(**_kwargs):
        raise FileExistsError("stale shared-memory name")

    monkeypatch.setattr(connector_module, "ShmRingBuffer", fail_create)

    with pytest.raises(connector_module._ConnectorRegistrationError, match="shm_create_failed"):
        await connector.register()

    assert connector._registered is False
    assert connector._ring is None


async def test_frame_without_registered_ring_marks_connector_unhealthy(
    hub,
    make_connector,
    settle,
):
    connector = make_connector()
    await settle()
    await connector.register()
    await connector.notify_participant_joined("alice", pts_us=1)
    await settle()

    consumer_ring = hub._ring_registry.pop(connector._connector_id)
    consumer_ring.close()
    await connector.push_frame(
        b"ABCD",
        width=1,
        height=1,
        fmt=PixelFormat.RGBA,
        pts_us=2,
        participant_id="alice",
        track_id="camera",
    )
    await settle()
    assert connector._connector_id in hub._unhealthy_connectors
    assert hub._running is False
