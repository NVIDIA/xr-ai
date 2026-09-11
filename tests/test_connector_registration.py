# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory registration handshake and frame-routing regressions."""
from __future__ import annotations

import asyncio

import device_io_hub.ipc._connector as connector_module
import device_io_hub.ipc._hub as hub_module
import pytest
from xr_ai_hub import FrameSignal, MsgType, ParticipantEvent, PixelFormat, encode

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
            connector._ring.unlink()
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
        raise ValueError(f"invalid layout in {name}")

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
    hub,
    make_connector,
    monkeypatch,
):
    connector = make_connector()
    registrations = []

    async def withhold_ack(reg, *_args):
        registrations.append(reg)

    monkeypatch.setattr(hub, "_acknowledge_registration", withhold_ack)
    monkeypatch.setattr(connector_module, "_REGISTRATION_RESEND_INTERVAL_S", 0.01)
    monkeypatch.setattr(connector_module, "_DEFAULT_REGISTRATION_TIMEOUT_S", 0.1)
    monkeypatch.setattr(connector_module, "_DEFAULT_REGISTRATION_ATTEMPTS", 1)
    with pytest.raises(
        connector_module._ConnectorRegistrationError,
        match="registration_timeout",
    ):
        await asyncio.wait_for(connector.register(), timeout=1.0)
    assert connector._registered is False
    assert len(registrations) >= 2
    assert {reg.shm_name for reg in registrations} == {connector._shm_name}


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


async def test_frame_without_registered_ring_does_not_stop_hub(
    hub,
    make_connector,
    settle,
):
    connector = make_connector()
    await settle()
    await connector.register()
    await connector.notify_participant_joined("alice", pts_us=1)
    await settle()

    await connector._push.send(encode(MsgType.PARTICIPANT_EVENT, ParticipantEvent(
        participant_id="foreign", joined=True, pts_us=1, connector_id="unregistered",
    )))
    await connector._push.send(encode(MsgType.FRAME_SIGNAL, FrameSignal(
        slot=0, seq=1, pts_us=2, width=1, height=1, fmt=PixelFormat.RGBA,
        data_sz=4, participant_id="foreign", track_id="camera",
    )))
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
    assert ("foreign", "camera") not in hub._latest_slots
    assert bytes(hub._latest_slots[("alice", "camera")][1].data) == b"ABCD"
    assert hub._running is True


async def test_duplicate_registration_preserves_mapping_and_held_frame(
    hub, make_connector, settle, monkeypatch,
):
    connector = make_connector()
    await connector.register()
    await connector.notify_participant_joined("alice", pts_us=1)
    await connector.push_frame(b"ABCD", 1, 1, PixelFormat.RGBA, 2, "alice", "camera")
    await settle()
    ring = hub._ring_registry[connector._connector_id]
    held = hub._latest_slots[("alice", "camera")]

    def unexpected_attach(**_kwargs):
        pytest.fail("duplicate registration must not reattach the segment")

    monkeypatch.setattr(hub_module, "ShmRingBuffer", unexpected_attach)
    await connector.register()

    assert hub._ring_registry[connector._connector_id] is ring
    assert hub._latest_slots[("alice", "camera")] is held
    assert bytes(held[1].data) == b"ABCD"


async def test_registration_resend_recovers_a_lost_ack(hub, make_connector, monkeypatch):
    connector = make_connector()
    acknowledge = hub._acknowledge_registration
    registered_rings = []

    async def drop_first_ack(reg, *args):
        registered_rings.append(hub._ring_registry[reg.connector_id])
        if len(registered_rings) > 1:
            await acknowledge(reg, *args)

    monkeypatch.setattr(hub, "_acknowledge_registration", drop_first_ack)
    monkeypatch.setattr(connector_module, "_REGISTRATION_RESEND_INTERVAL_S", 0.01)
    await asyncio.wait_for(connector.register(), timeout=1.0)

    assert connector._registered is True
    assert len(registered_rings) >= 2
    assert all(ring is registered_rings[0] for ring in registered_rings)
