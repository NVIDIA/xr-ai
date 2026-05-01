# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test helpers — keep the cross-talk suite readable.

The hub IPC tests repeatedly need to:
  * register N connectors (one per simulated client) and join one
    participant per connector;
  * spin up each connector's ``run()`` task so return-traffic actually
    flows;
  * collect per-pid received messages into deterministic dicts.

These helpers wrap that boilerplate so each test reads as just the
scenario under verification.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from xr_ai_agent      import AudioChunk, DataMessage, ReturnAudioFlush
from xr_media_hub.ipc import ConnectorEndpoint


@dataclass
class FakeClient:
    """One simulated client = one connector + one participant id.

    The connector's ``run()`` task pulls from the hub's PUB socket so the
    test can assert on what the client *would* have received (return data,
    return audio, flush messages).
    """
    pid:        str
    connector:  ConnectorEndpoint
    task:       asyncio.Task
    return_data:        list[DataMessage]      = field(default_factory=list)
    return_audio:       list[AudioChunk]       = field(default_factory=list)
    return_audio_flush: list[ReturnAudioFlush] = field(default_factory=list)


async def setup_client(
    make_connector: Callable[..., ConnectorEndpoint],
    pid:            str,
    pts_us:         int = 1,
) -> FakeClient:
    """Create one FakeClient — connector registered, participant joined,
    callbacks installed, run task running.

    Caller is responsible for shutting it down via :func:`teardown_clients`.
    """
    conn = make_connector(connector_id=f"{pid}_conn")
    await conn.register()
    # Tiny settle so the hub processes the registration before we send
    # PARTICIPANT_EVENT — otherwise the hub doesn't yet know the
    # connector_id ↔ shm mapping that subsequent media depends on.
    await asyncio.sleep(0.05)

    fc = FakeClient(pid=pid, connector=conn, task=None)  # type: ignore[arg-type]

    async def cb_data(msg, fc=fc):  fc.return_data.append(msg)
    async def cb_audio(msg, fc=fc): fc.return_audio.append(msg)
    async def cb_flush(msg, fc=fc): fc.return_audio_flush.append(msg)

    conn.on_return_data(cb_data)
    conn.on_return_audio(cb_audio)
    conn.on_return_audio_flush(cb_flush)

    fc.task = asyncio.create_task(conn.run(), name=f"{pid}_conn_run")

    await conn.notify_participant_joined(pid, pts_us=pts_us)
    # Let the JOIN message land at the hub (so subsequent send_return_*
    # validations find the participant connected).
    await asyncio.sleep(0.05)

    return fc


async def teardown_clients(clients: list[FakeClient]) -> None:
    for fc in clients:
        fc.connector.stop()
    for fc in clients:
        fc.task.cancel()
    await asyncio.gather(
        *(fc.task for fc in clients), return_exceptions=True,
    )


async def wait_for(predicate: Callable[[], bool], *, timeout: float = 1.5) -> None:
    """Poll ``predicate`` every 50 ms up to ``timeout`` seconds."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)


async def wait_for_subscribed(*processors, pids) -> None:
    """Wait until every processor has auto-subscribed to all *pids*.

    Use this when the agent is constructed *after* the clients are
    already connected — the agent has to round-trip a roster request to
    the hub, and tests that push data immediately would race the
    catch-up handshake.
    """
    expected = frozenset(pids)
    await wait_for(lambda: all(
        ep.subscribed_participants >= expected for ep in processors
    ))
    # setsockopt(SUBSCRIBE) updates Python state synchronously but the ZMQ
    # subscription message still propagates to the PUB over ipc://; without
    # this settle, data pushed immediately after can race the subscription.
    await asyncio.sleep(0.05)


def silence(pid: str, *, pts_us: int = 0, sample_rate: int = 48_000,
            samples: int = 480) -> AudioChunk:
    """Build an AudioChunk of silence attributed to *pid*."""
    return AudioChunk(
        pts_us=pts_us, sample_rate=sample_rate, channels=1,
        samples=samples, data=b"\x00" * (samples * 2),
        participant_id=pid, track_id="mic",
    )
