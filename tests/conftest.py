"""
Shared pytest fixtures for xr-ai multi-client / multi-agent tests.

The IPC suite (hub + connectors + processor endpoints) does **not** require
LiveKit, Docker, or NVENC — it talks to ZMQ over ``ipc://`` sockets only.
This keeps the bulk of the multi-client / multi-agent coverage runnable on
a developer laptop in seconds.

Fixtures
--------
``hub_addrs``           — fresh ``(pull, pub)`` ZMQ addresses for each test.
``hub``                 — a running :class:`HubEndpoint`.
``make_connector``      — factory yielding fresh :class:`ConnectorEndpoint`s
                          tied to ``hub_addrs`` (each represents one client
                          on its own ring buffer).
``make_processor``      — factory yielding fresh :class:`ProcessorEndpoint`s
                          (each represents one agent / consumer).
``settle``              — small ``await asyncio.sleep`` that lets ZMQ
                          PUB/SUB plumbing flush a pending message.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from typing import AsyncIterator, Callable

import pytest

from xr_ai_agent          import ProcessorEndpoint
from xr_media_hub.ipc     import ConnectorEndpoint, HubEndpoint


# ── address fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def hub_addrs(tmp_path) -> tuple[str, str]:
    """Return a unique (pull, pub) pair so tests never collide on sockets."""
    uid  = uuid.uuid4().hex[:8]
    pull = f"ipc://{tmp_path}/hub_in_{uid}"
    pub  = f"ipc://{tmp_path}/hub_pub_{uid}"
    return pull, pub


# ── hub fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
async def hub(hub_addrs) -> AsyncIterator[HubEndpoint]:
    pull, pub = hub_addrs
    ep = HubEndpoint(pull_addr=pull, pub_addr=pub)
    task = asyncio.create_task(ep.run(), name="hub")
    try:
        await asyncio.sleep(0.05)  # let bind() complete
        yield ep
    finally:
        ep.stop()
        # Run loop checks `_running` only after the next recv(); send a sentinel
        # via PULL by closing the socket — easier to just cancel the task here.
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        ep.close()


# ── connector / processor factories ─────────────────────────────────────────


@pytest.fixture
async def make_connector(hub_addrs):
    """Factory creating client-side ``ConnectorEndpoint``s wired to the hub.

    Each call yields a fresh connector with its own ring buffer / shm name.
    The fixture cleans up every connector it produced when the test exits.
    """
    pull, pub = hub_addrs
    created: list[ConnectorEndpoint] = []

    def _make(connector_id: str | None = None) -> ConnectorEndpoint:
        cid  = connector_id or f"conn_{uuid.uuid4().hex[:8]}"
        # Use a short shm name to stay under POSIX limits.
        shm  = f"xr_test_{uuid.uuid4().hex[:10]}"
        ep   = ConnectorEndpoint(
            push_addr=pull,
            sub_addr=pub,
            connector_id=cid,
            shm_name=shm,
            num_slots=4,
            max_frame_bytes=64 * 1024,
        )
        created.append(ep)
        return ep

    yield _make

    for ep in created:
        ep.stop()
        try:
            ep.close()
        except Exception:
            pass


@pytest.fixture
async def make_processor(hub_addrs):
    """Factory creating consumer-side :class:`ProcessorEndpoint`s.

    Each instance gets its own background ``run()`` task; tearing the
    fixture down stops and closes every processor it produced.

    All keyword arguments are forwarded straight to ``ProcessorEndpoint``
    (e.g. ``filter=Subscribe.AUDIO`` or ``auto_subscribe=False``).
    """
    pull, pub = hub_addrs
    created: list[tuple[ProcessorEndpoint, asyncio.Task]] = []

    def _make(**kwargs) -> ProcessorEndpoint:
        ep = ProcessorEndpoint(sub_addr=pub, push_addr=pull, **kwargs)
        task = asyncio.create_task(ep.run(), name=f"proc_{uuid.uuid4().hex[:6]}")
        created.append((ep, task))
        return ep

    yield _make

    for ep, task in created:
        ep.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        ep.close()


# ── helpers ─────────────────────────────────────────────────────────────────


@pytest.fixture
def settle() -> Callable[[], "asyncio.Future[None]"]:
    """Coroutine that waits long enough for a PUB/SUB hop to drain."""
    async def _settle() -> None:
        # ZMQ PUB/SUB inproc/ipc is fast but not synchronous; ~50 ms is the
        # smallest interval that's been reliably non-flaky in CI for the
        # other ZMQ-based suites in this repo.
        await asyncio.sleep(0.05)
    return _settle
