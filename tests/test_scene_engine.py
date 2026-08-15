# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only lifecycle and scene-resync regressions for the XR renderer."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path

import msgpack
import pytest
import zmq
import zmq.asyncio
from xr_render_scene.engine import Config, SceneDispatcher

_asyncio = pytest.mark.asyncio


def _unique_ipc(tmp_path: Path) -> str:
    return f"ipc://{tmp_path}/scene_{uuid.uuid4().hex[:8]}"


async def _cancel_task(task: asyncio.Task[object] | None) -> None:
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _recv_op(pull: zmq.asyncio.Socket, timeout: float = 1.0) -> dict:
    raw = await asyncio.wait_for(pull.recv(), timeout=timeout)
    return msgpack.unpackb(raw, raw=False)


class _FakeLovrProc:
    """Stand-in for a managed process that exits when triggered."""
    def __init__(self) -> None:
        self._done = asyncio.Event()

    async def wait(self) -> int:
        await self._done.wait()
        return 0

    def trigger_exit(self) -> None:
        """Simulate the LOVR child exiting, unblocking ``wait()``."""
        self._done.set()


class _FakeManagedProcessCtx:
    def __init__(self) -> None:
        self.proc = _FakeLovrProc()
        self.exited = False

    async def __aenter__(self) -> _FakeLovrProc:
        return self.proc

    async def __aexit__(self, *exc) -> None:
        self.exited = True
        return None


@_asyncio
async def test_close_cancels_lovr_watch_task(tmp_path: Path, monkeypatch):
    """Closing the dispatcher cancels its child-process watch task."""
    sock_path = _unique_ipc(tmp_path)
    lovr_bin  = tmp_path / "lovr.sh"
    lovr_bin.write_text("#!/bin/sh\nsleep 999\n")
    lovr_bin.chmod(0o755)
    xr_app_dir = tmp_path / "xr_app"
    xr_app_dir.mkdir()

    cfg = Config(
        lovr_bin         = lovr_bin,
        xr_app_dir       = xr_app_dir,
        scene_socket     = sock_path,
        cloudxr_env_file = None,
        endpoint         = f"ipc://{tmp_path}/unused",
    )

    monkeypatch.setattr("xr_render_scene.engine.ManagedProcess",
                        lambda *a, **kw: _FakeManagedProcessCtx())

    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    try:
        disp = SceneDispatcher(cfg, stack)
        result = await disp.start_lovr_once()
        assert result == {"status": "started"}
        assert disp._watch_task is not None
        assert not disp._watch_task.done()

        disp.close()
        await _cancel_task(disp._watch_task)
        assert disp._watch_task.done()
    finally:
        await stack.__aexit__(None, None, None)


@_asyncio
async def test_lovr_respawn_closes_previous_launch_context(tmp_path: Path, monkeypatch):
    """A respawn closes the previous per-launch process context."""
    sock_path = _unique_ipc(tmp_path)
    lovr_bin  = tmp_path / "lovr.sh"
    lovr_bin.write_text("#!/bin/sh\nsleep 999\n")
    lovr_bin.chmod(0o755)
    xr_app_dir = tmp_path / "xr_app"
    xr_app_dir.mkdir()

    cfg = Config(
        lovr_bin         = lovr_bin,
        xr_app_dir       = xr_app_dir,
        scene_socket     = sock_path,
        cloudxr_env_file = None,
        endpoint         = f"ipc://{tmp_path}/unused",
    )

    created: list[_FakeManagedProcessCtx] = []

    def _make_ctx(*_a, **_kw) -> _FakeManagedProcessCtx:
        ctx = _FakeManagedProcessCtx()
        created.append(ctx)
        return ctx

    monkeypatch.setattr("xr_render_scene.engine.ManagedProcess", _make_ctx)

    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    try:
        disp = SceneDispatcher(cfg, stack)

        assert await disp.start_lovr_once() == {"status": "started"}
        assert len(created) == 1
        assert disp._launch_stack is not None
        assert created[0].exited is False  # live

        created[0].proc.trigger_exit()
        await disp._watch_task

        assert created[0].exited is True
        assert disp._launch_stack is None
        assert disp._lovr_started is False

        assert await disp.start_lovr_once() == {"status": "started"}
        assert len(created) == 2
        assert created[0].exited is True
        assert created[1].exited is False
        assert sum(c.exited for c in created) == 1  # only the dead one closed

        disp.close()
        await _cancel_task(disp._watch_task)
    finally:
        await stack.__aexit__(None, None, None)

    assert created[1].exited is True


# ── Scene resync ─────────────────────────────────────────────────────────────


@_asyncio
async def test_resync_delivers_after_late_peer_connect(tmp_path: Path):
    """Resync waits for a late LOVR peer and delivers every scene object."""
    sock_path = _unique_ipc(tmp_path)
    cfg = Config(
        lovr_bin=Path("/nonexistent"), xr_app_dir=tmp_path,
        scene_socket=sock_path, cloudxr_env_file=None,
        endpoint=f"ipc://{tmp_path}/unused",
    )
    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    pull = None
    try:
        disp = SceneDispatcher(cfg, stack)
        a = disp.add("sphere", {"x": 0, "y": 0, "z": 0}, {"r": 1, "g": 0, "b": 0}, 0.1)
        b = disp.add("box",    {"x": 1, "y": 2, "z": 3}, {"r": 0, "g": 1, "b": 0}, 0.2)

        resync_task = asyncio.create_task(disp._resync_scene())
        await asyncio.sleep(0.1)
        assert not resync_task.done()

        ctx = zmq.asyncio.Context.instance()
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.LINGER, 0)
        pull.connect(sock_path)

        ops = [await _recv_op(pull, timeout=2.0) for _ in range(2)]
        await asyncio.wait_for(resync_task, timeout=2.0)

        assert all(o["op"] == "scene.add" for o in ops)
        assert {o["value"]["id"] for o in ops} == {a, b}
    finally:
        if pull is not None:
            pull.close(linger=0)
        disp.close()
        await stack.__aexit__(None, None, None)


@_asyncio
async def test_live_forward_fast_drops_during_resync_window(tmp_path: Path, monkeypatch):
    """Live mutations fast-drop while blocking resync waits for LOVR."""
    sock_path = _unique_ipc(tmp_path)
    lovr_bin  = tmp_path / "lovr.sh"
    lovr_bin.write_text("#!/bin/sh\nsleep 999\n")
    lovr_bin.chmod(0o755)
    xr_app_dir = tmp_path / "xr_app"
    xr_app_dir.mkdir()
    cfg = Config(
        lovr_bin=lovr_bin, xr_app_dir=xr_app_dir, scene_socket=sock_path,
        cloudxr_env_file=None, endpoint=f"ipc://{tmp_path}/unused",
    )
    monkeypatch.setattr("xr_render_scene.engine.ManagedProcess",
                        lambda *a, **kw: _FakeManagedProcessCtx())

    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    pull = None
    try:
        disp = SceneDispatcher(cfg, stack)
        disp.add("sphere", {"x": 0, "y": 0, "z": 0}, {"r": 1, "g": 0, "b": 0}, 0.1)

        start_task = asyncio.create_task(disp.start_lovr_once())
        await asyncio.sleep(0.1)
        assert not start_task.done()
        assert disp._lovr_started is False

        res = await disp.forward("scene.update", {"id": "sphere-0", "x": 1})
        assert res == {"ok": False, "reason": "not_started"}

        ctx = zmq.asyncio.Context.instance()
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.LINGER, 0)
        pull.connect(sock_path)
        _ = await _recv_op(pull, timeout=2.0)
        result = await asyncio.wait_for(start_task, timeout=2.0)
        assert result == {"status": "started"}
        assert disp._lovr_started is True
    finally:
        if pull is not None:
            pull.close(linger=0)
        disp.close()
        await stack.__aexit__(None, None, None)


@_asyncio
async def test_resync_is_bounded_when_lovr_never_connects(tmp_path: Path, monkeypatch):
    """Resync returns by its deadline when LOVR never connects."""
    monkeypatch.setattr("xr_render_scene.engine._RESYNC_TIMEOUT_S", 0.2)
    sock_path = _unique_ipc(tmp_path)
    cfg = Config(
        lovr_bin=Path("/nonexistent"), xr_app_dir=tmp_path,
        scene_socket=sock_path, cloudxr_env_file=None,
        endpoint=f"ipc://{tmp_path}/unused",
    )
    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    try:
        disp = SceneDispatcher(cfg, stack)
        disp.add("sphere", {"x": 0, "y": 0, "z": 0}, {"r": 1, "g": 1, "b": 1}, 0.1)

        start = asyncio.get_running_loop().time()
        await asyncio.wait_for(disp._resync_scene(), timeout=3.0)
        elapsed = asyncio.get_running_loop().time() - start
        assert elapsed < 2.0
    finally:
        disp.close()
        await stack.__aexit__(None, None, None)
