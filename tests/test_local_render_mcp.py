# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-only tests for render-mcp's ZMQ scene-socket wire format and lifecycle.

LOVR (the OpenXR rendering app) is stubbed — the goal is to verify render-mcp's
ZMQ PUSH wire format, the in-memory scene state, the FastMCP tool surface,
config validation, and the lifecycle bookkeeping that keeps the LOVR watch
task from leaking. No real OpenXR / Vulkan is exercised.

Marked ``gpu`` because the suite is local-development-only: it spins up
ephemeral ipc:// sockets in /tmp and is not part of the CI matrix.
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path

import msgpack
import pytest
import zmq
import zmq.asyncio

import render_mcp.__main__ as render_main
from render_mcp.__main__ import (
    Config,
    SceneDispatcher,
    _build_config,
    _validate_scene_socket,
    build_mcp,
)

pytestmark = pytest.mark.gpu

# Per-class asyncio marker so the synchronous helper tests below don't trip
# pytest-asyncio's "marked but not async" warning. asyncio_mode = "auto" also
# auto-detects but the explicit marker is what the rest of the suite uses.
_asyncio = pytest.mark.asyncio


# ── ZMQ fixtures ─────────────────────────────────────────────────────────────


def _unique_ipc(tmp_path: Path) -> str:
    """Return a per-test ipc:// socket path under tmp_path."""
    return f"ipc://{tmp_path}/scene_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def dispatcher(tmp_path: Path):
    """Yield (SceneDispatcher, pull_socket) sharing a fresh ipc:// path.

    The dispatcher binds PUSH; the test acts as LOVR via a PULL on the same
    address. ``_lovr_started`` is flipped True so ``forward()`` actually
    sends — the test is exercising the wire format, not the spawn gate.
    """
    sock_path = _unique_ipc(tmp_path)
    cfg = Config(
        lovr_bin         = Path("/nonexistent"),  # never spawned in these tests
        xr_app_dir       = tmp_path,
        scene_socket     = sock_path,
        cloudxr_env_file = None,
        host             = "127.0.0.1",
        port             = 0,
    )
    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    disp = SceneDispatcher(cfg, stack)
    disp._lovr_started = True  # bypass spawn-gating

    ctx  = zmq.asyncio.Context.instance()
    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.LINGER, 0)
    pull.connect(sock_path)
    # ZMQ PUSH/PULL over ipc:// connects asynchronously; give the kernel a tick
    # before the first send so messages aren't queued before the peer attaches.
    await asyncio.sleep(0.05)

    try:
        yield disp, pull
    finally:
        pull.close(linger=0)
        disp.close()
        await stack.__aexit__(None, None, None)


async def _recv_op(pull: zmq.asyncio.Socket, timeout: float = 1.0) -> dict:
    raw = await asyncio.wait_for(pull.recv(), timeout=timeout)
    return msgpack.unpackb(raw, raw=False)


# ── scene_socket validation (tech debt #13) ──────────────────────────────────


class TestSceneSocketValidation:
    def test_accepts_ipc(self):
        _validate_scene_socket("ipc:///tmp/foo")

    def test_accepts_tcp(self):
        _validate_scene_socket("tcp://127.0.0.1:5555")

    def test_accepts_inproc(self):
        _validate_scene_socket("inproc://name")

    def test_rejects_empty_ipc_path(self):
        with pytest.raises(SystemExit):
            _validate_scene_socket("ipc:///")

    def test_rejects_bare_ipc(self):
        with pytest.raises(SystemExit):
            _validate_scene_socket("ipc://")

    def test_rejects_tcp_without_port(self):
        with pytest.raises(SystemExit):
            _validate_scene_socket("tcp://127.0.0.1")

    def test_rejects_garbage(self):
        with pytest.raises(SystemExit):
            _validate_scene_socket("not-a-socket")

    def test_rejects_non_string(self):
        with pytest.raises(SystemExit):
            _validate_scene_socket(None)  # type: ignore[arg-type]


class TestBuildConfigRejectsBadSceneSocket:
    """_build_config propagates the validation failure end-to-end."""

    def test_malformed_scene_socket_fails_fast(self, tmp_path: Path, monkeypatch):
        # Synthesize a valid lovr_bin so we get past the earlier checks.
        lovr_bin = tmp_path / "lovr"
        lovr_bin.write_text("#!/bin/sh\nexit 0\n")
        lovr_bin.chmod(0o755)
        xr_app_dir = tmp_path / "xr_app"
        xr_app_dir.mkdir()

        yaml_path = tmp_path / "render_mcp.yaml"
        yaml_path.write_text("")  # _build_config takes raw dict directly

        raw = {
            "lovr_bin":     str(lovr_bin),
            "xr_app_dir":   str(xr_app_dir),
            "scene_socket": "ipc:///",   # malformed
        }
        with pytest.raises(SystemExit):
            _build_config(yaml_path, raw)


# ── scene state bookkeeping ──────────────────────────────────────────────────


class TestSceneState:
    """Pure-state methods don't touch ZMQ — they're synchronous and cheap."""

    def _disp(self, tmp_path: Path) -> SceneDispatcher:
        cfg = Config(
            lovr_bin=Path("/nonexistent"), xr_app_dir=tmp_path,
            scene_socket=_unique_ipc(tmp_path),
            cloudxr_env_file=None, host="127.0.0.1", port=0,
        )
        return SceneDispatcher(cfg, contextlib.AsyncExitStack())

    def test_add_assigns_unique_ids(self, tmp_path: Path):
        disp = self._disp(tmp_path)
        try:
            a = disp.add("sphere", {"x": 0, "y": 0, "z": 0}, {"r": 1, "g": 0, "b": 0}, 0.1)
            b = disp.add("sphere", {"x": 1, "y": 0, "z": 0}, {"r": 0, "g": 1, "b": 0}, 0.2)
            c = disp.add("box",    {"x": 0, "y": 1, "z": 0}, {"r": 0, "g": 0, "b": 1}, 0.3)
            assert a == "sphere-0"
            assert b == "sphere-1"
            assert c == "box-0"
        finally:
            disp.close()

    def test_update_merges_position_partially(self, tmp_path: Path):
        disp = self._disp(tmp_path)
        try:
            oid = disp.add("sphere", {"x": 0, "y": 1, "z": 2}, {"r": 1, "g": 1, "b": 1}, 0.1)
            disp.update(oid, {"position": {"y": 5.0}})
            obj = disp.get_object(oid)
            assert obj["position"] == {"x": 0, "y": 5.0, "z": 2}
        finally:
            disp.close()

    def test_remove_returns_false_on_unknown_id(self, tmp_path: Path):
        disp = self._disp(tmp_path)
        try:
            # Call outside the assert — `python -O` strips assert bodies.
            removed = disp.remove("does-not-exist")
            assert removed is False
        finally:
            disp.close()

    def test_snapshots_reflect_lifecycle(self, tmp_path: Path):
        disp = self._disp(tmp_path)
        try:
            assert disp.health_snapshot() == {
                "status": "ok", "lovr_started": False,
                "spawn_error": None, "render_drops": 0,
            }
            disp.add("sphere", {"x": 0, "y": 0, "z": 0}, {"r": 1, "g": 1, "b": 1}, 0.1)
            snap = disp.scene_snapshot()
            assert len(snap["objects"]) == 1 and snap["objects"][0]["id"] == "sphere-0"
        finally:
            disp.close()


# ── ZMQ forward wire format ──────────────────────────────────────────────────


@_asyncio
class TestForwardWireFormat:
    async def test_forward_emits_msgpack_op_value_frame(self, dispatcher):
        disp, pull = dispatcher
        result = await disp.forward("scene.add", {"id": "sphere-0", "type": "sphere"})
        assert result == {"ok": True}
        msg = await _recv_op(pull)
        assert msg == {"op": "scene.add", "value": {"id": "sphere-0", "type": "sphere"}}

    async def test_forward_drops_when_lovr_not_started(self, tmp_path: Path):
        cfg = Config(
            lovr_bin=Path("/nonexistent"), xr_app_dir=tmp_path,
            scene_socket=_unique_ipc(tmp_path),
            cloudxr_env_file=None, host="127.0.0.1", port=0,
        )
        disp = SceneDispatcher(cfg, contextlib.AsyncExitStack())
        try:
            assert disp._lovr_started is False
            result = await disp.forward("scene.add", {"id": "x"})
            assert result == {"ok": False, "reason": "not_started"}
            assert disp.health_snapshot()["render_drops"] == 1
        finally:
            disp.close()


# ── FastMCP tool surface end-to-end through the dispatcher ───────────────────


@_asyncio
class TestMcpToolsThroughDispatcher:
    async def test_add_primitive_pushes_scene_add_op(self, dispatcher):
        disp, pull = dispatcher
        mcp = build_mcp(disp)
        result = await mcp.call_tool("add_primitive", {
            "prim_type": "sphere",
            "x": 0.0, "y": 1.5, "z": -2.0,
            "r": 1.0, "g": 0.0, "b": 0.0,
            "size": 0.25,
        })
        body = result.structured_content
        assert body["id"] == "sphere-0" and body["ok"] is True

        msg = await _recv_op(pull)
        assert msg["op"] == "scene.add"
        v = msg["value"]
        assert v["id"] == "sphere-0"
        assert v["type"] == "sphere"
        assert v["position"] == [0.0, 1.5, -2.0]
        assert v["color"]    == [1.0, 0.0, 0.0]
        assert v["size"]     == 0.25

    async def test_remove_primitive_pushes_scene_remove_op(self, dispatcher):
        disp, pull = dispatcher
        mcp = build_mcp(disp)
        await mcp.call_tool("add_primitive", {"prim_type": "box"})
        _ = await _recv_op(pull)  # drain add

        result = await mcp.call_tool("remove_primitive", {"obj_id": "box-0"})
        assert result.structured_content == {"ok": True}
        msg = await _recv_op(pull)
        assert msg == {"op": "scene.remove", "value": {"id": "box-0"}}

    async def test_remove_unknown_id_does_not_push(self, dispatcher):
        disp, pull = dispatcher
        mcp = build_mcp(disp)
        result = await mcp.call_tool("remove_primitive", {"obj_id": "ghost"})
        assert result.structured_content == {"ok": False, "reason": "not_found"}
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(pull.recv(), timeout=0.2)

    async def test_update_primitive_partial_position_update(self, dispatcher):
        disp, pull = dispatcher
        mcp = build_mcp(disp)
        await mcp.call_tool("add_primitive", {"prim_type": "sphere", "y": 1.0})
        _ = await _recv_op(pull)

        await mcp.call_tool("update_primitive", {"obj_id": "sphere-0", "y": 2.5})
        msg = await _recv_op(pull)
        assert msg["op"] == "scene.update"
        # Position is rewritten with the merged state — y bumped, x/z preserved.
        assert msg["value"]["id"] == "sphere-0"
        assert msg["value"]["position"][1] == 2.5

    async def test_get_health_and_scene_state(self, dispatcher):
        disp, pull = dispatcher
        mcp = build_mcp(disp)
        await mcp.call_tool("add_primitive", {"prim_type": "sphere"})
        _ = await _recv_op(pull)

        scene = await mcp.call_tool("get_scene_state", {})
        ids = [o["id"] for o in scene.structured_content["objects"]]
        assert ids == ["sphere-0"]

        health = await mcp.call_tool("get_health", {})
        assert health.structured_content["lovr_started"] is True
        assert health.structured_content["render_drops"] == 0


# ── Tech debt #2: watch-task cleanup ─────────────────────────────────────────


class _FakeLovrProc:
    """Stand-in for ManagedProcess that exits only when told (or never)."""
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
        # Records whether this launch's context was torn down — the leak in
        # issue #196 is exactly that this never ran until process shutdown.
        self.exited = False

    async def __aenter__(self) -> _FakeLovrProc:
        return self.proc

    async def __aexit__(self, *exc) -> None:
        self.exited = True
        return None


@_asyncio
async def test_close_cancels_lovr_watch_task(tmp_path: Path, monkeypatch):
    """Without the fix, _watch() leaks on shutdown. With the fix, close() cancels it."""
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
        host             = "127.0.0.1",
        port             = 0,
    )

    # Swap ManagedProcess out before SceneDispatcher.start_lovr_once runs.
    monkeypatch.setattr(render_main, "ManagedProcess",
                        lambda *a, **kw: _FakeManagedProcessCtx())

    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    try:
        disp = SceneDispatcher(cfg, stack)
        result = await disp.start_lovr_once()
        assert result == {"status": "started"}
        assert disp._watch_task is not None
        assert not disp._watch_task.done()

        # close() must cancel the watch task so it can't leak past teardown.
        disp.close()
        # Let the cancellation propagate.
        with contextlib.suppress(asyncio.CancelledError):
            await disp._watch_task
        assert disp._watch_task.done()
    finally:
        await stack.__aexit__(None, None, None)


@_asyncio
async def test_lovr_respawn_closes_previous_launch_context(tmp_path: Path, monkeypatch):
    """Issue #196: each LOVR launch's ManagedProcess context must be torn down
    on respawn instead of accumulating in the app-lifetime stack.

    Without the fix the previous context's ``__aexit__`` (pipe-task cancel +
    log-sink close) only runs at whole-process shutdown, so N restarts leak
    N-1 contexts. With the fix, ``_watch`` closes the per-launch stack as soon
    as the child exits, before the next ``start_lovr_once``.
    """
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
        host             = "127.0.0.1",
        port             = 0,
    )

    created: list[_FakeManagedProcessCtx] = []

    def _make_ctx(*_a, **_kw) -> _FakeManagedProcessCtx:
        ctx = _FakeManagedProcessCtx()
        created.append(ctx)
        return ctx

    monkeypatch.setattr(render_main, "ManagedProcess", _make_ctx)

    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    try:
        disp = SceneDispatcher(cfg, stack)

        # First launch.
        assert await disp.start_lovr_once() == {"status": "started"}
        assert len(created) == 1
        assert disp._launch_stack is not None
        assert created[0].exited is False  # live

        # Simulate the LOVR child exiting and let _watch run to completion.
        created[0].proc.trigger_exit()
        await disp._watch_task

        # The previous launch context is torn down on respawn — not leaked.
        assert created[0].exited is True
        assert disp._launch_stack is None
        assert disp._lovr_started is False

        # Second launch reuses a fresh context; the first stays closed (no
        # accumulation), the second is live.
        assert await disp.start_lovr_once() == {"status": "started"}
        assert len(created) == 2
        assert created[0].exited is True
        assert created[1].exited is False
        assert sum(c.exited for c in created) == 1  # only the dead one closed

        disp.close()
        with contextlib.suppress(asyncio.CancelledError):
            await disp._watch_task
    finally:
        await stack.__aexit__(None, None, None)

    # Safety net: the still-live second context is closed when the
    # app-lifetime stack unwinds at shutdown.
    assert created[1].exited is True


# ── Issue #198: scene resync survives a late LOVR connect ─────────────────────


@_asyncio
async def test_resync_delivers_after_late_peer_connect(tmp_path: Path):
    """Resync must reach LOVR even though LOVR connects AFTER it starts.

    The old path routed resync through ``forward()`` (NOBLOCK); a PUSH with no
    connected peer returns EAGAIN immediately (SNDHWM only buffers after a peer
    attaches), so every restore message was dropped. The blocking resync queues
    once LOVR's PULL connects.
    """
    sock_path = _unique_ipc(tmp_path)
    cfg = Config(
        lovr_bin=Path("/nonexistent"), xr_app_dir=tmp_path,
        scene_socket=sock_path, cloudxr_env_file=None,
        host="127.0.0.1", port=0,
    )
    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    pull = None
    try:
        disp = SceneDispatcher(cfg, stack)
        # Two primitives "carried over" from a previous LOVR session.
        a = disp.add("sphere", {"x": 0, "y": 0, "z": 0}, {"r": 1, "g": 0, "b": 0}, 0.1)
        b = disp.add("box",    {"x": 1, "y": 2, "z": 3}, {"r": 0, "g": 1, "b": 0}, 0.2)

        # Start the resync with NO peer connected. The old code would have
        # dropped both messages instantly; the fix blocks until a peer attaches.
        resync_task = asyncio.create_task(disp._resync_scene())
        await asyncio.sleep(0.1)
        assert not resync_task.done()  # waiting for LOVR, not dropping

        # LOVR connects late.
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
    """PR #219 review nit: while the post-spawn resync is parked waiting for
    LOVR to connect, ``_lovr_started`` stays False, so a concurrent live
    ``forward()`` fast-drops as ``not_started`` instead of contending on the
    shared PUSH socket behind the parked (blocking) resync send. The flag flips
    to True only once resync completes (LOVR connected and scene restored).
    """
    sock_path = _unique_ipc(tmp_path)
    lovr_bin  = tmp_path / "lovr.sh"
    lovr_bin.write_text("#!/bin/sh\nsleep 999\n")
    lovr_bin.chmod(0o755)
    xr_app_dir = tmp_path / "xr_app"
    xr_app_dir.mkdir()
    cfg = Config(
        lovr_bin=lovr_bin, xr_app_dir=xr_app_dir, scene_socket=sock_path,
        cloudxr_env_file=None, host="127.0.0.1", port=0,
    )
    monkeypatch.setattr(render_main, "ManagedProcess",
                        lambda *a, **kw: _FakeManagedProcessCtx())

    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    pull = None
    try:
        disp = SceneDispatcher(cfg, stack)
        disp.add("sphere", {"x": 0, "y": 0, "z": 0}, {"r": 1, "g": 0, "b": 0}, 0.1)

        # start_lovr_once parks inside _resync_scene (blocking send, no peer).
        start_task = asyncio.create_task(disp.start_lovr_once())
        await asyncio.sleep(0.1)
        assert not start_task.done()         # parked waiting for LOVR
        assert disp._lovr_started is False   # not advertised during resync

        # A live op in this window must fast-drop, not queue behind the parked
        # send. (Before the fix, _lovr_started was True here and forward() would
        # attempt the contended NOBLOCK send.)
        res = await disp.forward("scene.update", {"id": "sphere-0", "x": 1})
        assert res == {"ok": False, "reason": "not_started"}

        # LOVR connects → resync drains → start_lovr_once finishes.
        ctx = zmq.asyncio.Context.instance()
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.LINGER, 0)
        pull.connect(sock_path)
        _ = await _recv_op(pull, timeout=2.0)            # the resync scene.add
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
    """A LOVR that never connects must not wedge the spawn — resync returns
    after the deadline instead of blocking forever."""
    monkeypatch.setattr(render_main, "_RESYNC_TIMEOUT_S", 0.2)
    sock_path = _unique_ipc(tmp_path)
    cfg = Config(
        lovr_bin=Path("/nonexistent"), xr_app_dir=tmp_path,
        scene_socket=sock_path, cloudxr_env_file=None,
        host="127.0.0.1", port=0,
    )
    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    try:
        disp = SceneDispatcher(cfg, stack)
        disp.add("sphere", {"x": 0, "y": 0, "z": 0}, {"r": 1, "g": 1, "b": 1}, 0.1)

        start = asyncio.get_running_loop().time()
        # No peer ever connects; must return ~promptly, not hang.
        await asyncio.wait_for(disp._resync_scene(), timeout=3.0)
        elapsed = asyncio.get_running_loop().time() - start
        assert elapsed < 2.0
    finally:
        disp.close()
        await stack.__aexit__(None, None, None)
