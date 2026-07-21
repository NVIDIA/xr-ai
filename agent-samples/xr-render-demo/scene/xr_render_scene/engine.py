# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XR render demo scene engine and LOVR lifecycle.

Manages a scene of typed 3D primitives and forwards state changes to LOVR
(the OpenXR rendering app) as msgpack over ZMQ PUSH (``scene_socket``).

  start_xr                                    spawn LOVR (idempotent)
  add_primitive(...)                          add a new primitive; returns assigned id
  update_primitive(id, ...)                   partially update an existing primitive
  remove_primitive(id)                        remove a primitive from the scene
  get_scene_state                             current state of all scene objects
  get_health                                  {status, lovr_started, spawn_error, render_drops}

LOVR can't be spawned at process start because CloudXR returns
``XR_ERROR_FORM_FACTOR_UNAVAILABLE`` from ``xrGetSystem`` until a streaming
client has connected; spawning early lands LOVR in the desktop simulator
forever. Callers should invoke ``start_xr`` only after seeing the streaming
client come up.
"""
from __future__ import annotations

import asyncio
import contextlib
import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import yaml
import zmq
import zmq.asyncio
from loguru import logger
from xr_ai_launcher import ManagedProcess, load_cloudxr_env

# Max seconds the post-spawn scene resync waits for LOVR to connect its PULL
# socket. LOVR normally attaches within a second or two of spawn; the bound
# stops a LOVR that never connects from wedging the spawn path indefinitely.
_RESYNC_TIMEOUT_S = 10.0


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    lovr_bin:         Path
    xr_app_dir:       Path
    scene_socket:     str
    cloudxr_env_file: Path | None
    endpoint:         str


def _load_raw(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _build_config(yaml_path: Path, raw: dict) -> Config:
    yaml_dir = yaml_path.resolve().parent

    # LOVR binary: scene_service.yaml lovr_bin > $LOVR_BIN > fail.
    lovr_bin_raw = raw.get("lovr_bin") or os.environ.get("LOVR_BIN")
    if not lovr_bin_raw:
        sys.exit(
            "xr-render-scene: LOVR binary not configured.\n"
            "  Set LOVR_BIN in the environment, or 'lovr_bin: <path>' in scene_service.yaml.\n"
            "  Point it at your existing LOVR build (e.g. ~/hub/lovr/build/bin/lovr)."
        )
    lovr_bin = Path(lovr_bin_raw).expanduser()
    if not lovr_bin.exists():
        sys.exit(f"xr-render-scene: LOVR binary not found at {lovr_bin}")
    if not os.access(lovr_bin, os.X_OK):
        sys.exit(f"xr-render-scene: LOVR binary at {lovr_bin} is not executable")

    xr_app_dir = _resolve(yaml_dir, raw.get("xr_app_dir", "./lovr"))
    if not xr_app_dir.is_dir():
        sys.exit(f"xr-render-scene: xr_app_dir {xr_app_dir} is not a directory")

    env_file_raw = raw.get("cloudxr_env_file")
    env_file: Path | None = None
    if env_file_raw:
        env_file = _resolve(yaml_dir, env_file_raw)

    scene_socket = raw.get("scene_socket", "ipc:///tmp/xr_render_scene")
    _validate_scene_socket(scene_socket)

    return Config(
        lovr_bin         = lovr_bin,
        xr_app_dir       = xr_app_dir,
        scene_socket     = scene_socket,
        cloudxr_env_file = env_file,
        endpoint         = raw.get("endpoint", "tcp://0.0.0.0:8320"),
    )


# ZMQ transport prefixes we accept for the LOVR-facing PUSH socket. The ipc
# branch requires a non-empty path under the leading slash so we reject
# `ipc:///` — that's the exact typo that produces a cryptic bind() error
# deep inside pyzmq.
_SCENE_SOCKET_RE = re.compile(
    r"^(?:ipc://(?:/[^/].*|[^/].*)|tcp://[^/]+:\d+|inproc://.+)$"
)


def _validate_scene_socket(value: str) -> None:
    """Fail fast on malformed scene_socket URLs (e.g. 'ipc:///' with no path).

    Catches the common typos before pyzmq's bind() raises an opaque error.
    """
    if not isinstance(value, str) or not _SCENE_SOCKET_RE.match(value):
        sys.exit(
            f"xr-render-scene: invalid scene_socket {value!r}.\n"
            "  Expected ipc://<path>, tcp://<host>:<port>, or inproc://<name>."
        )


def _find_bundled_libzmq() -> Path | None:
    """Locate the libzmq shared object pyzmq ships in its wheel for LOVR FFI."""
    site_pkgs = Path(zmq.__file__).resolve().parent.parent
    for candidate_dir in (site_pkgs / "pyzmq.libs", site_pkgs):
        if not candidate_dir.is_dir():
            continue
        matches = sorted(
            glob.glob(str(candidate_dir / "libzmq*.so*"))
            + glob.glob(str(candidate_dir / "**/libzmq*.so*"), recursive=True)
        )
        if matches:
            return Path(matches[0])
    return None


# ── Scene dispatcher ──────────────────────────────────────────────────────────

class SceneDispatcher:
    """ZMQ PUSH to LOVR + LOVR child lifecycle + in-memory scene state.

    Scene state is mirrored in ``_objects`` so ``get_scene_state()`` answers
    immediately without a round-trip to LOVR. All scene mutations go through
    ``add`` / ``update`` / ``remove`` for state bookkeeping and then through
    ``forward()`` to push the op to LOVR.
    """

    def __init__(self, cfg: Config, stack: contextlib.AsyncExitStack) -> None:
        self._cfg   = cfg
        self._stack = stack

        ctx = zmq.asyncio.Context.instance()
        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        # SNDHWM 256: headroom for a burst of scene ops while LOVR's PULL socket
        # connects on spawn; anything beyond that is NOBLOCK-dropped, not queued.
        self._push.setsockopt(zmq.SNDHWM, 256)
        self._push.bind(cfg.scene_socket)
        logger.info("xr-render-scene: bound PUSH on {}", cfg.scene_socket)

        self._lovr_started: bool = False
        self._spawn_lock: asyncio.Lock = asyncio.Lock()
        self._render_drops: int = 0
        self._spawn_error: str | None = None
        self._watch_task: asyncio.Task | None = None
        # Per-launch context for the current LOVR child. Each launch gets its
        # own AsyncExitStack so its ManagedProcess teardown (pipe-task cancel +
        # log-sink close) can run on respawn instead of piling up in the
        # app-lifetime stack until whole-process shutdown.
        self._launch_stack: contextlib.AsyncExitStack | None = None
        # Safety net: close whatever launch context is live when the server
        # shuts down (LOVR still running). Registered once on the app-lifetime
        # stack so it never accumulates per launch.
        self._stack.push_async_callback(self._aclose_live_launch)

        # Scene state: { id → { type, position, color, scale } }
        self._objects: dict[str, dict] = {}
        self._id_counters: dict[str, int] = {}

    async def start_lovr_once(self) -> dict:
        """Spawn LOVR if not already running. Idempotent. Failures are cached."""
        async with self._spawn_lock:
            if self._lovr_started:
                return {"status": "already_started"}
            if self._spawn_error is not None:
                return {"status": "error", "error": self._spawn_error}

            cfg = self._cfg

            if cfg.cloudxr_env_file:
                if not cfg.cloudxr_env_file.exists():
                    msg = (f"cloudxr env file not found: {cfg.cloudxr_env_file}. "
                           "Ensure cloudxr-runtime starts before xr-render-scene.")
                    self._spawn_error = msg
                    logger.error("xr-render-scene: {}", msg)
                    return {"status": "error", "error": msg}
                load_cloudxr_env(cfg.cloudxr_env_file)
                logger.info("xr-render-scene: cloudxr env loaded from {}", cfg.cloudxr_env_file)
            else:
                logger.warning(
                    "xr-render-scene: no cloudxr_env_file configured — LOVR will use "
                    "whatever OpenXR runtime is registered on this machine"
                )

            # AppImages need FUSE by default; --appimage-extract-and-run avoids it.
            lovr_cmd: list[str] = [str(cfg.lovr_bin)]
            if cfg.lovr_bin.suffix.lower() == ".appimage":
                lovr_cmd.append("--appimage-extract-and-run")
            lovr_cmd.append(str(cfg.xr_app_dir))

            logger.info(
                "xr-render-scene: starting LOVR  bin={}  app={}", cfg.lovr_bin, cfg.xr_app_dir,
            )
            launch_stack = contextlib.AsyncExitStack()
            lovr_proc = await launch_stack.enter_async_context(
                ManagedProcess("lovr", lovr_cmd, cwd=cfg.xr_app_dir)
            )
            self._launch_stack = launch_stack

            async def _watch() -> None:
                rc = await lovr_proc.wait()
                logger.warning(
                    "xr-render-scene: LOVR child exited (rc={}) — "
                    "resetting lovr_started so next start_xr respawns it", rc,
                )
                # Tear down this launch's context (cancel the two pipe-forward
                # tasks, close the log sink; terminate is a no-op since the
                # child already exited) BEFORE allowing a respawn, so dead
                # contexts don't accumulate in the app-lifetime stack.
                with contextlib.suppress(Exception):
                    await launch_stack.aclose()
                if self._launch_stack is launch_stack:
                    self._launch_stack = None
                self._lovr_started = False

            self._watch_task = asyncio.create_task(_watch(), name="lovr-watch")

            logger.info("xr-render-scene: LOVR spawned (xr.start handled)")
            # Restore the scene BEFORE advertising started. While resync's first
            # (blocking) send is parked waiting for LOVR's PULL to attach,
            # ``_lovr_started`` stays False so concurrent live ``forward()`` ops
            # keep fast-dropping as "not_started" instead of queueing behind the
            # parked send on the shared PUSH socket. It also makes the flag mean
            # what it says: LOVR connected AND scene restored.
            await self._resync_scene()
            # Only advertise started if LOVR is still alive — it may have exited
            # during the resync wait, in which case ``_watch`` has already run to
            # completion (its post-wait body has no awaits) and cleared the flag,
            # so we must not re-set it True.
            if self._watch_task is not None and not self._watch_task.done():
                self._lovr_started = True
            return {"status": "started"}

    async def _resync_scene(self) -> None:
        """Re-push scene.add for every known primitive after a LOVR (re)start.

        LOVR has just been spawned and has NOT yet connected its PULL socket.
        A PUSH socket with zero connected peers does NOT buffer up to SNDHWM —
        it returns ``EAGAIN`` immediately under ``NOBLOCK`` — so routing the
        resync through the live ``forward()`` path silently drops the entire
        restore. Send these as *blocking* sends instead, so each
        one queues the moment LOVR attaches, bounded by an overall deadline so
        a LOVR that never connects can't wedge the spawn (we hold the spawn
        lock here)."""
        objs = list(self._objects.items())
        if not objs:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _RESYNC_TIMEOUT_S
        sent = 0
        for obj_id, obj in objs:
            pos = obj["position"]
            col = obj["color"]
            payload = msgpack.packb(
                {"op": "scene.add", "value": {
                    "id":       obj_id,
                    "type":     obj["type"],
                    "position": [pos["x"], pos["y"], pos["z"]],
                    "color":    [col["r"], col["g"], col["b"]],
                    "size":     obj["size"],
                }},
                use_bin_type=True,
            )
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                # Blocking send (no NOBLOCK): the first one waits until LOVR's
                # PULL connects; the rest queue (up to SNDHWM) and return fast.
                await asyncio.wait_for(self._push.send(payload), timeout=remaining)
            except asyncio.TimeoutError:
                break
            sent += 1
            logger.debug("xr-render-scene: resync  id={}", obj_id)

        if sent == len(objs):
            if sent:
                logger.info("xr-render-scene: scene resync sent {} primitive(s) to LOVR", sent)
        else:
            logger.warning(
                "xr-render-scene: scene resync incomplete ({}/{} primitives) — LOVR did "
                "not connect within {:.0f}s", sent, len(objs), _RESYNC_TIMEOUT_S,
            )

    # ── scene state ───────────────────────────────────────────────────────────

    def _make_id(self, prim_type: str) -> str:
        n = self._id_counters.get(prim_type, 0)
        self._id_counters[prim_type] = n + 1
        return f"{prim_type}-{n}"

    def add(self, prim_type: str, position: dict, color: dict,
            size: float) -> str:
        """Add a new object; return its server-assigned id."""
        obj_id = self._make_id(prim_type)
        self._objects[obj_id] = {
            "type":     prim_type,
            "position": dict(position),
            "color":    dict(color),
            "size":     size,
        }
        return obj_id

    def get_object(self, obj_id: str) -> dict | None:
        """Return the stored object for *obj_id*, or None if not found."""
        return self._objects.get(obj_id)

    def update(self, obj_id: str, props: dict) -> bool:
        """Partially merge *props* into an existing object. Returns False if
        the id is unknown."""
        obj = self._objects.get(obj_id)
        if obj is None:
            return False
        for k, v in props.items():
            if isinstance(v, dict) and isinstance(obj.get(k), dict):
                obj[k].update(v)
            else:
                obj[k] = v
        return True

    def remove(self, obj_id: str) -> bool:
        return self._objects.pop(obj_id, None) is not None

    def health_snapshot(self) -> dict:
        return {
            "status":       "ok",
            "lovr_started": self._lovr_started,
            "spawn_error":  self._spawn_error,
            "render_drops": self._render_drops,
        }

    def scene_snapshot(self) -> dict:
        return {
            "objects": [{"id": obj_id, **obj} for obj_id, obj in self._objects.items()]
        }

    # ── wire ──────────────────────────────────────────────────────────────────

    async def forward(self, op: str, value: Any) -> dict:
        """msgpack-encode ``{op, value}`` and PUSH to LOVR. Drops until
        ``start_xr`` has succeeded."""
        if not self._lovr_started:
            self._render_drops += 1
            if self._render_drops % 200 == 1:
                logger.debug(
                    "xr-render-scene: dropping op {!r} (LOVR not started) — drops={}",
                    op, self._render_drops,
                )
            return {"ok": False, "reason": "not_started"}

        payload = msgpack.packb({"op": op, "value": value}, use_bin_type=True)
        try:
            await self._push.send(payload, zmq.NOBLOCK)
            return {"ok": True}
        except zmq.Again:
            return {"ok": False, "reason": "backpressure"}

    async def _aclose_live_launch(self) -> None:
        """Close the current launch context, if any.

        Registered once on the app-lifetime stack so a LOVR child still running
        at server shutdown is torn down. On a normal LOVR exit ``_watch`` has
        already closed and cleared the launch stack, so this is a no-op.
        """
        launch_stack, self._launch_stack = self._launch_stack, None
        if launch_stack is not None:
            with contextlib.suppress(Exception):
                await launch_stack.aclose()

    def close(self) -> None:
        if self._watch_task is not None and not self._watch_task.done():
            self._watch_task.cancel()
        with contextlib.suppress(Exception):
            self._push.close(linger=0)
