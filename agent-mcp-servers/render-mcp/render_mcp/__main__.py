# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
render-mcp — generic OpenXR rendering MCP adapter.

Manages a scene of typed 3D primitives and forwards state changes to LOVR
(the OpenXR rendering app) as msgpack over ZMQ PUSH (``scene_socket``).

  /mcp/start_xr                              spawn LOVR (idempotent)
  /mcp/add_primitive(...)                    add a new primitive; returns assigned id
  /mcp/update_primitive(id, ...)             partially update an existing primitive
  /mcp/remove_primitive(id)                  remove a primitive from the scene
  /mcp/get_scene_state                       current state of all scene objects
  /mcp/get_health                            {status, lovr_started, spawn_error, render_drops}

LOVR can't be spawned at process start because CloudXR returns
``XR_ERROR_FORM_FACTOR_UNAVAILABLE`` from ``xrGetSystem`` until a streaming
client has connected; spawning early lands LOVR in the desktop simulator
forever. Callers should invoke ``start_xr`` only after seeing the streaming
client come up.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import uvicorn
import yaml
import zmq
import zmq.asyncio
from fastmcp import FastMCP
from loguru import logger

from xr_ai_launcher import ManagedProcess, XR_RUNTIME_VAR, load_cloudxr_env
from xr_ai_logging import setup_logging

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "render_mcp.yaml"


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    lovr_bin:         Path
    xr_app_dir:       Path
    scene_socket:     str
    cloudxr_env_file: Path | None
    host:             str
    port:             int


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

    # LOVR binary: render_mcp.yaml lovr_bin > $LOVR_BIN > fail.
    lovr_bin_raw = raw.get("lovr_bin") or os.environ.get("LOVR_BIN")
    if not lovr_bin_raw:
        sys.exit(
            "render-mcp: LOVR binary not configured.\n"
            "  Set LOVR_BIN in the environment, or 'lovr_bin: <path>' in render_mcp.yaml.\n"
            "  Point it at your existing LOVR build (e.g. ~/hub/lovr/build/bin/lovr)."
        )
    lovr_bin = Path(lovr_bin_raw).expanduser()
    if not lovr_bin.exists():
        sys.exit(f"render-mcp: LOVR binary not found at {lovr_bin}")
    if not os.access(lovr_bin, os.X_OK):
        sys.exit(f"render-mcp: LOVR binary at {lovr_bin} is not executable")

    xr_app_dir = _resolve(yaml_dir, raw.get("xr_app_dir", "./xr_app"))
    if not xr_app_dir.is_dir():
        sys.exit(f"render-mcp: xr_app_dir {xr_app_dir} is not a directory")

    env_file_raw = raw.get("cloudxr_env_file")
    env_file: Path | None = None
    if env_file_raw:
        env_file = _resolve(yaml_dir, env_file_raw)

    return Config(
        lovr_bin         = lovr_bin,
        xr_app_dir       = xr_app_dir,
        scene_socket     = raw.get("scene_socket", "ipc:///tmp/xr_render_scene"),
        cloudxr_env_file = env_file,
        host             = raw.get("host", "0.0.0.0"),
        port             = int(raw.get("port", 8220)),
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
        # 8 = enough headroom for a single LLM-round-trip burst, small enough
        # that anything beyond a burst is dropped via NOBLOCK rather than queued.
        # 256 = room for ~5 s of 50 Hz scale messages while LOVR starts up,
        # plus command messages. The old value of 8 let scale messages crowd
        # out scene.add before LOVR connected.
        self._push.setsockopt(zmq.SNDHWM, 256)
        self._push.bind(cfg.scene_socket)
        logger.info("render-mcp: bound PUSH on {}", cfg.scene_socket)

        self._lovr_started: bool = False
        self._spawn_lock: asyncio.Lock = asyncio.Lock()
        self._render_drops: int = 0
        self._spawn_error: str | None = None

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
                           "Ensure cloudxr-runtime starts before render-mcp.")
                    self._spawn_error = msg
                    logger.error("render-mcp: {}", msg)
                    return {"status": "error", "error": msg}
                load_cloudxr_env(cfg.cloudxr_env_file)
                logger.info("render-mcp: cloudxr env loaded from {}", cfg.cloudxr_env_file)
            else:
                logger.warning(
                    "render-mcp: no cloudxr_env_file configured — LOVR will use "
                    "whatever OpenXR runtime is registered on this machine"
                )

            # AppImages need FUSE by default; --appimage-extract-and-run avoids it.
            lovr_cmd: list[str] = [str(cfg.lovr_bin)]
            if cfg.lovr_bin.suffix.lower() == ".appimage":
                lovr_cmd.append("--appimage-extract-and-run")
            lovr_cmd.append(str(cfg.xr_app_dir))

            logger.info(
                "render-mcp: starting LOVR  bin={}  app={}", cfg.lovr_bin, cfg.xr_app_dir,
            )
            lovr_proc = await self._stack.enter_async_context(
                ManagedProcess("lovr", lovr_cmd, cwd=cfg.xr_app_dir)
            )

            async def _watch() -> None:
                rc = await lovr_proc.wait()
                logger.warning(
                    "render-mcp: LOVR child exited (rc={}) — "
                    "resetting lovr_started so next start_xr respawns it", rc,
                )
                self._lovr_started = False

            asyncio.create_task(_watch(), name="lovr-watch")

            self._lovr_started = True
            logger.info("render-mcp: LOVR spawned (xr.start handled)")
            # Resync current scene state into LOVR's ZMQ receive buffer so any
            # previously-added primitives survive a LOVR restart.
            await self._resync_scene()
            return {"status": "started"}

    async def _resync_scene(self) -> None:
        """Push scene.add for every known primitive into LOVR's ZMQ buffer.
        Called after LOVR (re)starts so primitives added in a previous session
        are visible immediately when LOVR connects."""
        for obj_id, obj in list(self._objects.items()):
            pos = obj["position"]
            col = obj["color"]
            await self.forward("scene.add", {
                "id":       obj_id,
                "type":     obj["type"],
                "position": [pos["x"], pos["y"], pos["z"]],
                "color":    [col["r"], col["g"], col["b"]],
                "size":    obj["size"],
            })
            logger.debug("render-mcp: resync  id={}", obj_id)

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
                    "render-mcp: dropping op {!r} (LOVR not started) — drops={}",
                    op, self._render_drops,
                )
            return {"ok": False, "reason": "not_started"}

        payload = msgpack.packb({"op": op, "value": value}, use_bin_type=True)
        try:
            await self._push.send(payload, zmq.NOBLOCK)
            return {"ok": True}
        except zmq.Again:
            return {"ok": False, "reason": "backpressure"}

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._push.close(linger=0)


# ── MCP tool surface ──────────────────────────────────────────────────────────

def build_mcp(disp: SceneDispatcher) -> FastMCP:
    mcp = FastMCP("render-mcp")

    @mcp.tool()
    async def start_xr() -> dict:
        """Spawn LOVR (the OpenXR rendering app) if it hasn't been started.

        Idempotent — returns ``{"status": "already_started"}`` if already up.
        Returns immediately; the CloudXR readiness wait and LOVR launch run in
        a background task. Poll ``get_health`` until ``lovr_started`` flips
        before sending scene ops you can't tolerate dropping.

        Terminal failures are cached so retries fail fast.
        """
        snap = disp.health_snapshot()
        if snap["lovr_started"]:
            return {"status": "already_started"}
        if snap["spawn_error"] is not None:
            return {"status": "error", "error": snap["spawn_error"]}

        async def _spawn() -> None:
            try:
                await disp.start_lovr_once()
            except Exception:
                logger.exception("render-mcp: start_xr crashed")

        asyncio.create_task(_spawn(), name="lovr-spawn")
        return {"status": "starting"}

    @mcp.tool()
    async def add_primitive(
        prim_type: str,
        x: float = 0.0, y: float = 1.6, z: float = -1.5,
        r: float = 0.2, g: float = 0.9, b: float = 1.0,
        size: float = 0.1,
    ) -> dict:
        """Add a new primitive to the scene. Returns its server-assigned id.

        Supported types: sphere, box (others fall back to sphere).
        Position is world-space metres (OpenXR Y-up).
        size is in METRES: radius for spheres, edge half-length for boxes.
          0.05 m = 5 cm (tiny)   0.1 m = 10 cm (default)
          0.3 m = 30 cm          1.0 m = 1 metre (large)

        COLOR — ALWAYS specify all three components together, never partial:
          red    r=1, g=0, b=0      green   r=0, g=0.8, b=0
          blue   r=0, g=0.4, b=1    yellow  r=1, g=1,   b=0
          white  r=1, g=1,   b=1    orange  r=1, g=0.5, b=0
          cyan   r=0, g=0.9, b=1    purple  r=0.6, g=0, b=1
        Omitting any of r/g/b uses the default (0.2/0.9/1.0 = cyan).
        Example: 'red sphere' → r=1.0, g=0.0, b=0.0  (all three required)

        Examples:
          "add a red sphere"     prim_type="sphere", r=1, g=0, b=0
          "add a green cube"     prim_type="box",    r=0, g=0.8, b=0
          "add a blue sphere"    prim_type="sphere", r=0, g=0.4, b=1
        """
        position = {"x": float(x), "y": float(y), "z": float(z)}
        color    = {"r": float(r), "g": float(g), "b": float(b)}
        obj_id   = disp.add(prim_type, position, color, float(size))
        result   = await disp.forward("scene.add", {
            "id": obj_id, "type": prim_type,
            "position": [x, y, z],
            "color":    [r, g, b],
            "size":     size,
        })
        logger.debug("render-mcp: add_primitive id={} type={}", obj_id, prim_type)
        return {"id": obj_id, **result}

    @mcp.tool()
    async def update_primitive(
        obj_id: str,
        prim_type: str | None = None,
        x: float | None = None, y: float | None = None, z: float | None = None,
        r: float | None = None, g: float | None = None, b: float | None = None,
        size: float | None = None,
    ) -> dict:
        """Update one or more properties of an existing primitive (partial update).

        Only the fields you pass change; omitted fields keep their current values.
        All coordinates are world-space metres (OpenXR Y-up).

        TYPE CHANGE: pass prim_type="box" or prim_type="sphere" to convert a
        primitive to a different shape while keeping its position, color, and size.
          "change the sphere to a cube" → obj_id=<id>, prim_type="box"
          "convert the box to a sphere" → obj_id=<id>, prim_type="sphere"

        COLOR: when changing color, ALWAYS pass all three of r, g, b together.
          "make it red"   → r=1.0, g=0.0, b=0.0  (all three required)
          "make it green" → r=0.0, g=0.8, b=0.0

        Examples:
          "change to a cube"       obj_id=<id>, prim_type="box"
          "make it red"            obj_id=<id>, r=1.0, g=0.0, b=0.0
          "move it up 1m"          obj_id=<id>, y=<current_y + 1.0>
          "make it twice as big"   obj_id=<id>, size=<current_size * 2>
        """
        obj = disp.get_object(obj_id)
        if obj is None:
            return {"ok": False, "reason": "not_found"}

        # Apply scalar-property updates first.
        props: dict = {}
        if any(v is not None for v in (x, y, z)):
            props["position"] = {k: v for k, v in (("x", x), ("y", y), ("z", z)) if v is not None}
        if any(v is not None for v in (r, g, b)):
            props["color"] = {k: v for k, v in (("r", r), ("g", g), ("b", b)) if v is not None}
        if size is not None:
            props["size"] = float(size)
        if props:
            disp.update(obj_id, props)

        if prim_type is not None and prim_type != obj["type"]:
            # Type change: remove old, re-add with new type preserving merged state.
            merged = disp.get_object(obj_id)
            p, c = merged["position"], merged["color"]
            disp.remove(obj_id)
            await disp.forward("scene.remove", {"id": obj_id})
            new_id = disp.add(prim_type, p, c, merged["size"])
            result = await disp.forward("scene.add", {
                "id": new_id, "type": prim_type,
                "position": [p["x"], p["y"], p["z"]],
                "color":    [c["r"], c["g"], c["b"]],
                "size":     merged["size"],
            })
            logger.debug("render-mcp: type change {} → {}  new_id={}", obj_id, prim_type, new_id)
            return {"ok": result.get("ok", True), "new_id": new_id}

        if not props:
            return {"ok": True}   # no-op

        # Build wire payload for property-only updates.
        obj  = disp.get_object(obj_id)
        wire: dict = {"id": obj_id}
        if "position" in props:
            p = obj["position"]
            wire["position"] = [p["x"], p["y"], p["z"]]
        if "color" in props:
            c = obj["color"]
            wire["color"] = [c["r"], c["g"], c["b"]]
        if "size" in props:
            wire["size"] = obj["size"]
        return await disp.forward("scene.update", wire)

    @mcp.tool()
    async def remove_primitive(obj_id: str) -> dict:
        """Remove a primitive from the scene by id.

        Example: "remove the sphere" → obj_id=<id from get_scene_state>
        """
        if not disp.remove(obj_id):
            return {"ok": False, "reason": "not_found"}
        return await disp.forward("scene.remove", {"id": obj_id})

    @mcp.tool()
    async def get_scene_state() -> dict:
        """Return the current state of all scene objects.

        Response shape::

            {
              "objects": [
                {
                  "id":       "<string>",
                  "type":     "<string>",
                  "position": {"x": float, "y": float, "z": float},
                  "color":    {"r": float, "g": float, "b": float},
                  "size":    float
                },
                ...
              ]
            }

        All coordinates are world-space metres (OpenXR Y-up).
        """
        return disp.scene_snapshot()

    @mcp.tool()
    async def get_health() -> dict:
        """Server status. Use ``lovr_started`` as the readiness signal after
        ``start_xr``."""
        return disp.health_snapshot()

    return mcp


def build_app(disp: SceneDispatcher):
    """Pure FastMCP app — no REST routes, all ops via MCP tools."""
    return build_mcp(disp).http_app(path="/mcp")


# ── Entry point ───────────────────────────────────────────────────────────────

async def _serve(cfg: Config, ready_file: Path | None = None) -> None:
    os.environ["RENDER_SCENE_SOCKET"] = cfg.scene_socket
    bundled = _find_bundled_libzmq()
    if bundled:
        os.environ["RENDER_ZMQ_LIB"] = str(bundled)
        logger.info("render-mcp: LOVR will load libzmq from {}", bundled)
    else:
        logger.warning("render-mcp: no bundled libzmq found — LOVR will try the system copy")

    async with contextlib.AsyncExitStack() as stack:
        disp = SceneDispatcher(cfg, stack)
        try:
            app = build_app(disp)
            uv_cfg = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
            server = uvicorn.Server(uv_cfg)
            logger.info(
                "render-mcp  mcp=/mcp  port={}  scene_socket={}",
                cfg.port, cfg.scene_socket,
            )
            if ready_file:
                ready_file.touch()
            await server.serve()
        finally:
            disp.close()
            logger.info(
                "render-mcp: stopped (render drops={})",
                disp.health_snapshot()["render_drops"],
            )


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    ns, _ = p.parse_known_args()

    setup_logging("render-mcp")
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    yaml_path = ns.config or _DEFAULT_YAML
    if not yaml_path.exists():
        sys.exit(f"render-mcp: config file not found: {yaml_path}")
    raw = _load_raw(yaml_path)
    cfg = _build_config(yaml_path, raw)

    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
