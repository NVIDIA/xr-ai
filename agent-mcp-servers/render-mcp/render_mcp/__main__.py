# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
render-mcp — OpenXR rendering MCP adapter.

Spawns LOVR (the OpenXR rendering app) on demand and forwards scene ops to
it as msgpack over ZMQ PUSH (``scene_socket``).

  POST /sphere/radius   {"value": F}        per-audio-chunk volume → radius
  /mcp/start_xr                              spawn LOVR (idempotent)
  /mcp/set_sphere_color(r, g, b)             RGB floats in [0, 1]
  /mcp/set_sphere_position(x, y, z)          world-space, metres (OpenXR Y-up)
  /mcp/reset_sphere                          restore default colour + position
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
import logging
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
from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import BaseModel

from xr_ai_launcher import (
    ManagedProcess,
    XR_RUNTIME_VAR,
    load_cloudxr_env,
    wait_for_cloudxr_env,
    wait_for_cloudxr_runtime_started,
)

log = logging.getLogger("render_mcp")

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


# ── Sphere dispatcher ─────────────────────────────────────────────────────────

class SphereDispatcher:
    """ZMQ PUSH to LOVR + the LOVR child lifecycle. All tools funnel through
    ``forward(op, value)``; ops are dropped until ``start_lovr_once`` has run."""

    def __init__(self, cfg: Config, stack: contextlib.AsyncExitStack) -> None:
        self._cfg   = cfg
        self._stack = stack

        ctx = zmq.asyncio.Context.instance()
        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        # 8 = enough headroom for a single LLM-round-trip burst (colour +
        # position + radius), small enough that anything beyond a burst is
        # correctly classified as a stall and dropped via NOBLOCK.
        self._push.setsockopt(zmq.SNDHWM, 8)
        self._push.bind(cfg.scene_socket)
        log.info("render-mcp: bound PUSH on %s", cfg.scene_socket)

        self._lovr_started: bool = False
        self._spawn_lock: asyncio.Lock = asyncio.Lock()
        self._render_drops: int = 0
        # Set once on a terminal cloudxr-readiness timeout so retries fail
        # fast instead of re-running the multi-minute wait.
        self._spawn_error: str | None = None

    async def start_lovr_once(self) -> dict:
        """Spawn LOVR if not already running. Idempotent. Failures are cached."""
        async with self._spawn_lock:
            if self._lovr_started:
                return {"status": "already_started"}
            if self._spawn_error is not None:
                return {"status": "error", "error": self._spawn_error}

            cfg = self._cfg

            if cfg.cloudxr_env_file:
                log.info("render-mcp: waiting for cloudxr env file at %s", cfg.cloudxr_env_file)
                if not await wait_for_cloudxr_env(
                    cfg.cloudxr_env_file, log_prefix="render-mcp",
                ):
                    msg = (f"timed out waiting for {cfg.cloudxr_env_file} "
                           f"({XR_RUNTIME_VAR} missing). cloudxr-runtime did not become ready.")
                    self._spawn_error = msg
                    return {"status": "error", "error": msg}
                load_cloudxr_env(cfg.cloudxr_env_file)

                log.info("render-mcp: waiting for cloudxr runtime to be ready")
                if not await wait_for_cloudxr_runtime_started(log_prefix="render-mcp"):
                    msg = ("timed out waiting for cloudxr runtime_started. "
                           "Check the cloudxr-runtime logs for startup errors.")
                    self._spawn_error = msg
                    return {"status": "error", "error": msg}
                log.info("render-mcp: cloudxr OpenXR runtime is ready")
            else:
                log.warning(
                    "render-mcp: no cloudxr_env_file configured — LOVR will use "
                    "whatever OpenXR runtime is registered on this machine"
                )

            # AppImages need FUSE by default; --appimage-extract-and-run avoids it.
            lovr_cmd: list[str] = [str(cfg.lovr_bin)]
            if cfg.lovr_bin.suffix.lower() == ".appimage":
                lovr_cmd.append("--appimage-extract-and-run")
            lovr_cmd.append(str(cfg.xr_app_dir))

            log.info("render-mcp: starting LOVR  bin=%s  app=%s", cfg.lovr_bin, cfg.xr_app_dir)
            lovr_proc = await self._stack.enter_async_context(
                ManagedProcess("lovr", lovr_cmd, cwd=cfg.xr_app_dir)
            )

            async def _watch() -> None:
                rc = await lovr_proc.wait()
                log.warning("render-mcp: LOVR child exited (rc=%s)", rc)

            asyncio.create_task(_watch(), name="lovr-watch")

            self._lovr_started = True
            log.info("render-mcp: LOVR spawned (xr.start handled)")
            return {"status": "started"}

    def health_snapshot(self) -> dict:
        return {
            "status":       "ok",
            "lovr_started": self._lovr_started,
            "spawn_error":  self._spawn_error,
            "render_drops": self._render_drops,
        }

    async def forward(self, op: str, value: Any) -> dict:
        """msgpack-encode ``{op, value}`` and PUSH to LOVR. Drops until xr.start."""
        if not self._lovr_started:
            self._render_drops += 1
            if self._render_drops % 200 == 1:
                log.debug(
                    "render-mcp: dropping render op %r (LOVR not started yet) — drops=%d",
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

def build_mcp(disp: SphereDispatcher) -> FastMCP:
    mcp = FastMCP("render-mcp")

    @mcp.tool()
    async def start_xr() -> dict:
        """
        Spawn LOVR (the OpenXR rendering app) if it hasn't been started.

        Idempotent — calling again after success returns
        ``{"status": "already_started"}``. Returns immediately; the
        cloudxr-runtime readiness wait + LOVR launch happen in a background
        task. Poll ``get_health`` until ``lovr_started`` flips before sending
        render ops you can't tolerate dropping.

        Failure (cloudxr never became ready) is cached: every subsequent
        call returns the same error rather than re-running the wait.
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
                log.exception("render-mcp: start_xr crashed")

        asyncio.create_task(_spawn(), name="lovr-spawn")
        return {"status": "starting"}

    @mcp.tool()
    async def set_sphere_color(r: float, g: float, b: float) -> dict:
        """Set sphere RGB. Components in [0, 1]."""
        return await disp.forward("sphere.color", [float(r), float(g), float(b)])

    @mcp.tool()
    async def set_sphere_position(x: float, y: float, z: float) -> dict:
        """Set sphere world-space position in metres (OpenXR Y-up; default
        anchor is (0, 1.6, -1.5) — head height, 1.5 m in front of origin)."""
        return await disp.forward("sphere.position", [float(x), float(y), float(z)])

    @mcp.tool()
    async def reset_sphere() -> dict:
        """Restore default sphere colour + position. Radius keeps tracking voice."""
        return await disp.forward("sphere.reset", None)

    @mcp.tool()
    async def get_health() -> dict:
        """Server status. Use ``lovr_started`` as the readiness signal after
        ``start_xr``."""
        return disp.health_snapshot()

    return mcp


# ── Streaming HTTP route ──────────────────────────────────────────────────────

class _RadiusBody(BaseModel):
    value: float


def build_app(disp: SphereDispatcher) -> FastAPI:
    """FastAPI app hosting POST /sphere/radius plus the FastMCP tool
    surface mounted at /mcp. The outer app inherits the inner FastMCP
    app's lifespan so its session-store startup hooks run."""
    mcp_app = build_mcp(disp).http_app(path="/mcp")
    app = FastAPI(title="render-mcp", docs_url=None, redoc_url=None,
                  lifespan=mcp_app.lifespan)

    @app.post("/sphere/radius")
    async def sphere_radius(body: _RadiusBody) -> dict:
        return await disp.forward("sphere.radius", body.value)

    app.mount("/", mcp_app)
    return app


# ── Entry point ───────────────────────────────────────────────────────────────

async def _serve(cfg: Config) -> None:
    os.environ["RENDER_SCENE_SOCKET"] = cfg.scene_socket
    bundled = _find_bundled_libzmq()
    if bundled:
        os.environ["RENDER_ZMQ_LIB"] = str(bundled)
        log.info("render-mcp: LOVR will load libzmq from %s", bundled)
    else:
        log.warning("render-mcp: no bundled libzmq found — LOVR will try the system copy")

    async with contextlib.AsyncExitStack() as stack:
        disp = SphereDispatcher(cfg, stack)
        try:
            app = build_app(disp)
            uv_cfg = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
            server = uvicorn.Server(uv_cfg)
            log.info("render-mcp  http=/sphere/radius  mcp=/mcp  port=%d  scene_socket=%s",
                     cfg.port, cfg.scene_socket)
            await server.serve()
        finally:
            disp.close()
            log.info("render-mcp: stopped (render drops=%d)",
                     disp.health_snapshot()["render_drops"])


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    ns, _ = p.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    yaml_path = ns.config or _DEFAULT_YAML
    if not yaml_path.exists():
        sys.exit(f"render-mcp: config file not found: {yaml_path}")
    raw = _load_raw(yaml_path)
    cfg = _build_config(yaml_path, raw)

    asyncio.run(_serve(cfg))


if __name__ == "__main__":
    run()
