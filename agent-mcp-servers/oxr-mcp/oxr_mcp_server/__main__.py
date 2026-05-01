# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
oxr-mcp — OpenXR tracking MCP adapter.

Pure FastMCP. Opens a SECOND OpenXR session against CloudXR in headless mode
(XR_MND_HEADLESS) so the rendering OpenXR client (e.g. LOVR via render-mcp)
keeps full ownership of frame submission while we read pose. Both sessions
co-exist because the headless one never submits frames.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  get_head_pose() → dict
      ``{is_valid, position, orientation, ts, error?}``. Position is metres
      in world space (OpenXR Y-up); orientation is a unit quaternion
      (qx, qy, qz, qw). ``is_valid: false`` (with optional ``error`` reason)
      until tracking is established — callers should retry rather than
      treat it as a hard failure.

  get_health() → dict
      ``{status, session_open, open_attempts, last_open_error}``.

Pose is fetched fresh per request via xrLocateSpace; no background polling,
no staleness.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn
import yaml
from fastmcp import FastMCP

import isaacteleop.deviceio as deviceio
import isaacteleop.oxr as oxr

from xr_ai_launcher import (
    load_cloudxr_env,
    wait_for_cloudxr_env,
    wait_for_cloudxr_runtime_started,
)

log = logging.getLogger("oxr_mcp_server")

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "oxr_mcp_server.yaml"


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    host:             str
    port:             int
    cloudxr_env_file: Path | None


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
    env_file_raw = raw.get("cloudxr_env_file")
    env_file: Path | None = None
    if env_file_raw:
        env_file = _resolve(yaml_dir, env_file_raw)
    return Config(
        host             = raw.get("host", "0.0.0.0"),
        port             = int(raw.get("port", 8230)),
        cloudxr_env_file = env_file,
    )


# ── Pose source ──────────────────────────────────────────────────────────────

class PoseSource:
    """Headless OpenXR session + HeadTracker, returns fresh pose per request.

    Session opening is deferred to the first ``get_pose()`` call: CloudXR
    returns ``XR_ERROR_FORM_FACTOR_UNAVAILABLE`` from ``xrGetSystem`` until a
    streaming client connects. Failed opens are retried on each subsequent
    request.
    """

    # Log open-failure on attempts 1, 2, 4, 8, …, 128, then every 128.
    _OPEN_LOG_AT_ATTEMPTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tracker = deviceio.HeadTracker()
        self._oxr_session: oxr.OpenXRSession | None = None
        self._dev_session: deviceio.DeviceIOSession | None = None
        self._open_attempts: int = 0
        self._last_open_error: str | None = None
        self._next_log_idx: int = 0

    def _try_open(self) -> str | None:
        """Open sessions if needed. Returns error string on failure, None on success.
        Caller must hold the lock."""
        if self._oxr_session is not None:
            return None
        self._open_attempts += 1
        try:
            exts = deviceio.DeviceIOSession.get_required_extensions([self._tracker])
            sess = oxr.OpenXRSession("oxr-mcp", extensions=list(exts))
            sess.__enter__()
            handles = sess.get_handles()
            dev = deviceio.DeviceIOSession.run([self._tracker], handles)
            dev.__enter__()
            self._oxr_session = sess
            self._dev_session = dev
            log.info("oxr-mcp: OpenXR + DeviceIO sessions opened "
                     "(attempt=%d, instance=%#x session=%#x)",
                     self._open_attempts, handles.instance, handles.session)
            self._last_open_error = None
            return None
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            self._last_open_error = err
            self._maybe_log_open_failure(err)
            return err

    def _maybe_log_open_failure(self, err: str) -> None:
        schedule = self._OPEN_LOG_AT_ATTEMPTS
        if self._next_log_idx < len(schedule):
            if self._open_attempts >= schedule[self._next_log_idx]:
                self._next_log_idx += 1
                self._emit_open_failure_log(err)
        elif self._open_attempts % schedule[-1] == 0:
            self._emit_open_failure_log(err)

    def _emit_open_failure_log(self, err: str) -> None:
        log.warning("oxr-mcp: open attempt %d failed: %s "
                    "(form factor not yet available — waiting for streaming client)",
                    self._open_attempts, err)

    def health_snapshot(self) -> dict:
        return {
            "status":          "ok",
            "session_open":    self._oxr_session is not None,
            "open_attempts":   self._open_attempts,
            "last_open_error": self._last_open_error,
        }

    def close(self) -> None:
        with self._lock:
            if self._dev_session is not None:
                try:
                    self._dev_session.__exit__(None, None, None)
                except Exception:
                    log.exception("oxr-mcp: error closing DeviceIOSession")
                self._dev_session = None
            if self._oxr_session is not None:
                try:
                    self._oxr_session.__exit__(None, None, None)
                except Exception:
                    log.exception("oxr-mcp: error closing OpenXRSession")
                self._oxr_session = None

    def get_pose(self) -> dict:
        """Pose snapshot. Returns ``is_valid=False`` (with optional ``error``)
        when the session can't open yet, so callers can distinguish "no pose
        yet" from a real failure."""
        with self._lock:
            if self._dev_session is None:
                err = self._try_open()
                if err is not None:
                    return {
                        "is_valid":    False,
                        "position":    [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                        "ts":          int(time.time() * 1000),
                        "error":       f"session_not_ready: {err}",
                    }
            self._dev_session.update()
            tracked = self._tracker.get_head(self._dev_session)
            data = tracked.data
            pose = data.pose
            return {
                "is_valid":    bool(data.is_valid),
                "position":    [float(pose.position.x),
                                float(pose.position.y),
                                float(pose.position.z)],
                "orientation": [float(pose.orientation.x),
                                float(pose.orientation.y),
                                float(pose.orientation.z),
                                float(pose.orientation.w)],
                "ts":          int(time.time() * 1000),
            }


# ── MCP tool surface ──────────────────────────────────────────────────────────

def build_mcp(source: PoseSource) -> FastMCP:
    mcp = FastMCP("oxr-mcp")

    @mcp.tool()
    async def get_head_pose() -> dict:
        """
        Return the user's current head pose as observed by CloudXR.

        Result shape::

            {
              "is_valid":    bool,
              "position":    [x, y, z],         # metres, world-space
              "orientation": [qx, qy, qz, qw],  # unit quaternion
              "ts":          <ms since Unix epoch>,
              "error":       str                # ONLY on failure paths
            }

        Coordinate convention is OpenXR's right-handed Y-up world: +x right,
        +y up, -z forward. ``is_valid`` is False when the headset has not
        yet established tracking (CloudXR still spinning up, or headset off).
        Treat that as "wait and retry", not a hard failure — the optional
        ``error`` field carries a short reason string.
        """
        return await asyncio.get_running_loop().run_in_executor(None, source.get_pose)

    @mcp.tool()
    async def get_health() -> dict:
        """Server status. ``session_open`` is True once the headless OpenXR
        session has been established — typically once the first ``get_head_pose``
        call after the streaming client connects."""
        return source.health_snapshot()

    return mcp


# ── Entry point ───────────────────────────────────────────────────────────────

async def _serve(cfg: Config) -> None:
    if cfg.cloudxr_env_file:
        log.info("oxr-mcp: waiting for cloudxr env file at %s", cfg.cloudxr_env_file)
        if await wait_for_cloudxr_env(cfg.cloudxr_env_file, log_prefix="oxr-mcp"):
            load_cloudxr_env(cfg.cloudxr_env_file)
            log.info("oxr-mcp: waiting for cloudxr runtime to be ready")
            if not await wait_for_cloudxr_runtime_started(log_prefix="oxr-mcp"):
                log.error("oxr-mcp: cloudxr runtime never became ready — pose will be unavailable")
        else:
            log.error("oxr-mcp: cloudxr env file timed out — pose will be unavailable")
    else:
        log.warning("oxr-mcp: no cloudxr_env_file configured — using whatever XR_RUNTIME_JSON is set in env")

    source = PoseSource()
    log.info("oxr-mcp: ready to serve get_head_pose (session opens lazily on first request)")
    try:
        app = build_mcp(source).http_app(path="/mcp")
        uv_cfg = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
        server = uvicorn.Server(uv_cfg)
        log.info("oxr-mcp  mcp=/mcp  port=%d", cfg.port)
        await server.serve()
    finally:
        await asyncio.get_running_loop().run_in_executor(None, source.close)
        log.info("oxr-mcp: stopped")


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
        sys.exit(f"oxr-mcp: config file not found: {yaml_path}")
    raw = _load_raw(yaml_path)
    cfg = _build_config(yaml_path, raw)

    asyncio.run(_serve(cfg))


if __name__ == "__main__":
    run()
