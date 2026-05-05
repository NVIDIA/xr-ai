# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
oxr-mcp — OpenXR tracking MCP adapter.

Pure FastMCP. Opens a SECOND OpenXR session against CloudXR in headless mode
(XR_MND_HEADLESS) so the rendering OpenXR client (e.g. LOVR via render-mcp)
keeps full ownership of frame submission while we read pose.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  get_head_pose() → dict
      LLM-friendly pose with derived spatial vectors — no raw quaternions.
      Fields: is_valid, position {x,y,z}, forward {x,y,z}, right {x,y,z},
      up {x,y,z}, yaw_deg, pitch_deg, ts.

  position_ahead(distance) → dict {x,y,z}
      World position 'distance' metres along the user's gaze direction.

  position_relative(forward, right, up) → dict {x,y,z}
      Convert head-relative offsets (metres) to world-space position.
      forward>0 = ahead, right>0 = right, up>0 = above eye level.

  get_health() → dict
      {status, session_open, open_attempts, last_open_error}.

Pose is fetched fresh per request via xrLocateSpace; no background polling.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger

import isaacteleop.deviceio as deviceio
import isaacteleop.oxr as oxr

from xr_ai_launcher import load_cloudxr_env
from xr_ai_logging import setup_logging

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "oxr_mcp_server.yaml"


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _rotate_vec(
    q: tuple[float, float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate vector *v* by unit quaternion *q* = (qx, qy, qz, qw)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx)


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
            logger.info(
                "oxr-mcp: OpenXR + DeviceIO sessions opened "
                "(attempt={}, instance={:#x} session={:#x})",
                self._open_attempts, handles.instance, handles.session,
            )
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
        logger.warning(
            "oxr-mcp: open attempt {} failed: {} "
            "(form factor not yet available — waiting for streaming client)",
            self._open_attempts, err,
        )

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
                    logger.exception("oxr-mcp: error closing DeviceIOSession")
                self._dev_session = None
            if self._oxr_session is not None:
                try:
                    self._oxr_session.__exit__(None, None, None)
                except Exception:
                    logger.exception("oxr-mcp: error closing OpenXRSession")
                self._oxr_session = None

    def get_pose(self) -> dict:
        """Pose snapshot with derived spatial vectors — no raw quaternions.

        Returns ``is_valid=False`` (with optional ``error``) when the session
        can't open yet so callers can distinguish "not ready" from failure.
        """
        with self._lock:
            if self._dev_session is None:
                err = self._try_open()
                if err is not None:
                    return {
                        "is_valid":  False,
                        "position":  {"x": 0.0, "y": 1.6, "z": 0.0},
                        "forward":   {"x": 0.0, "y": 0.0, "z": -1.0},
                        "right":     {"x": 1.0, "y": 0.0, "z": 0.0},
                        "up":        {"x": 0.0, "y": 1.0, "z": 0.0},
                        "yaw_deg":   0.0,
                        "pitch_deg": 0.0,
                        "ts":        int(time.time() * 1000),
                        "error":     f"session_not_ready: {err}",
                    }
            self._dev_session.update()
            tracked = self._tracker.get_head(self._dev_session)
            data = tracked.data
            pose = data.pose
            px = float(pose.position.x)
            py = float(pose.position.y)
            pz = float(pose.position.z)
            qx = float(pose.orientation.x)
            qy = float(pose.orientation.y)
            qz = float(pose.orientation.z)
            qw = float(pose.orientation.w)

        fwd = _rotate_vec((qx, qy, qz, qw), (0.0,  0.0, -1.0))
        rgt = _rotate_vec((qx, qy, qz, qw), (1.0,  0.0,  0.0))
        up  = _rotate_vec((qx, qy, qz, qw), (0.0,  1.0,  0.0))
        yaw   = math.degrees(math.atan2(
            2.0 * (qw * qy + qx * qz),
            1.0 - 2.0 * (qy * qy + qz * qz)))
        pitch = math.degrees(math.asin(
            max(-1.0, min(1.0, 2.0 * (qw * qx - qy * qz)))))

        return {
            "is_valid":  bool(data.is_valid),
            "position":  {"x": round(px, 3), "y": round(py, 3), "z": round(pz, 3)},
            "forward":   {"x": round(fwd[0], 3), "y": round(fwd[1], 3), "z": round(fwd[2], 3)},
            "right":     {"x": round(rgt[0], 3), "y": round(rgt[1], 3), "z": round(rgt[2], 3)},
            "up":        {"x": round(up[0],  3), "y": round(up[1],  3), "z": round(up[2],  3)},
            "yaw_deg":   round(yaw,   1),
            "pitch_deg": round(pitch, 1),
            "ts":        int(time.time() * 1000),
        }


# ── MCP tool surface ──────────────────────────────────────────────────────────

def build_mcp(source: PoseSource) -> FastMCP:
    mcp = FastMCP("oxr-mcp")

    @mcp.tool()
    async def get_head_pose() -> dict:
        """Return the user's head position and orientation as human-readable vectors.

        Fields (all world-space, OpenXR Y-up, +x right, +y up, -z forward):
          is_valid  — False until tracking is established; retry, don't fail hard
          position  — {x, y, z} head position in metres
          forward   — {x, y, z} unit vector in the direction the user is looking
          right     — {x, y, z} unit vector pointing to the user's right
          up        — {x, y, z} unit vector pointing up from the user's head
          yaw_deg   — horizontal rotation in degrees (0 = facing -z, 90 = facing +x)
          pitch_deg — vertical tilt in degrees (positive = looking up)
          ts        — ms since Unix epoch

        No raw quaternions — use forward/right/up for spatial reasoning.
        """
        return await asyncio.get_running_loop().run_in_executor(None, source.get_pose)

    @mcp.tool()
    async def position_ahead(distance: float = 1.5) -> dict:
        """Compute the world position *distance* metres in front of the user.

        Use for: "in front of me", "where I'm looking", "ahead of me".

        Returns {x, y, z} world-space position, or {error: "pose unavailable"} if
        tracking is not yet established — in that case do not use any position
        values; retry after a short delay.
        """
        pose = await asyncio.get_running_loop().run_in_executor(None, source.get_pose)
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        p = pose["position"]
        f = pose["forward"]
        return {
            "x": round(p["x"] + f["x"] * distance, 3),
            "y": round(p["y"] + f["y"] * distance, 3),
            "z": round(p["z"] + f["z"] * distance, 3),
        }

    @mcp.tool()
    async def position_relative(
        forward: float = 0.0,
        right:   float = 0.0,
        up:      float = 0.0,
    ) -> dict:
        """Compute a world position from head-relative offsets (metres).

        forward > 0 = in front of user   (use for "ahead", "in front of me")
        forward < 0 = behind user
        right   > 0 = to the user's right
        right   < 0 = to the user's left
        up      > 0 = above eye level
        up      < 0 = below eye level

        Examples:
          "1m to my right"   → right=1.0
          "2m ahead and left"→ forward=2.0, right=-0.5
          "at arm's length"  → forward=0.7

        Returns {x, y, z} world-space position, or {error: "pose unavailable"} if
        tracking is not yet established — do not use any position values in
        that case; retry after a short delay.
        """
        pose = await asyncio.get_running_loop().run_in_executor(None, source.get_pose)
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        p = pose["position"]
        f = pose["forward"]
        r = pose["right"]
        u = pose["up"]
        return {
            "x": round(p["x"] + f["x"]*forward + r["x"]*right + u["x"]*up, 3),
            "y": round(p["y"] + f["y"]*forward + r["y"]*right + u["y"]*up, 3),
            "z": round(p["z"] + f["z"]*forward + r["z"]*right + u["z"]*up, 3),
        }

    @mcp.tool()
    async def get_health() -> dict:
        """Server status. ``session_open`` is True once the headless OpenXR
        session has been established."""
        return source.health_snapshot()

    return mcp


# ── Entry point ───────────────────────────────────────────────────────────────

async def _serve(cfg: Config, ready_file: Path | None = None) -> None:
    if cfg.cloudxr_env_file:
        if cfg.cloudxr_env_file.exists():
            load_cloudxr_env(cfg.cloudxr_env_file)
            logger.info("oxr-mcp: cloudxr env loaded from {}", cfg.cloudxr_env_file)
        else:
            logger.error(
                "oxr-mcp: cloudxr env file not found: {} — pose will be unavailable",
                cfg.cloudxr_env_file,
            )
    else:
        logger.warning(
            "oxr-mcp: no cloudxr_env_file configured — using whatever XR_RUNTIME_JSON is set in env",
        )

    source = PoseSource()
    logger.info("oxr-mcp: ready to serve get_head_pose (session opens lazily on first request)")
    try:
        app = build_mcp(source).http_app(path="/mcp")
        uv_cfg = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
        server = uvicorn.Server(uv_cfg)
        logger.info("oxr-mcp  mcp=/mcp  port={}", cfg.port)
        if ready_file:
            ready_file.touch()
        await server.serve()
    finally:
        await asyncio.get_running_loop().run_in_executor(None, source.close)
        logger.info("oxr-mcp: stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    ns, _ = p.parse_known_args()

    setup_logging("oxr-mcp")
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    yaml_path = ns.config or _DEFAULT_YAML
    if not yaml_path.exists():
        sys.exit(f"oxr-mcp: config file not found: {yaml_path}")
    raw = _load_raw(yaml_path)
    cfg = _build_config(yaml_path, raw)

    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
