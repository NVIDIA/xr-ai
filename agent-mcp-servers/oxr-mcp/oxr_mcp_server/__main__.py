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

  position_relative(forward, right, up, origin_x=, origin_y=, origin_z=) → dict {x,y,z}
      Convert user-frame offsets (metres) to world-space position. Origin
      defaults to the user's head; pass origin_* to move an existing
      object user-relatively without doing the vector math yourself.

  place_user_relative(direction, distance) → dict {x,y,z}
      High-level: world position 'distance' metres in a named direction
      (front/back/left/right/above/below) from the user. No signs, no origin.

  place_object_relative(origin_x, origin_y, origin_z, direction, distance) → dict {x,y,z}
      High-level: world position 'distance' metres from an existing object
      in a named direction (front/back/left/right/above/below/next_to).
      No signs, no math.

  place_inside_by_id(movee_id, container_x, container_y, container_z) → dict
      Containment for "put X in Y". Returns {obj_id: movee_id, x, y, z}
      so the model can feed the result straight into update_primitive
      without picking which noun's coords to use.

  displace_object(current_x, current_y, current_z,
                  right=0.0, up=0.0, forward=0.0) → dict {x,y,z}
      User-frame displacement of an existing object. Add user-frame
      (right/up/forward) metres to the object's current world position.
      Handles multi-axis moves in one call ("up and to the left").

  displace_objects(object_ids, current_xs, current_ys, current_zs,
                   right=0.0, up=0.0, forward=0.0) → dict
      Batch displacement: same user-frame delta applied to N objects in
      one call. Returns {items: [{obj_id, x, y, z}, ...]} so the model
      can fan out to N update_primitive calls without re-deriving each
      new position.

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
from typing import Literal

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

        Fields (all world-space, +x right, +y up, -z forward):
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

    def _ground_basis(pose: dict) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ((fx, fz), (rx, rz)): pose.forward / pose.right projected onto
        the y=0 plane and renormalised."""
        f, r = pose["forward"], pose["right"]
        fx, fz = f["x"], f["z"]
        mag = math.sqrt(fx * fx + fz * fz)
        if mag < 1e-6:
            rx0, rz0 = r["x"], r["z"]
            mag2 = math.sqrt(rx0 * rx0 + rz0 * rz0)
            if mag2 < 1e-6:
                fx, fz = 0.0, -1.0
            else:
                rx0, rz0 = rx0 / mag2, rz0 / mag2
                fx, fz = rz0, -rx0
        else:
            fx, fz = fx / mag, fz / mag
        return (fx, fz), (-fz, fx)

    @mcp.tool()
    async def position_relative(
        forward: float = 0.0,
        right:   float = 0.0,
        up:      float = 0.0,
        origin_x: float | None = None,
        origin_y: float | None = None,
        origin_z: float | None = None,
    ) -> dict:
        """Compute a world position from user-frame offsets (metres).

        Direction conventions:
          forward → user's facing direction projected onto the GROUND
                    PLANE (yaw is honoured; pitch/roll are ignored, so a
                    head tilt does NOT make the result diagonal).
          right   → 90° clockwise from forward in the ground plane.
          up      → world +Y (gravity).

        Yawing the head DOES change "right" / "left" / "forward" — the
        result follows the direction the user's body is facing. Tilting
        the head (pitch / roll) does NOT — vertical moves stay vertical.

        For "in front of me along where I'm looking" (gaze-aware, includes
        pitch), use position_ahead instead.

        Origin defaults to the user's head position when omitted — use this
        to place a NEW object relative to the user. Pass origin_x/y/z to
        MOVE an existing object in a user-frame direction without doing the
        vector arithmetic yourself: pass the object's current position as
        origin and the desired offset.

        Examples:
          "1m to my right"
              → position_relative(right=1.0)
          "0.5m to my left and 0.3m above me"
              → position_relative(right=-0.5, up=0.3)
          Move object at (0, 1.7, -1.5) one metre to user's left:
              → position_relative(origin_x=0, origin_y=1.7, origin_z=-1.5,
                                  right=-1.0)

        Returns {x, y, z} world-space position, or {error: "pose unavailable"}
        if tracking is not yet established.
        """
        pose = await asyncio.get_running_loop().run_in_executor(None, source.get_pose)
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        p = pose["position"]
        ox = p["x"] if origin_x is None else origin_x
        oy = p["y"] if origin_y is None else origin_y
        oz = p["z"] if origin_z is None else origin_z
        (fx, fz), (rx, rz) = _ground_basis(pose)
        return {
            "x": round(ox + fx*forward + rx*right,        3),
            "y": round(oy + up,                            3),
            "z": round(oz + fz*forward + rz*right,        3),
        }

    @mcp.tool()
    async def place_user_relative(
        direction: Literal["front", "back", "left", "right", "above", "below"],
        distance: float = 1.5,
    ) -> dict:
        """Compute a world position *distance* metres in a named user-frame
        direction. Use this in preference to position_relative when the user
        names a single cardinal direction relative to themselves.

        The tool handles signs and origin internally — distance is ALWAYS a
        positive number, and the origin is always the user's head. You only
        pick the named direction.

        Direction semantics (all gravity-aligned — head pitch/roll do not
        bleed in; only yaw rotates the horizontal axes):
          front  → user's facing direction projected onto the ground plane
          back   → opposite of front
          right  → 90° clockwise from front (the user's actual right)
          left   → opposite of right
          above  → world +Y
          below  → world -Y

        Use for utterances like:
          "in front of me"  → place_user_relative("front", 1.5)
          "behind me"       → place_user_relative("back",  1.5)
          "to my left"      → place_user_relative("left",  1.0)
          "above me"        → place_user_relative("above", 1.0)

        Returns {x, y, z} world-space position, or {error: "pose unavailable"}
        if tracking is not yet established.
        """
        if distance < 0:
            return {"error": "distance must be non-negative; flip the direction instead"}
        pose = await asyncio.get_running_loop().run_in_executor(None, source.get_pose)
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        p = pose["position"]
        (fx, fz), (rx, rz) = _ground_basis(pose)
        dx = dy = dz = 0.0
        if direction == "front":
            dx, dz = fx * distance, fz * distance
        elif direction == "back":
            dx, dz = -fx * distance, -fz * distance
        elif direction == "right":
            dx, dz = rx * distance, rz * distance
        elif direction == "left":
            dx, dz = -rx * distance, -rz * distance
        elif direction == "above":
            dy = distance
        elif direction == "below":
            dy = -distance
        return {
            "x": round(p["x"] + dx, 3),
            "y": round(p["y"] + dy, 3),
            "z": round(p["z"] + dz, 3),
        }

    @mcp.tool()
    async def place_object_relative(
        origin_x: float,
        origin_y: float,
        origin_z: float,
        direction: Literal["front", "back", "left", "right", "above", "below", "next_to"],
        distance: float = 0.3,
    ) -> dict:
        """Compute a world position *distance* metres in a named direction
        from an object at (origin_x, origin_y, origin_z). Use this in
        preference to position_relative + world_offset when placing or moving
        relative to an existing scene object.

        Direction semantics (user-frame applied at the object's origin):
          front  → on the side of the object facing OPPOSITE the user's
                   gaze. Coincides with "toward the user" only when the
                   user is looking at the object; if the user is gazing
                   away from it, this points further along the gaze
                   direction (away from "between user and object"). For
                   a true toward-user vector, use vec-mcp.along_direction
                   with the user's head position as target.
          back   → on the side of the object further along the user's
                   gaze direction. Same caveat as `front` when the user
                   is not looking at the object.
          right  → user's right at the object's location (gaze-independent
                   in the horizontal plane).
          left   → user's left at the object's location.
          above  → world +Y from the object.
          below  → world -Y from the object.
          next_to → `distance` metres to the user's right of the object
                    (default 0.3 m when the user just says "next to obj").

        right / left / above / below are robust regardless of where the
        user is looking. front / back assume the user is looking at the
        object — true for "behind the cube" / "in front of the cube"
        utterances in practice. Distance is ALWAYS a positive number;
        pick the direction enum to flip sign.

        Use for utterances like:
          "Add a sphere behind the cube"
              → place_object_relative(cube.x, cube.y, cube.z, "back", 0.3)
          "Put a sphere on top of the cube"
              → place_object_relative(cube.x, cube.y, cube.z, "above", cube.size)
          "Put a sphere next to the cube"
              → place_object_relative(cube.x, cube.y, cube.z, "next_to")

        Returns {x, y, z} world-space position, or {error: "pose unavailable"}
        if tracking is not yet established (front/back/left/right need pose;
        above/below do not).
        """
        if distance < 0:
            return {"error": "distance must be non-negative; flip the direction instead"}
        if direction in ("front", "back", "left", "right", "next_to"):
            pose = await asyncio.get_running_loop().run_in_executor(None, source.get_pose)
            if not pose.get("is_valid"):
                return {"error": "pose unavailable"}
            (fx, fz), (rx, rz) = _ground_basis(pose)
        else:
            fx = fz = rx = rz = 0.0
        dx = dy = dz = 0.0
        if direction == "front":
            dx, dz = -fx * distance, -fz * distance
        elif direction == "back":
            dx, dz = fx * distance, fz * distance
        elif direction == "right":
            dx, dz = rx * distance, rz * distance
        elif direction == "left":
            dx, dz = -rx * distance, -rz * distance
        elif direction == "next_to":
            dx, dz = rx * distance, rz * distance
        elif direction == "above":
            dy = distance
        elif direction == "below":
            dy = -distance
        return {
            "x": round(origin_x + dx, 3),
            "y": round(origin_y + dy, 3),
            "z": round(origin_z + dz, 3),
        }

    @mcp.tool()
    async def displace_object(
        current_x: float,
        current_y: float,
        current_z: float,
        right:   float = 0.0,
        up:      float = 0.0,
        forward: float = 0.0,
    ) -> dict:
        """Shift an object by a user-frame delta — preferred tool for
        "move it N metres to my right / up / forward".

        Inputs:
          current_x/y/z  — the object's CURRENT world position (read from
                           the SCENE block; never (0,0,0) unless the object
                           really is at the world origin).
          right          — metres along the user's right axis (negative = left)
          up             — metres along world +Y (negative = down)
          forward        — metres along the user's facing direction projected
                           onto the ground plane (negative = backward)

        The user's frame is gravity-aligned: head pitch/roll do NOT bleed
        into horizontal moves (yaw is honoured). "Up" is always world +Y.

        Use this for ANY "move it <distance> <user-direction>" utterance,
        including multi-axis ones — pass non-zero values to multiple of
        right/up/forward in a single call:
          "move it 1 m to my right"          → right=1.0
          "shift it 30 cm down"              → up=-0.3
          "push it 0.5 m forward"            → forward=0.5
          "up and to my left"                → right=-0.5, up=0.5
          "down and back"                    → up=-0.5, forward=-0.5

        Returns {x, y, z} world-space position, or {error: "pose unavailable"}
        if tracking is not yet established.
        """
        pose = await asyncio.get_running_loop().run_in_executor(None, source.get_pose)
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        (fx, fz), (rx, rz) = _ground_basis(pose)
        return {
            "x": round(current_x + fx * forward + rx * right, 3),
            "y": round(current_y + up,                        3),
            "z": round(current_z + fz * forward + rz * right, 3),
        }

    @mcp.tool()
    async def place_inside_by_id(
        movee_id: str,
        container_x: float,
        container_y: float,
        container_z: float,
    ) -> dict:
        """Containment for "put X in Y" / "drop X inside Y" / "stick X into Y".

        Argument names are `movee_id` + `container_*` (not `origin_*`) so
        that "put X in Y" parses unambiguously: X is the movee, Y is the
        container.

        Returns {obj_id: movee_id, x, y, z} where (x, y, z) is the
        container's position. Feed the entire dict into update_primitive
        verbatim:
            update_primitive(obj_id=ret.obj_id, x=ret.x, y=ret.y, z=ret.z)
        """
        return {
            "obj_id": movee_id,
            "x":      round(container_x, 3),
            "y":      round(container_y, 3),
            "z":      round(container_z, 3),
        }

    @mcp.tool()
    async def displace_objects(
        object_ids:  list[str],
        current_xs:  list[float],
        current_ys:  list[float],
        current_zs:  list[float],
        right:   float = 0.0,
        up:      float = 0.0,
        forward: float = 0.0,
    ) -> dict:
        """Batch user-frame displacement: same delta applied to every
        object in parallel.

        Use for utterances referencing multiple objects ("them",
        "all of them", "everything", "the spheres"). Returns one item
        per input object in the same order.

        Parallel lists: object_ids[i] / current_xs[i] / current_ys[i] /
        current_zs[i] describe the i-th object. All four lists must be
        the same length. right/up/forward are signed metres in the
        user's frame (same semantics as displace_object).

        Returns {"items": [{"obj_id", "x", "y", "z"}, ...]}.
        """
        n = len(object_ids)
        if not (len(current_xs) == n and len(current_ys) == n and len(current_zs) == n):
            return {"error": "object_ids / current_xs / current_ys / current_zs "
                             "must all be the same length"}
        if n == 0:
            return {"items": []}
        pose = await asyncio.get_running_loop().run_in_executor(None, source.get_pose)
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        (fx, fz), (rx, rz) = _ground_basis(pose)
        items = []
        for i in range(n):
            cx, cy, cz = current_xs[i], current_ys[i], current_zs[i]
            items.append({
                "obj_id": object_ids[i],
                "x": round(cx + fx * forward + rx * right, 3),
                "y": round(cy + up,                         3),
                "z": round(cz + fz * forward + rz * right, 3),
            })
        return {"items": items}

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
        # log_config=None skips uvicorn's own dictConfig install so its
        # uvicorn / uvicorn.error / uvicorn.access loggers fall back to root
        # and get intercepted by the loguru bridge installed in setup_logging.
        # log_level="warning" still applies (set independently via setLevel
        # on each uvicorn logger), so only WARNING+ records reach loguru.
        uv_cfg = uvicorn.Config(app, host=cfg.host, port=cfg.port,
                                log_level="warning", log_config=None)
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
