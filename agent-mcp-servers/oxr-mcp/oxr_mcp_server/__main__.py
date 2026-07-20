# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility MCP adapter over native XR-tracking and spatial math."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger

from xr_ai_logging import setup_logging
from xr_ai_nat.functions.spatial_math import SpatialFrame, Vector3
from xr_ai_nat.functions.spatial_math import _math as spatial_math
from xr_ai_nat.functions.xr_tracking._client import OpenXRClient

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "oxr_mcp_server.yaml"


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    service_endpoint: str


def _build_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) or {}
    return Config(
        host=str(raw.get("host", "0.0.0.0")),
        port=int(raw.get("port", 8230)),
        service_endpoint=str(raw.get("service_endpoint", "tcp://127.0.0.1:8330")),
    )


class PoseSource:
    """Preserve the legacy MCP pose shape over the typed service client."""

    def __init__(self, endpoint: str) -> None:
        self._client = OpenXRClient(endpoint)

    async def get_pose(self) -> dict:
        value = (await self._client.get_head_pose()).model_dump(mode="python")
        value["ts"] = value.pop("timestamp_ms")
        return value

    async def health_snapshot(self) -> dict:
        return (await self._client.get_health()).model_dump(mode="python")

    async def close(self) -> None:
        await self._client.close()


# ── MCP tool surface ──────────────────────────────────────────────────────────

def _spatial_frame(pose: dict) -> SpatialFrame:
    return SpatialFrame(
        origin=Vector3.model_validate(pose["position"]),
        forward=Vector3.model_validate(pose["forward"]),
        right=Vector3.model_validate(pose["right"]),
        up=Vector3.model_validate(pose["up"]),
    )

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
        return await source.get_pose()

    @mcp.tool()
    async def position_ahead(distance: float = 1.5) -> dict:
        """Compute the world position *distance* metres in front of the user.

        Use for: "in front of me", "where I'm looking", "ahead of me".

        Returns {x, y, z} world-space position, or {error: "pose unavailable"} if
        tracking is not yet established — in that case do not use any position
        values; retry after a short delay.
        """
        pose = await source.get_pose()
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        return spatial_math.compute_gaze_target(_spatial_frame(pose), distance)

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
        pose = await source.get_pose()
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        p = pose["position"]
        origin = Vector3(
            x=p["x"] if origin_x is None else origin_x,
            y=p["y"] if origin_y is None else origin_y,
            z=p["z"] if origin_z is None else origin_z,
        )
        return spatial_math.offset_position_in_user_frame(
            _spatial_frame(pose),
            start_position=origin,
            forward_meters=forward,
            right_meters=right,
            up_meters=up,
        )

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
        pose = await source.get_pose()
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        return spatial_math.compute_user_relative_position(
            _spatial_frame(pose),
            direction_from_user=direction,
            distance_meters=distance,
        )

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
            pose = await source.get_pose()
            if not pose.get("is_valid"):
                return {"error": "pose unavailable"}
        else:
            pose = {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "forward": {"x": 0.0, "y": 0.0, "z": -1.0},
                "right": {"x": 1.0, "y": 0.0, "z": 0.0},
                "up": {"x": 0.0, "y": 1.0, "z": 0.0},
            }
        relations = {
            "front": "toward_user",
            "back": "away_from_user",
            "left": "left_of",
            "right": "right_of",
            "above": "above",
            "below": "below",
            "next_to": "right_of",
        }
        return spatial_math.compute_position_relative_to_anchor(
            _spatial_frame(pose),
            anchor_position=Vector3(x=origin_x, y=origin_y, z=origin_z),
            relation_to_anchor=relations[direction],
            distance_meters=distance,
        )

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
        pose = await source.get_pose()
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        return spatial_math.offset_position_in_user_frame(
            _spatial_frame(pose),
            start_position=Vector3(x=current_x, y=current_y, z=current_z),
            forward_meters=forward,
            right_meters=right,
            up_meters=up,
        )

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
            "x": round(container_x, 3),
            "y": round(container_y, 3),
            "z": round(container_z, 3),
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
        pose = await source.get_pose()
        if not pose.get("is_valid"):
            return {"error": "pose unavailable"}
        frame = _spatial_frame(pose)
        items = []
        for i in range(n):
            position = spatial_math.offset_position_in_user_frame(
                frame,
                start_position=Vector3(x=current_xs[i], y=current_ys[i], z=current_zs[i]),
                forward_meters=forward,
                right_meters=right,
                up_meters=up,
            )
            items.append({"obj_id": object_ids[i], **position})
        return {"items": items}

    @mcp.tool()
    async def get_health() -> dict:
        """Server status. ``session_open`` is True once the headless OpenXR
        session has been established."""
        return await source.health_snapshot()

    return mcp


# ── Entry point ───────────────────────────────────────────────────────────────

async def _serve(cfg: Config, ready_file: Path | None = None) -> None:
    source = PoseSource(cfg.service_endpoint)
    try:
        app = build_mcp(source).http_app(path="/mcp")
        uv_cfg = uvicorn.Config(
            app,
            host=cfg.host,
            port=cfg.port,
            log_level="warning",
            log_config=None,
        )
        server = uvicorn.Server(uv_cfg)
        logger.info("oxr-mcp mcp=/mcp port={} service={}", cfg.port, cfg.service_endpoint)
        if ready_file:
            ready_file.touch()
        await server.serve()
    finally:
        await source.close()
        logger.info("oxr-mcp: stopped")


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()

    setup_logging("oxr-mcp")
    config_path = args.config or _DEFAULT_YAML
    if not config_path.exists():
        sys.exit(f"oxr-mcp: config file not found: {config_path}")
    asyncio.run(_serve(_build_config(config_path), ready_file=args.ready_file))


if __name__ == "__main__":
    run()
