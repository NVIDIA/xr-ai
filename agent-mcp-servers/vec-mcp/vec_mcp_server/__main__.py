# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vec-mcp: pure-math spatial primitives for agentic XR scenes.

Handles arithmetic the LLM is bad at (vector scaling, midpoints, axis offsets).

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  between_anchors(a_x, a_y, a_z, b_x, b_y, b_z) → {x, y, z}
      Component-wise midpoint of two world positions. Use when the
      utterance says "between A and B" / "halfway between" / "in the
      middle of".

  world_offset(origin, dx, dy, dz) → {x, y, z}
      origin shifted by axis-aligned deltas (world Y-up). Use for
      object-relative motion: "30 cm above sphere" = world_offset(sphere, dy=0.3).

  along_direction(origin, target, distance) → {x, y, z}
      origin moved `distance` metres along the line toward `target`.
      Use for "closer to / further from <obj>": pass A's coords as
      origin and B's coords as target.

  scale_value(current, factor) → {value}
      Scalar multiplication for sizes e.g., "3× bigger", "half".
      Returned as a dict so it composes uniformly with the vec results.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_nat.functions.spatial_math import Vector3
from xr_ai_nat.functions.spatial_math import _math as spatial_math

_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "vec_mcp_server.yaml"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    host: str
    port: int


def _load_raw(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_config(raw: dict) -> Config:
    return Config(
        host = raw.get("host", "0.0.0.0"),
        port = int(raw.get("port", 8250)),
    )


# ── Tool surface ──────────────────────────────────────────────────────────────


def build_mcp() -> FastMCP:
    mcp = FastMCP("vec-mcp")

    @mcp.tool()
    async def between_anchors(
        a_x: float, a_y: float, a_z: float,
        b_x: float, b_y: float, b_z: float,
    ) -> dict:
        """The point halfway between anchor A and anchor B.

        Use whenever the utterance contains "between"/"halfway"/"in the
        middle of"; the ONLY two positions that enter the math are A
        and B, even if other objects are visible in the scene.

        a_x/a_y/a_z = first anchor world position.
        b_x/b_y/b_z = second anchor world position.

        Returns {x, y, z} — the midpoint. Feed straight into
        add_primitive or update_primitive.
        """
        return spatial_math.compute_midpoint(
            Vector3(x=a_x, y=a_y, z=a_z),
            Vector3(x=b_x, y=b_y, z=b_z),
        )

    @mcp.tool()
    async def world_offset(
        origin_x: float, origin_y: float, origin_z: float,
        dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
    ) -> dict:
        """origin + (dx, dy, dz) along world axes (Y-up).

        Use for: "above sphere", "below cube", "0.5 m to the right of obj"
        (world right = +x; not user-relative — for that, use
        oxr-mcp.position_relative with origin_* args).

        Returns {x, y, z}.

        Example: sphere at (0, 1.5, -1.5), "30 cm above" →
            world_offset(0, 1.5, -1.5, dy=0.3) → (0, 1.8, -1.5).
        """
        return spatial_math.world_offset(
            Vector3(x=origin_x, y=origin_y, z=origin_z),
            x_meters=dx,
            y_meters=dy,
            z_meters=dz,
        )

    @mcp.tool()
    async def along_direction(
        origin_x: float, origin_y: float, origin_z: float,
        target_x: float, target_y: float, target_z: float,
        distance: float = 0.5,
    ) -> dict:
        """origin moved `distance` metres along the line toward target.

        Use for "closer to / further from <obj>": pass A's coords as
        origin and B's coords as target. Positive distance moves toward
        the target; negative moves away.

        Returns {x, y, z}, or {error} if origin and target coincide.
        """
        try:
            return spatial_math.compute_position_toward_or_away_from_reference(
                Vector3(x=origin_x, y=origin_y, z=origin_z),
                Vector3(x=target_x, y=target_y, z=target_z),
                movement_direction="toward" if distance >= 0 else "away",
                distance_meters=abs(distance),
            )
        except ValueError:
            return {"error": "origin and target coincide"}

    @mcp.tool()
    async def scale_value(current: float, factor: float) -> dict:
        """current × factor. Use for scaling sizes ("3× bigger" → factor=3).

        Returns {value} (rounded to 3 decimals).
        """
        return {"value": round(current * factor, 3)}

    return mcp


# ── Entry point ───────────────────────────────────────────────────────────────

async def _serve(cfg: Config, ready_file: Path | None = None) -> None:
    app = build_mcp().http_app(path="/mcp")
    uv_cfg = uvicorn.Config(app, host=cfg.host, port=cfg.port,
                            log_level="warning", log_config=None)
    server = uvicorn.Server(uv_cfg)
    logger.info("vec-mcp  mcp=/mcp  port={}", cfg.port)
    if ready_file:
        ready_file.touch()
    await server.serve()


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    ns, _ = p.parse_known_args()

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    setup_logging("vec-mcp")

    yaml_path = ns.config or _DEFAULT_YAML
    raw = _load_raw(yaml_path) if yaml_path.exists() else {}
    cfg = _build_config(raw)

    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
