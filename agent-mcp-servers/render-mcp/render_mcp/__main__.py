# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility MCP adapter over the XR render demo scene functions."""

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger
from xr_ai_logging import setup_logging
from xr_render_scene import (
    AddPrimitiveRequest,
    EmptyRequest,
    RemovePrimitiveRequest,
    SceneClient,
    UpdatePrimitiveRequest,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "render_mcp.yaml"


def build_mcp(client: SceneClient) -> FastMCP:
    """Preserve the legacy render tool names and argument schemas."""

    mcp = FastMCP("render-mcp")

    @mcp.tool()
    async def start_xr() -> dict:
        """Start LOVR if needed and return immediately while startup continues."""
        result = await client.start_xr(EmptyRequest())
        return result.model_dump(mode="python", exclude_none=True)

    @mcp.tool()
    async def add_primitive(
        prim_type: str,
        x: float = 0.0,
        y: float = 1.6,
        z: float = -1.5,
        r: float = 0.2,
        g: float = 0.9,
        b: float = 1.0,
        size: float = 0.1,
    ) -> dict:
        """Add a sphere or box at a world-space position measured in metres."""
        result = await client.add_primitive(
            AddPrimitiveRequest(
                prim_type=prim_type,
                x=x,
                y=y,
                z=z,
                r=r,
                g=g,
                b=b,
                size=size,
            )
        )
        return result.model_dump(mode="python", exclude_none=True)

    @mcp.tool()
    async def update_primitive(
        obj_id: str,
        prim_type: str | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        r: float | None = None,
        g: float | None = None,
        b: float | None = None,
        size: float | None = None,
    ) -> dict:
        """Partially update an existing primitive; omitted fields remain unchanged."""
        result = await client.update_primitive(
            UpdatePrimitiveRequest(
                obj_id=obj_id,
                prim_type=prim_type,
                x=x,
                y=y,
                z=z,
                r=r,
                g=g,
                b=b,
                size=size,
            )
        )
        return result.model_dump(mode="python", exclude_none=True)

    @mcp.tool()
    async def remove_primitive(obj_id: str) -> dict:
        """Remove a primitive from the scene by object ID."""
        result = await client.remove_primitive(RemovePrimitiveRequest(obj_id=obj_id))
        return result.model_dump(mode="python", exclude_none=True)

    @mcp.tool()
    async def get_scene_state() -> dict:
        """Return every current scene object and its properties."""
        result = await client.get_scene_state(EmptyRequest())
        return result.model_dump(mode="python")

    @mcp.tool()
    async def get_health() -> dict:
        """Return LOVR lifecycle and scene-delivery status."""
        result = await client.get_health(EmptyRequest())
        return result.model_dump(mode="python")

    return mcp


async def _serve(config: dict, ready_file: Path | None) -> None:
    endpoint = str(config.get("service_endpoint", "tcp://127.0.0.1:8320"))
    client = SceneClient(endpoint)
    try:
        await client.get_health(EmptyRequest())
        app = build_mcp(client).http_app(path="/mcp")
        port = int(config.get("port", 8220))
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=str(config.get("host", "0.0.0.0")),
                port=port,
                log_level="warning",
                log_config=None,
            )
        )
        logger.info("render-mcp mcp=/mcp port={} service={}", port, endpoint)
        if ready_file:
            ready_file.touch()
        await server.serve()
    finally:
        await client.close()


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()

    setup_logging("render-mcp")
    config_path = args.config or _DEFAULT_CONFIG
    if not config_path.exists():
        sys.exit(f"render-mcp: config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    asyncio.run(_serve(config, args.ready_file))


if __name__ == "__main__":
    run()
