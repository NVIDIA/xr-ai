# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastMCP client for ``space-mcp`` — the topological place-memory
backend wired into slam-example on the ``feat/spatial-memory`` branch.

Unlike the metric SLAM backends on the other branches (pose-mcp /
kimera-mcp / droid-mcp), space-mcp's tool surface is region-centric:
``process_frame`` returns the inferred region id and state rather than
a 6-DoF pose, and there is no IMU / intrinsics path (DINOv2 doesn't
care about either).  We keep a minimal client here — just the tools
the worker drives.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger


class SlamClient:
    """Long-lived MCP connection to space-mcp.  If it drops the next
    call re-opens.  Construct with ``url=None`` to disable the SLAM
    path entirely (the worker treats that as a feature flag)."""

    def __init__(self, url: str) -> None:
        self._url    = url.rstrip("/")
        self._client: Any = None
        self._lock   = asyncio.Lock()

    @property
    def health_url(self) -> str:
        return self._url

    async def _ensure_open(self) -> None:
        if self._client is not None:
            return
        from fastmcp import Client
        self._client = Client(self._url)
        await self._client.__aenter__()

    async def _aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._client = None

    async def _call(self, tool: str, args: dict) -> dict:
        async with self._lock:
            try:
                await self._ensure_open()
                result = await self._client.call_tool(tool, args)
            except Exception:
                await self._aclose()
                raise
        if hasattr(result, "data") and result.data is not None:
            return result.data
        try:
            return json.loads(result.content[0].text)
        except Exception:
            return {"error": "unparseable SLAM MCP response"}

    # ── tool surface ────────────────────────────────────────────────────

    async def process_frame(self, image_path: str, timestamp_us: int = 0) -> dict:
        return await self._call("process_frame", {
            "image_path": image_path, "timestamp_us": int(timestamp_us),
        })

    async def list_regions(self) -> dict:
        return await self._call("list_regions", {})

    async def reset_map(self) -> dict:
        return await self._call("reset_map", {})

    async def close(self) -> None:
        async with self._lock:
            await self._aclose()


async def wait_for_slam_mcp(client: "SlamClient | None") -> None:
    """Poll the SLAM MCP server until ``list_regions`` answers.
    space-mcp has no ``get_map_stats`` — ``list_regions`` is the
    cheapest no-arg tool and serves the same purpose."""
    if client is None:
        return
    while True:
        try:
            await client._ensure_open()
            await client._client.call_tool("list_regions", {})
            logger.info("SLAM-MCP ready ({})", client.health_url)
            return
        except Exception as exc:
            logger.info("still waiting for SLAM-MCP: {}", exc.__class__.__name__)
            await client._aclose()
            await asyncio.sleep(5.0)
