# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastMCP client for any of the SLAM MCP backends in this repo:
``pose-mcp``, ``kimera-mcp``, ``droid-mcp``.  All three expose the same
tool surface (``estimate_pose``, ``push_imu``, ``set_camera_intrinsics``,
``get_map_stats``, ``reset_map``) so this client is backend-agnostic —
which one the worker actually talks to is a function of the URL.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger


class SlamClient:
    """Long-lived MCP connection to a SLAM backend.  If it drops the
    next call re-opens.  Construct with ``url=None`` to disable the
    SLAM path entirely (the worker treats that as a feature flag)."""

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

    async def estimate_pose(self, image_path: str, timestamp_us: int = 0) -> dict:
        return await self._call("estimate_pose", {
            "image_path": image_path, "timestamp_us": int(timestamp_us),
        })

    async def push_imu(self, samples: list[list[float]]) -> dict:
        """Forward a batch of ``[ts_ms, gx, gy, gz, ax, ay, az]`` rows."""
        return await self._call("push_imu", {"samples": samples})

    async def set_camera_intrinsics(
        self, *, width: int, height: int,
        fx: float, fy: float, cx: float, cy: float,
    ) -> dict:
        return await self._call("set_camera_intrinsics", {
            "width": int(width), "height": int(height),
            "fx": float(fx),     "fy": float(fy),
            "cx": float(cx),     "cy": float(cy),
        })

    async def get_map_stats(self) -> dict:
        return await self._call("get_map_stats", {})

    async def reset_map(self) -> dict:
        return await self._call("reset_map", {})

    async def close(self) -> None:
        async with self._lock:
            await self._aclose()


async def wait_for_slam_mcp(client: "SlamClient | None") -> None:
    """Poll the SLAM MCP server until ``get_map_stats`` answers."""
    if client is None:
        return
    while True:
        try:
            await client._ensure_open()
            await client._client.call_tool("get_map_stats", {})
            logger.info("SLAM-MCP ready ({})", client.health_url)
            return
        except Exception as exc:
            logger.info("still waiting for SLAM-MCP: {}", exc.__class__.__name__)
            await client._aclose()
            await asyncio.sleep(5.0)
