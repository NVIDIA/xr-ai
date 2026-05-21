# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin async wrappers around `NatRuntime.call_tool` for the MCP servers the
worker talks to (vlm-mcp, video-mcp), plus a helper that resolves the latest
camera frame path for a participant.
"""
from __future__ import annotations

import logging

from nat_runtime import NatRuntime

log = logging.getLogger("glasses_agent_nat.mcp_shims")


async def call_vlm(
    nat_runtime: NatRuntime,
    tool: str,
    args: dict,
    *,
    silent: bool = False,
) -> dict | str | None:
    try:
        return await nat_runtime.call_tool("vlm_mcp", tool, args)
    except Exception as exc:
        if not silent:
            log.error("vlm-mcp %s failed: %s", tool, exc)
        return {"error": str(exc)}


async def call_video(
    nat_runtime: NatRuntime,
    tool: str,
    args: dict,
    *,
    silent: bool = False,
) -> dict | str | None:
    try:
        return await nat_runtime.call_tool("video_mcp", tool, args)
    except Exception as exc:
        if not silent:
            log.error("video-mcp %s failed: %s", tool, exc)
        return {"error": str(exc)}


async def get_latest_frame_path(
    nat_runtime: NatRuntime,
    pid: str,
    ref_us: int,
) -> str | None:
    """Resolve the latest frame path from video-mcp for *pid*.

    Tries `get_latest_frame` (live-only / recording-disabled mode) first,
    then falls back to `get_frame_from_time` (recording-enabled mode).
    """
    candidates = [
        ("get_latest_frame", {"participant_id": pid}),
        (
            "get_frame_from_time",
            {"participant_id": pid, "second_ago": 0, "reference_time_us": ref_us or 0},
        ),
    ]
    for tool, args in candidates:
        try:
            data = await call_video(nat_runtime, tool, args, silent=True)
            if isinstance(data, dict) and "path" in data:
                return data["path"]
        except Exception as exc:
            log.debug("pre-fetch frame via %s failed: %s", tool, exc)
    return None
