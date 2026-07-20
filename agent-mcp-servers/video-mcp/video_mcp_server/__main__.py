# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility MCP adapter over the native video-memory functions."""

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger

from xr_ai_logging import setup_logging
from xr_ai_nat.functions._rpc import RPCError
from xr_ai_nat.functions.video_memory._client import VideoMemoryClient
from xr_ai_nat.functions.video_memory.schemas import (
    FrameAtTimeRequest,
    QueryVideoRequest,
    VideoStatsRequest,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "video_mcp_server.yaml"


def _error(error: RPCError) -> dict:
    return {"error": str(error)}


def build_mcp(client: VideoMemoryClient, *, recording_enabled: bool) -> FastMCP:
    """Preserve the legacy tool set while delegating capability work."""

    mcp = FastMCP("video-mcp")

    @mcp.tool()
    async def list_live_participants() -> list[str]:
        """Return participants whose current live camera frame is available."""
        return (await client.list_live_participants()).participants

    @mcp.tool()
    async def get_frame_from_time(
        participant_id: str,
        second_ago: int = 0,
        reference_time_us: int = 0,
    ) -> dict:
        """Return a PNG frame relative to the wall clock or an explicit timestamp.

        An unanchored second_ago=0 request uses the live camera. Anchored or
        historical requests use recorded video and require recording to be enabled.
        """
        try:
            result = await client.get_frame_from_time(
                FrameAtTimeRequest(
                    participant_id=participant_id,
                    second_ago=second_ago,
                    reference_time_us=reference_time_us,
                )
            )
        except (RPCError, ValueError) as error:
            return {"error": str(error)}
        return result.model_dump(mode="python")

    if not recording_enabled:

        @mcp.tool()
        async def get_latest_frame(participant_id: str) -> dict:
            """Return the current live frame. Deprecated; use get_frame_from_time."""
            try:
                frame = await client.get_frame_from_time(
                    FrameAtTimeRequest(participant_id=participant_id)
                )
            except RPCError as error:
                return _error(error)
            result = frame.model_dump(mode="python")
            return {
                key: result[key]
                for key in ("path", "width", "height", "timestamp_us")
            }

        return mcp

    @mcp.tool()
    async def list_recorded_participants() -> list[str]:
        """Return participants with recorded video chunks."""
        return (await client.list_recorded_participants()).participants

    @mcp.tool()
    async def get_video_stats(participant_id: str) -> dict:
        """Return storage and time-range statistics for recorded video."""
        try:
            result = await client.get_video_stats(
                VideoStatsRequest(participant_id=participant_id)
            )
        except RPCError as error:
            return _error(error)
        return result.model_dump(mode="python")

    @mcp.tool()
    async def query_video(participant_id: str, start_us: int, end_us: int) -> dict:
        """Write an H.264 clip for an absolute time window and return its path."""
        try:
            result = await client.query_video(
                QueryVideoRequest(
                    participant_id=participant_id,
                    start_us=start_us,
                    end_us=end_us,
                )
            )
        except (RPCError, ValueError) as error:
            return {"error": str(error)}
        return result.model_dump(mode="python")

    return mcp


async def _serve(config: dict, ready_file: Path | None) -> None:
    client = VideoMemoryClient(
        str(config.get("service_endpoint", "tcp://127.0.0.1:8310"))
    )
    try:
        health = await client.get_health()
        app = build_mcp(
            client,
            recording_enabled=health.recording_enabled,
        ).http_app(path="/mcp")
        port = int(config.get("port", 8210))
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=str(config.get("host", "0.0.0.0")),
                port=port,
                log_level="warning",
                log_config=None,
            )
        )
        logger.info(
            "video-mcp mcp=/mcp port={} service={} recording_enabled={}",
            port,
            config.get("service_endpoint", "tcp://127.0.0.1:8310"),
            health.recording_enabled,
        )
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

    setup_logging("video-mcp")
    config_path = args.config or _DEFAULT_CONFIG
    if not config_path.exists():
        sys.exit(f"video-mcp: config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    asyncio.run(_serve(config, args.ready_file))


if __name__ == "__main__":
    run()
