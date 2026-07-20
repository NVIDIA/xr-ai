# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT functions backed by the long-running video-memory service."""

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import Field

from ._client import VideoMemoryClient
from .schemas import EmptyRequest


class VideoMemoryFunctionsConfig(FunctionGroupBaseConfig, name="xr_video_memory"):
    """Configure live-frame and recorded-video functions."""

    endpoint: str = Field(description="Private video-memory service endpoint.")
    timeout_s: float = Field(default=30.0, gt=0.0)


@register_function_group(config_type=VideoMemoryFunctionsConfig)
async def video_memory_functions(config: VideoMemoryFunctionsConfig, _builder: Builder):
    client = VideoMemoryClient(config.endpoint, timeout_s=config.timeout_s)
    group = FunctionGroup(config=config)

    async def list_live_participants(request: EmptyRequest) -> list[str]:
        del request
        return (await client.list_live_participants()).participants

    async def list_recorded_participants(request: EmptyRequest) -> list[str]:
        del request
        return (await client.list_recorded_participants()).participants

    group.add_function(
        "list_live_participants",
        list_live_participants,
        description="List participants whose live camera can currently provide a frame.",
    )
    group.add_function(
        "list_recorded_participants",
        list_recorded_participants,
        description="List participants with recorded camera history.",
    )
    group.add_function(
        "get_video_stats",
        client.get_video_stats,
        description="Return the recorded time range and storage statistics for one participant.",
    )
    group.add_function(
        "query_video",
        client.query_video,
        description="Write an H.264 clip for an absolute participant time window and return its path.",
    )
    group.add_function(
        "get_frame_from_time",
        client.get_frame_from_time,
        description=(
            "Return a PNG camera frame at or before an utterance timestamp. "
            "Use reference_time_us to anchor the lookup to the user's request."
        ),
    )

    try:
        yield group
    finally:
        await client.close()


__all__ = ["VideoMemoryFunctionsConfig"]
