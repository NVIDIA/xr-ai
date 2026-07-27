# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT recorded-video functions backed by the video-memory service."""

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import Field

from ._client import VideoHealthRequest, VideoMemoryClient


class VideoMemoryFunctionsConfig(FunctionGroupBaseConfig, name="xr_video_memory"):
    """Configure recorded-video query operations."""

    endpoint: str = Field(
        description="Typed video-memory service endpoint.",
    )
    timeout_s: float = Field(default=30.0, gt=0.0)


class VideoMemoryControlFunctionsConfig(FunctionGroupBaseConfig, name="xr_video_memory_control"):
    """Configure video-memory service readiness operations."""

    endpoint: str = Field(description="Typed video-memory service endpoint.")
    timeout_s: float = Field(default=30.0, gt=0.0)


@register_function_group(config_type=VideoMemoryFunctionsConfig)
async def video_memory_functions(config: VideoMemoryFunctionsConfig, _builder: Builder):
    """Expose recorded-video capabilities without leaking their transport."""

    client = VideoMemoryClient(config.endpoint, timeout_s=config.timeout_s)

    group = FunctionGroup(config=config)
    group.add_function(
        "list_recorded_participants",
        client.list_recorded_participants,
        description="List participants with recorded video.",
    )
    group.add_function(
        "get_video_stats",
        client.get_video_stats,
        description="Return the recorded video time range and storage statistics.",
    )
    group.add_function(
        "query_video",
        client.query_video,
        description="Return a recorded H.264 clip covering a participant time window.",
    )
    group.add_function(
        "get_frame_from_time",
        client.get_frame_from_time,
        description="Return a recorded camera frame relative to an utterance timestamp.",
    )
    try:
        yield group
    finally:
        await client.close()


@register_function_group(config_type=VideoMemoryControlFunctionsConfig)
async def video_memory_control_functions(config: VideoMemoryControlFunctionsConfig, _builder: Builder):
    """Expose video-memory service readiness outside query groups."""

    client = VideoMemoryClient(config.endpoint, timeout_s=config.timeout_s)

    # NAT 1.8 group functions require exactly one input parameter.
    async def is_available(request: VideoHealthRequest) -> bool:
        del request
        return await client.health()

    group = FunctionGroup(config=config)
    group.add_function(
        "get_health",
        client.get_health,
        description="Return recorded-video service readiness.",
    )
    group.add_function(
        "is_available",
        is_available,
        description="Return whether the video-memory service is accepting requests.",
    )

    try:
        yield group
    finally:
        await client.close()


__all__ = ["VideoMemoryFunctionsConfig", "VideoMemoryControlFunctionsConfig"]
