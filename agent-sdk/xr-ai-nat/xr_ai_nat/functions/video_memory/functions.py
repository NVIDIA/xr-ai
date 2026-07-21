# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded-video NAT functions backed by the video-memory service."""

from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import Field

from ._client import VideoMemoryClient
from .schemas import EmptyRequest, ParticipantsResult


class VideoMemoryFunctionsConfig(FunctionGroupBaseConfig, name="xr_video_memory"):
    """Configure recorded-video discovery, query, and frame extraction."""

    endpoint: str = Field(
        description="Private msgpack/ZMQ endpoint of video-memory-service."
    )
    timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Maximum time to wait for an H.264 query or frame extraction.",
    )


@register_function_group(config_type=VideoMemoryFunctionsConfig)
async def video_memory_functions(config: VideoMemoryFunctionsConfig, _builder: Builder):
    """Expose recorded history without coupling agents to storage or NVDEC."""

    client = VideoMemoryClient(config.endpoint, timeout_s=config.timeout_s)
    group = FunctionGroup(config=config)

    async def list_recorded_participants(request: EmptyRequest) -> ParticipantsResult:
        del request
        return await client.list_recorded_participants()

    group.add_function(
        "list_recorded_participants",
        list_recorded_participants,
        description=(
            "List exact participant identities with persisted camera history. "
            "Call this before requesting recorded statistics, clips, or frames."
        ),
    )
    group.add_function(
        "get_video_stats",
        client.get_video_stats,
        description=(
            "Return a participant's recorded Unix-epoch microsecond range and storage "
            "statistics. Use the range to validate absolute clip windows."
        ),
    )
    group.add_function(
        "query_video",
        client.query_video,
        description=(
            "Write an H.264 clip overlapping an absolute Unix-epoch microsecond window "
            "and return its local path."
        ),
    )
    group.add_function(
        "get_frame_from_time",
        client.get_frame_from_time,
        description=(
            "Extract the recorded PNG frame nearest reference_time_us minus second_ago "
            "whole seconds. This never accesses a current live camera frame."
        ),
    )

    try:
        yield group
    finally:
        await client.close()


__all__ = ["VideoMemoryFunctionsConfig"]
