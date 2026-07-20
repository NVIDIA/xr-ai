# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private typed client for the video-memory service."""

from .._rpc import RPCClient
from .schemas import (
    EmptyRequest,
    FrameAtTimeRequest,
    FrameResult,
    ParticipantsResult,
    QueryVideoRequest,
    QueryVideoResult,
    VideoMemoryHealth,
    VideoStatsRequest,
    VideoStatsResult,
)


class VideoMemoryClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 30.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)

    async def list_live_participants(self) -> ParticipantsResult:
        return ParticipantsResult.model_validate(
            await self._rpc.call("list_live_participants")
        )

    async def list_recorded_participants(self) -> ParticipantsResult:
        return ParticipantsResult.model_validate(
            await self._rpc.call("list_recorded_participants")
        )

    async def get_video_stats(self, request: VideoStatsRequest) -> VideoStatsResult:
        return VideoStatsResult.model_validate(
            await self._rpc.call("get_video_stats", request.model_dump())
        )

    async def query_video(self, request: QueryVideoRequest) -> QueryVideoResult:
        return QueryVideoResult.model_validate(
            await self._rpc.call("query_video", request.model_dump())
        )

    async def get_frame_from_time(self, request: FrameAtTimeRequest) -> FrameResult:
        return FrameResult.model_validate(
            await self._rpc.call("get_frame_from_time", request.model_dump())
        )

    async def get_health(self, request: EmptyRequest | None = None) -> VideoMemoryHealth:
        arguments = (request or EmptyRequest()).model_dump()
        return VideoMemoryHealth.model_validate(
            await self._rpc.call("get_health", arguments, timeout_s=2.0)
        )

    async def close(self) -> None:
        await self._rpc.close()
