# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tools backed by the typed video-memory service."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .image import TimedImage
from .rpc import RPCClient
from .tools import Tool
from .types import EmptyRequest, StrictRequest


class ListRecordedParticipantsResult(BaseModel):
    participants: list[str]


class VideoStatsRequest(StrictRequest):
    participant_id: str = Field(min_length=1)


class VideoStatsResult(BaseModel):
    participant_id: str
    num_chunks: int
    total_bytes: int
    avg_chunk_bytes: int
    earliest_us: int
    latest_us: int


class _ParticipantRequest(StrictRequest):
    participant_id: str = Field(min_length=1)


class _DurationRequest(_ParticipantRequest):
    duration_seconds: int = Field(
        gt=0,
        le=300,
        description="Whole seconds of recorded history in the requested window.",
    )


class LatestVideoRequest(_DurationRequest):
    """Select the newest recorded video window."""


class HistoricalVideoRequest(_DurationRequest):
    """Select a recorded video window beginning at an absolute timestamp."""

    start_us: int = Field(
        gt=0,
        description="Unix-epoch timestamp in microseconds at the start of the window.",
    )


class RecordedVideoResult(BaseModel):
    path: str
    size: int
    start_us: int
    end_us: int


class _FrameSamplingRequest(_DurationRequest):
    frame_budget: int = Field(
        gt=0,
        le=256,
        description="Maximum total number of evenly distributed frames to return.",
    )
    max_width: int | None = Field(
        default=None,
        gt=0,
        description="Optional maximum exported width; requires max_height.",
    )
    max_height: int | None = Field(
        default=None,
        gt=0,
        description="Optional maximum exported height; requires max_width.",
    )

    @model_validator(mode="after")
    def validate_resolution(self) -> _FrameSamplingRequest:
        if (self.max_width is None) != (self.max_height is None):
            raise ValueError("max_width and max_height must be provided together")
        return self


class LatestFramesRequest(_FrameSamplingRequest):
    """Sample the newest recorded frame window."""


class HistoricalFramesRequest(_FrameSamplingRequest):
    """Sample a recorded frame window beginning at an absolute timestamp."""

    start_us: int = Field(
        gt=0,
        description="Unix-epoch timestamp in microseconds at the start of the window.",
    )


class SampledVideoFrame(TimedImage):
    timestamp_us: int = Field(
        ge=0,
        description="Estimated Unix-epoch timestamp interpolated from recording chunk metadata.",
    )
    path: str
    width: int
    height: int

    @model_validator(mode="before")
    @classmethod
    def add_image_reference(cls, value: object) -> object:
        if isinstance(value, dict) and "image" not in value and isinstance(value.get("path"), str):
            return {**value, "image": {"uri": value["path"]}}
        return value


class SampleFramesResult(BaseModel):
    frames: list[SampledVideoFrame]
    start_us: int
    end_us: int
    duration_seconds: int
    frame_budget: int
    max_width: int | None
    max_height: int | None


class HistoricalFrameRequest(_ParticipantRequest):
    start_us: int = Field(
        gt=0,
        description="Unix-epoch timestamp in microseconds of the requested frame.",
    )


class HistoricalFrameResult(TimedImage):
    timestamp_us: int = Field(
        ge=0,
        description="Estimated Unix-epoch timestamp interpolated from recording chunk metadata.",
    )
    path: str
    width: int
    height: int

    @model_validator(mode="before")
    @classmethod
    def add_image_reference(cls, value: object) -> object:
        if isinstance(value, dict) and "image" not in value and isinstance(value.get("path"), str):
            return {**value, "image": {"uri": value["path"]}}
        return value


class VideoHealthResult(BaseModel):
    ready: bool = True
    recording_enabled: bool


class VideoMemoryTools:
    """Own a video-memory service client and its finite tools."""

    def __init__(self, endpoint: str, *, timeout_s: float = 30.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)
        self.list_recorded_participants = Tool(
            "list_recorded_participants",
            "List exact participant identities with persisted camera history.",
            EmptyRequest,
            ListRecordedParticipantsResult,
            self._list_recorded_participants,
        )
        self.get_video_stats = Tool(
            "get_video_stats",
            "Return a participant's recorded Unix-epoch microsecond range and storage statistics.",
            VideoStatsRequest,
            VideoStatsResult,
            self._get_video_stats,
        )
        self.get_latest_video = Tool(
            "get_latest_video",
            "Return the newest recorded H.264 window for a participant and duration.",
            LatestVideoRequest,
            RecordedVideoResult,
            self._get_latest_video,
        )
        self.get_latest_frames = Tool(
            "get_latest_frames",
            "Return bounded timestamped frames from the newest recorded window.",
            LatestFramesRequest,
            SampleFramesResult,
            self._get_latest_frames,
        )
        self.get_historical_frame = Tool(
            "get_historical_frame",
            "Return the recorded PNG frame nearest an absolute Unix-epoch microsecond timestamp.",
            HistoricalFrameRequest,
            HistoricalFrameResult,
            self._get_historical_frame,
        )
        self.get_historical_video = Tool(
            "get_historical_video",
            "Return recorded H.264 beginning at start_us for the requested duration.",
            HistoricalVideoRequest,
            RecordedVideoResult,
            self._get_historical_video,
        )
        self.get_historical_frames = Tool(
            "get_historical_frames",
            "Return bounded timestamped frames beginning at start_us for the requested duration.",
            HistoricalFramesRequest,
            SampleFramesResult,
            self._get_historical_frames,
        )
        self.latest_tools = (
            self.get_latest_video,
            self.get_latest_frames,
        )
        self.historical_tools = (
            self.get_historical_frame,
            self.get_historical_frames,
            self.get_historical_video,
        )
        self.tools = (
            self.list_recorded_participants,
            self.get_video_stats,
            *self.latest_tools,
            *self.historical_tools,
        )

    async def _list_recorded_participants(
        self,
        request: EmptyRequest,
    ) -> ListRecordedParticipantsResult:
        return ListRecordedParticipantsResult.model_validate(
            await self._rpc.call("list_recorded_participants", request.model_dump())
        )

    async def _get_video_stats(self, request: VideoStatsRequest) -> VideoStatsResult:
        return VideoStatsResult.model_validate(
            await self._rpc.call("get_video_stats", request.model_dump())
        )

    async def _get_latest_video(
        self,
        request: LatestVideoRequest,
    ) -> RecordedVideoResult:
        return RecordedVideoResult.model_validate(
            await self._rpc.call("get_latest_video", request.model_dump())
        )

    async def _get_latest_frames(
        self,
        request: LatestFramesRequest,
    ) -> SampleFramesResult:
        return SampleFramesResult.model_validate(
            await self._rpc.call("get_latest_frames", request.model_dump())
        )

    async def _get_historical_frame(
        self,
        request: HistoricalFrameRequest,
    ) -> HistoricalFrameResult:
        return HistoricalFrameResult.model_validate(
            await self._rpc.call("get_historical_frame", request.model_dump())
        )

    async def _get_historical_video(
        self,
        request: HistoricalVideoRequest,
    ) -> RecordedVideoResult:
        return RecordedVideoResult.model_validate(
            await self._rpc.call("get_historical_video", request.model_dump())
        )

    async def _get_historical_frames(
        self,
        request: HistoricalFramesRequest,
    ) -> SampleFramesResult:
        return SampleFramesResult.model_validate(
            await self._rpc.call("get_historical_frames", request.model_dump())
        )

    async def get_health(self) -> VideoHealthResult:
        return VideoHealthResult.model_validate(
            await self._rpc.call("get_health", {}, timeout_s=2.0)
        )

    async def health(self) -> bool:
        try:
            return (await self.get_health()).ready
        except Exception:
            return False

    async def close(self) -> None:
        await self._rpc.close()


__all__ = [
    "HistoricalFrameRequest",
    "HistoricalFrameResult",
    "HistoricalFramesRequest",
    "HistoricalVideoRequest",
    "LatestFramesRequest",
    "LatestVideoRequest",
    "ListRecordedParticipantsResult",
    "RecordedVideoResult",
    "SampleFramesResult",
    "SampledVideoFrame",
    "VideoHealthResult",
    "VideoMemoryTools",
    "VideoStatsRequest",
    "VideoStatsResult",
]
