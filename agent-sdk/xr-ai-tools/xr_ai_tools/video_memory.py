# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tools backed by the typed video-memory service."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .image import ImageReference, TimedImage
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


class RecordedVideoRequest(StrictRequest):
    participant_id: str = Field(min_length=1)
    start_us: int = Field(gt=0)
    end_us: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_window(self) -> RecordedVideoRequest:
        if self.end_us <= self.start_us:
            raise ValueError("end_us must be greater than start_us")
        return self


class RecordedVideoResult(BaseModel):
    path: str
    size: int
    start_us: int
    end_us: int


class SampleFramesRequest(StrictRequest):
    participant_id: str = Field(
        min_length=1,
        description="Exact participant identity whose recorded video should be sampled.",
    )
    reference_time_us: int = Field(
        gt=0,
        description="Unix-epoch timestamp in microseconds at the end of the requested window.",
    )
    duration_seconds: int = Field(
        gt=0,
        le=300,
        description="Whole seconds of recorded history ending at reference_time_us.",
    )
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
    def validate_window(self) -> SampleFramesRequest:
        start_us = self.reference_time_us - self.duration_seconds * 1_000_000
        if start_us <= 0:
            raise ValueError("duration_seconds extends before the Unix epoch")
        if (self.max_width is None) != (self.max_height is None):
            raise ValueError("max_width and max_height must be provided together")
        return self


class SampledVideoFrame(TimedImage):
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


class HistoricalFrameRequest(StrictRequest):
    participant_id: str = Field(
        min_length=1,
        description="Exact participant identity whose recorded frame should be extracted.",
    )
    second_ago: int = Field(
        default=0,
        ge=0,
        description="Whole seconds before reference_time_us.",
    )
    reference_time_us: int = Field(
        gt=0,
        description="Unix-epoch timestamp in microseconds used as the lookup reference.",
    )


class HistoricalFrameResult(BaseModel):
    path: str
    image: ImageReference
    width: int
    height: int
    timestamp_us: int
    second_ago: int
    actual_second_ago: float

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
        self.get_recorded_video = Tool(
            "get_recorded_video",
            "Return an H.264 recording overlapping an absolute Unix-epoch microsecond window.",
            RecordedVideoRequest,
            RecordedVideoResult,
            self._get_recorded_video,
        )
        self.sample_recorded_frames = Tool(
            "sample_recorded_frames",
            "Return bounded timestamped image frames from recorded history ending at reference_time_us.",
            SampleFramesRequest,
            SampleFramesResult,
            self._sample_recorded_frames,
        )
        self.get_frame_from_time = Tool(
            "get_frame_from_time",
            "Extract the recorded PNG frame nearest reference_time_us minus second_ago whole seconds.",
            HistoricalFrameRequest,
            HistoricalFrameResult,
            self._get_frame_from_time,
        )
        self.tools = (
            self.list_recorded_participants,
            self.get_video_stats,
            self.get_recorded_video,
            self.sample_recorded_frames,
            self.get_frame_from_time,
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

    async def _get_recorded_video(
        self,
        request: RecordedVideoRequest,
    ) -> RecordedVideoResult:
        return RecordedVideoResult.model_validate(
            await self._rpc.call("get_recorded_video", request.model_dump())
        )

    async def _sample_recorded_frames(
        self,
        request: SampleFramesRequest,
    ) -> SampleFramesResult:
        return SampleFramesResult.model_validate(
            await self._rpc.call("sample_recorded_frames", request.model_dump())
        )

    async def _get_frame_from_time(
        self,
        request: HistoricalFrameRequest,
    ) -> HistoricalFrameResult:
        return HistoricalFrameResult.model_validate(
            await self._rpc.call("get_frame_from_time", request.model_dump())
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
    "ListRecordedParticipantsResult",
    "RecordedVideoRequest",
    "RecordedVideoResult",
    "SampleFramesRequest",
    "SampleFramesResult",
    "SampledVideoFrame",
    "VideoHealthResult",
    "VideoMemoryTools",
    "VideoStatsRequest",
    "VideoStatsResult",
]
