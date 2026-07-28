# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts and client for historical recorded video."""

from pydantic import BaseModel, Field, model_validator

from .._models import _StrictRequest
from .._service.rpc import RPCClient


class ListRecordedParticipantsRequest(_StrictRequest):
    """Discovery operation that does not require arguments."""


class ListRecordedParticipantsResult(BaseModel):
    """Participant identities returned by a discovery operation."""

    participants: list[str] = Field(
        description="Exact participant identities that can be supplied to subsequent video calls."
    )


class VideoStatsRequest(_StrictRequest):
    """Identify one participant's persisted camera history."""

    participant_id: str = Field(
        min_length=1,
        description="Exact participant identity returned by list_recorded_participants.",
    )


class VideoStatsResult(BaseModel):
    """The stored range and size for one participant's camera history."""

    participant_id: str = Field(description="Participant identity used for the lookup.")
    num_chunks: int = Field(description="Number of H.264 recording chunks in the range.")
    total_bytes: int = Field(description="Total bytes across all recording chunks.")
    avg_chunk_bytes: int = Field(description="Integer average chunk size in bytes.")
    earliest_us: int = Field(description="Earliest available Unix-epoch timestamp in microseconds.")
    latest_us: int = Field(description="Latest available Unix-epoch timestamp in microseconds.")


class QueryVideoRequest(_StrictRequest):
    """Request an H.264 clip over an absolute recording window."""

    participant_id: str = Field(
        min_length=1,
        description="Exact participant identity returned by list_recorded_participants.",
    )
    start_us: int = Field(
        gt=0,
        description="Inclusive Unix-epoch start timestamp in microseconds.",
    )
    end_us: int = Field(
        gt=0,
        description="Inclusive Unix-epoch end timestamp in microseconds; must exceed start_us.",
    )

    @model_validator(mode="after")
    def validate_window(self) -> "QueryVideoRequest":
        if self.end_us <= self.start_us:
            raise ValueError("end_us must be greater than start_us")
        return self


class QueryVideoResult(BaseModel):
    """A locally written H.264 clip and the requested absolute window."""

    path: str = Field(description="Local path to the concatenated H.264 Annex B clip.")
    size: int = Field(description="Clip size in bytes.")
    start_us: int = Field(description="Requested Unix-epoch start timestamp in microseconds.")
    end_us: int = Field(description="Requested Unix-epoch end timestamp in microseconds.")


class HistoricalFrameRequest(_StrictRequest):
    """Extract one recorded frame relative to a known event timestamp."""

    participant_id: str = Field(
        min_length=1,
        description="Exact participant identity returned by list_recorded_participants.",
    )
    second_ago: int = Field(
        default=0,
        ge=0,
        description=(
            "Whole seconds before reference_time_us. Use 0 for the frame nearest "
            "the reference event, not for a live camera frame."
        ),
    )
    reference_time_us: int = Field(
        gt=0,
        description=(
            "Unix-epoch timestamp in microseconds for the event being examined, "
            "normally supplied by the calling workflow."
        ),
    )


class HistoricalFrameResult(BaseModel):
    """A PNG export and the exact recorded time selected for the request."""

    path: str = Field(description="Local path to the extracted PNG frame.")
    width: int = Field(description="Frame width in pixels.")
    height: int = Field(description="Frame height in pixels.")
    timestamp_us: int = Field(
        description="Unix-epoch timestamp in microseconds of the selected recorded frame."
    )
    second_ago: int = Field(description="Whole-second offset requested by the caller.")
    actual_second_ago: float = Field(
        description=(
            "Actual seconds before reference_time_us for the selected frame; "
            "fractional because the nearest recorded frame may not land on a full second."
        )
    )


class VideoHealthRequest(_StrictRequest):
    """Readiness probe that does not require arguments."""


class VideoHealthResult(BaseModel):
    """Service readiness used by compatibility and orchestration code."""

    ready: bool = Field(default=True, description="Whether the service is accepting RPC requests.")
    recording_enabled: bool = Field(
        description="Whether this service was configured with a recordings directory."
    )


class VideoMemoryClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 30.0) -> None:
        self._rpc = RPCClient(endpoint, timeout_s=timeout_s)

    async def list_recorded_participants(
        self, request: ListRecordedParticipantsRequest | None = None
    ) -> ListRecordedParticipantsResult:
        arguments = (request or ListRecordedParticipantsRequest()).model_dump()
        return ListRecordedParticipantsResult.model_validate(
            await self._rpc.call("list_recorded_participants", arguments)
        )

    async def get_video_stats(self, request: VideoStatsRequest) -> VideoStatsResult:
        return VideoStatsResult.model_validate(
            await self._rpc.call("get_video_stats", request.model_dump())
        )

    async def query_video(self, request: QueryVideoRequest) -> QueryVideoResult:
        return QueryVideoResult.model_validate(
            await self._rpc.call("query_video", request.model_dump())
        )

    async def get_frame_from_time(
        self, request: HistoricalFrameRequest
    ) -> HistoricalFrameResult:
        return HistoricalFrameResult.model_validate(
            await self._rpc.call("get_frame_from_time", request.model_dump())
        )

    async def get_health(self, request: VideoHealthRequest | None = None) -> VideoHealthResult:
        arguments = (request or VideoHealthRequest()).model_dump()
        return VideoHealthResult.model_validate(
            await self._rpc.call("get_health", arguments, timeout_s=2.0)
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
    "ListRecordedParticipantsRequest",
    "ListRecordedParticipantsResult",
    "QueryVideoRequest",
    "QueryVideoResult",
    "VideoMemoryClient",
    "VideoHealthRequest",
    "VideoHealthResult",
    "VideoStatsRequest",
    "VideoStatsResult",
]
