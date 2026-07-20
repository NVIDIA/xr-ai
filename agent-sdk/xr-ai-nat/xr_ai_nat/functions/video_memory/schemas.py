# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed requests and results for live and recorded video memory."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoStatsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participant_id: str


class VideoStatsResult(BaseModel):
    participant_id: str
    num_chunks: int
    total_bytes: int
    avg_chunk_bytes: int
    earliest_us: int
    latest_us: int


class QueryVideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participant_id: str
    start_us: int
    end_us: int

    @model_validator(mode="after")
    def validate_window(self) -> "QueryVideoRequest":
        if self.end_us < self.start_us:
            raise ValueError("end_us must be greater than or equal to start_us")
        return self


class QueryVideoResult(BaseModel):
    path: str
    size: int
    start_us: int
    end_us: int


class FrameAtTimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participant_id: str
    second_ago: int = Field(default=0, ge=0)
    reference_time_us: int = Field(default=0, ge=0)


class FrameResult(BaseModel):
    path: str
    width: int
    height: int
    timestamp_us: int
    second_ago: int
    actual_second_ago: float


class ParticipantsResult(BaseModel):
    participants: list[str]


class VideoMemoryHealth(BaseModel):
    ready: bool = True
    recording_enabled: bool


__all__ = [
    "EmptyRequest",
    "FrameAtTimeRequest",
    "FrameResult",
    "ParticipantsResult",
    "QueryVideoRequest",
    "QueryVideoResult",
    "VideoMemoryHealth",
    "VideoStatsRequest",
    "VideoStatsResult",
]
