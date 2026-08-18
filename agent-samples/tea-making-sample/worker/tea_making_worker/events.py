# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed topics shared by the tea-making sample's peer agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_runtime import Topic
from xr_ai_voice import (
    UserQuery,
    VoiceInterrupted,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForegroundRecord(_Event):
    timestamp_us: int = Field(ge=0)
    query: str
    response: str
    tools: list[str] = Field(default_factory=list)


class GuidanceRecord(_Event):
    timestamp_us: int = Field(ge=0)
    event: str
    step_id: str | None = None
    message: str
    state: dict[str, Any] = Field(default_factory=dict)


class GuidanceNotice(_Event):
    timestamp_us: int = Field(ge=0)
    text: str


class BackgroundFact(_Event):
    timestamp_us: int = Field(ge=0)
    application: Literal["change_watch", "transcript", "video_log"]
    text: str = Field(min_length=1, max_length=500)


class ChangeWatchRecord(_Event):
    timestamp_us: int = Field(ge=0)
    record_type: Literal[
        "started",
        "baseline",
        "observation",
        "unavailable",
        "error",
        "stopped",
    ]
    instruction: str = ""
    caption: str = ""
    important: bool = False
    duplicate: bool = False
    summary: str = ""
    error: str = ""


class TranscriptRecord(_Event):
    timestamp_us: int = Field(ge=0)
    record_type: Literal["started", "utterance", "summary", "error", "stopped"]
    text: str = ""
    turn_count: int = Field(default=0, ge=0)
    error: str = ""


class VideoLogRecord(_Event):
    timestamp_us: int = Field(ge=0)
    record_type: Literal["started", "observation", "unavailable", "error", "stopped"]
    caption: str = ""
    delta: str = ""
    error: str = ""


USER_QUERY_TOPIC = Topic("tea-making.user-query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic(
    "tea-making.participant-joined",
    VoiceParticipantJoined,
)
PARTICIPANT_LEFT_TOPIC = Topic(
    "tea-making.participant-left",
    VoiceParticipantLeft,
)
INTERRUPTED_TOPIC = Topic("tea-making.interrupted", VoiceInterrupted)
FOREGROUND_RECORD_TOPIC = Topic("tea-making.foreground-record", ForegroundRecord)
GUIDANCE_RECORD_TOPIC = Topic("tea-making.guidance-record", GuidanceRecord)
GUIDANCE_NOTICE_TOPIC = Topic("tea-making.guidance-notice", GuidanceNotice)
BACKGROUND_FACT_TOPIC = Topic("tea-making.background-fact", BackgroundFact)
CHANGE_WATCH_RECORD_TOPIC = Topic(
    "tea-making.change-watch-record",
    ChangeWatchRecord,
)
TRANSCRIPT_RECORD_TOPIC = Topic("tea-making.transcript-record", TranscriptRecord)
VIDEO_LOG_RECORD_TOPIC = Topic("tea-making.video-log-record", VideoLogRecord)


__all__ = [
    "BACKGROUND_FACT_TOPIC",
    "CHANGE_WATCH_RECORD_TOPIC",
    "FOREGROUND_RECORD_TOPIC",
    "GUIDANCE_NOTICE_TOPIC",
    "GUIDANCE_RECORD_TOPIC",
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_JOINED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "TRANSCRIPT_RECORD_TOPIC",
    "USER_QUERY_TOPIC",
    "VIDEO_LOG_RECORD_TOPIC",
    "BackgroundFact",
    "ChangeWatchRecord",
    "ForegroundRecord",
    "GuidanceNotice",
    "GuidanceRecord",
    "TranscriptRecord",
    "VideoLogRecord",
]
