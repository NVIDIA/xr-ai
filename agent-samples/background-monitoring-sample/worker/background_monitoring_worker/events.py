# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed topics shared by the sample's peer agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_runtime import Topic
from xr_ai_voice import (
    UserQuery,
    VoiceInterrupted,
    VoiceParticipantLeft,
)


class ParticipantJoined(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_us: int = Field(ge=0)


class MonitorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_us: int = Field(ge=0)
    record_type: Literal["baseline", "observation", "unavailable", "error"]
    caption: str = ""
    changed: bool = False
    summary: str = ""
    error: str = ""


class InstrumentReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_us: int = Field(ge=0)
    qr_text: str = Field(min_length=1)
    meter_reading: str = Field(min_length=1)


class TranscriptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_us: int = Field(ge=0)
    text: str = Field(min_length=1)


class ForegroundRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_us: int = Field(ge=0)
    query: str = Field(min_length=1)
    response: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)


USER_QUERY_TOPIC = Topic("background-monitoring.user-query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic(
    "background-monitoring.participant-joined",
    ParticipantJoined,
)
PARTICIPANT_LEFT_TOPIC = Topic(
    "background-monitoring.participant-left",
    VoiceParticipantLeft,
)
INTERRUPTED_TOPIC = Topic(
    "background-monitoring.interrupted",
    VoiceInterrupted,
)
MONITOR_RECORD_TOPIC = Topic("background-monitoring.monitor-record", MonitorRecord)
INSTRUMENT_READING_TOPIC = Topic(
    "background-monitoring.instrument-reading",
    InstrumentReading,
)
TRANSCRIPT_RECORD_TOPIC = Topic(
    "background-monitoring.transcript-record",
    TranscriptRecord,
)
FOREGROUND_RECORD_TOPIC = Topic(
    "background-monitoring.foreground-record",
    ForegroundRecord,
)


__all__ = [
    "FOREGROUND_RECORD_TOPIC",
    "INTERRUPTED_TOPIC",
    "INSTRUMENT_READING_TOPIC",
    "MONITOR_RECORD_TOPIC",
    "PARTICIPANT_JOINED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "TRANSCRIPT_RECORD_TOPIC",
    "USER_QUERY_TOPIC",
    "ForegroundRecord",
    "InstrumentReading",
    "MonitorRecord",
    "ParticipantJoined",
    "TranscriptRecord",
]
