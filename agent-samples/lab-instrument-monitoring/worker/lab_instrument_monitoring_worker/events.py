# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed topics shared by the sample's peer agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_runtime import Topic
from xr_ai_tools.marker_tracking import MarkerType
from xr_ai_voice import (
    UserQuery,
    VoiceInterrupted,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)


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
    marker_type: MarkerType
    marker_id: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    meter_reading: str = Field(min_length=1)


class InstrumentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    marker_type: MarkerType
    marker_id: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    meter_reading: str = Field(min_length=1)
    first_seen_us: int = Field(ge=0)
    last_seen_us: int = Field(ge=0)
    tracking: bool = True


class InstrumentChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["change"] = "change"
    timestamp_us: int = Field(ge=0)
    change_type: Literal["discovered", "reading_changed"]
    marker_type: MarkerType
    marker_id: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    previous_reading: str | None = None
    meter_reading: str = Field(min_length=1)
    last_seen_us: int = Field(ge=0)


class InstrumentLost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["lost"] = "lost"
    timestamp_us: int = Field(ge=0)
    marker_type: MarkerType
    marker_id: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    meter_reading: str = Field(min_length=1)
    last_seen_us: int = Field(ge=0)


class InstrumentStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["state"] = "state"
    timestamp_us: int = Field(ge=0)
    instruments: list[InstrumentState] = Field(default_factory=list)


class ForegroundRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_us: int = Field(ge=0)
    query: str = Field(min_length=1)
    response: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)


USER_QUERY_TOPIC = Topic("lab-instrument-monitoring.user-query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic(
    "lab-instrument-monitoring.participant-joined",
    VoiceParticipantJoined,
)
PARTICIPANT_LEFT_TOPIC = Topic(
    "lab-instrument-monitoring.participant-left",
    VoiceParticipantLeft,
)
INTERRUPTED_TOPIC = Topic(
    "lab-instrument-monitoring.interrupted",
    VoiceInterrupted,
)
MONITOR_RECORD_TOPIC = Topic("lab-instrument-monitoring.monitor-record", MonitorRecord)
INSTRUMENT_CHANGE_TOPIC = Topic(
    "lab-instrument-monitoring.instrument-change",
    InstrumentChange,
)
INSTRUMENT_LOST_TOPIC = Topic(
    "lab-instrument-monitoring.instrument-lost",
    InstrumentLost,
)
INSTRUMENT_STATE_TOPIC = Topic(
    "lab-instrument-monitoring.instrument-state",
    InstrumentStateSnapshot,
)
FOREGROUND_RECORD_TOPIC = Topic(
    "lab-instrument-monitoring.foreground-record",
    ForegroundRecord,
)


__all__ = [
    "FOREGROUND_RECORD_TOPIC",
    "INTERRUPTED_TOPIC",
    "INSTRUMENT_CHANGE_TOPIC",
    "INSTRUMENT_LOST_TOPIC",
    "INSTRUMENT_STATE_TOPIC",
    "MONITOR_RECORD_TOPIC",
    "PARTICIPANT_JOINED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "USER_QUERY_TOPIC",
    "ForegroundRecord",
    "InstrumentChange",
    "InstrumentLost",
    "InstrumentReading",
    "InstrumentState",
    "InstrumentStateSnapshot",
    "MonitorRecord",
]
