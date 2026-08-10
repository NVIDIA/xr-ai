# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed events shared by independently composed sample applications."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_nat.events import EventTopic


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApplicationRequest(_Payload):
    text: str


class RawTranscript(_Payload):
    text: str


class ClockTick(_Payload):
    pass


class ApplicationReset(_Payload):
    pass


class BackgroundFact(_Payload):
    topic: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=500)
    source_ref: str | None = Field(default=None, max_length=500)


class OutputDestination(StrEnum):
    TEXT = "text"
    VOICE = "voice"


class OutputTiming(StrEnum):
    REPLY = "reply"
    NOTIFY = "notify"


class AudioTier(StrEnum):
    NORMAL = "normal"
    URGENT = "urgent"


class UserOutput(_Payload):
    text: str
    label: str = Field(default="guide.response", min_length=1, max_length=80)
    destinations: tuple[OutputDestination, ...] = (OutputDestination.VOICE,)
    timing: OutputTiming = OutputTiming.NOTIFY
    audio_tier: AudioTier = AudioTier.NORMAL


APPLICATION_REQUEST = EventTopic("application.request", ApplicationRequest)
RAW_TRANSCRIPT = EventTopic("voice.transcript", RawTranscript)
CLOCK_TICK = EventTopic("application.tick", ClockTick)
APPLICATION_RESET = EventTopic("application.reset", ApplicationReset)
BACKGROUND_FACT = EventTopic("application.fact", BackgroundFact)
USER_OUTPUT = EventTopic("user.output", UserOutput)

__all__ = [
    "APPLICATION_REQUEST",
    "ApplicationRequest",
    "AudioTier",
    "APPLICATION_RESET",
    "ApplicationReset",
    "BACKGROUND_FACT",
    "BackgroundFact",
    "CLOCK_TICK",
    "ClockTick",
    "OutputDestination",
    "OutputTiming",
    "RAW_TRANSCRIPT",
    "RawTranscript",
    "USER_OUTPUT",
    "UserOutput",
]
