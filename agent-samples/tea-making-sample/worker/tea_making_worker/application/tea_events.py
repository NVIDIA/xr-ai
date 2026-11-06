# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed topics shared by the tea-making sample's peer agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_interaction import Interrupted, ParticipantJoined, ParticipantLeft, UserQuery
from xr_ai_runtime import Topic


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuidanceRecord(_Event):
    run_id: str | None = None
    timestamp_us: int = Field(ge=0)
    event: str
    step_id: str | None = None
    message: str
    state: dict[str, Any] = Field(default_factory=dict)


class GuidanceNotice(_Event):
    timestamp_us: int = Field(ge=0)
    text: str


USER_QUERY_TOPIC = Topic("tea-making.user-query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic(
    "tea-making.participant-joined",
    ParticipantJoined,
)
PARTICIPANT_LEFT_TOPIC = Topic(
    "tea-making.participant-left",
    ParticipantLeft,
)
INTERRUPTED_TOPIC = Topic("tea-making.interrupted", Interrupted)
GUIDANCE_RECORD_TOPIC = Topic("tea-making.guidance-record", GuidanceRecord)
GUIDANCE_NOTICE_TOPIC = Topic("tea-making.guidance-notice", GuidanceNotice)
__all__ = [
    "GUIDANCE_NOTICE_TOPIC",
    "GUIDANCE_RECORD_TOPIC",
    "INTERRUPTED_TOPIC",
    "PARTICIPANT_JOINED_TOPIC",
    "PARTICIPANT_LEFT_TOPIC",
    "USER_QUERY_TOPIC",
    "GuidanceNotice",
    "GuidanceRecord",
]
