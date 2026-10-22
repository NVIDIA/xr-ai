# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define participant interaction messages for the ``xr_ai_interaction`` package."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UserQuery(BaseModel):
    """One accepted user query from an input boundary such as voice or text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    """Gate-accepted speech or direct typed text."""

    timestamp_us: int = Field(ge=0)
    """Input presentation time as Unix microseconds."""


class ParticipantJoined(BaseModel):
    """Notification that one participant joined the interaction."""

    model_config = ConfigDict(extra="forbid")


class ParticipantLeft(BaseModel):
    """Notification that one participant left the interaction."""

    model_config = ConfigDict(extra="forbid")


class Interrupted(BaseModel):
    """Notification that participant-scoped or global work was interrupted."""

    model_config = ConfigDict(extra="forbid")


class ResponseOutput(BaseModel):
    """One complete response or one chunk of an incremental response.

    The consumer renders it, for example as speech or text.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    """Complete response text or one incremental fragment."""

    response_id: str | None = Field(default=None, min_length=1)
    """Identifier shared by incremental chunks; recent reuse is ignored with one warning."""

    final: bool = True
    """Whether this message completes the response."""

    interrupt: bool = False
    """Whether the first response message supersedes active output."""

    timestamp_us: int | None = Field(default=None, ge=0)
    """Optional originating input timestamp propagated to the consumer."""

    @model_validator(mode="after")
    def validate_boundary(self) -> ResponseOutput:
        """Require identity for incremental output and text for finite output."""

        if not self.final and self.response_id is None:
            raise ValueError("non-final response output needs a response_id")
        if self.response_id is None and not self.text.strip():
            raise ValueError("complete response output must contain text")
        if self.interrupt and self.response_id is not None and not self.text.strip():
            raise ValueError("an empty stream terminator cannot interrupt output")
        return self
