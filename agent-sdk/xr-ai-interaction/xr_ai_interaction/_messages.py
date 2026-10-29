# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define participant interaction messages for the ``xr_ai_interaction`` package."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class UserQuery(BaseModel):
    """One accepted user query from an input boundary such as voice or text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    """Gate-accepted speech or direct typed text."""

    timestamp_us: int = Field(ge=0)
    """Input presentation time as Unix microseconds."""

    request_id: str | None = Field(default=None, min_length=1)
    """Optional input-owned request identity, including semantic voice turns.

    Consumers use this identity when present; otherwise use the runtime
    message ID. Responses must answer the same identity. An unset identity
    is omitted from serialized payloads, preserving ordinary query payloads.
    """

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        result = handler(self)
        if self.request_id is None:
            result.pop("request_id", None)
        return result


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
    """One complete response or one fragment of a streamed response.

    A producer declares what kind of output it emits; the consumer decides
    when and how to render it, for example as speech or text. A ``reply``
    answers one request. An ``update`` is unprompted information. A ``withdraw``
    carries no content: it asks the consumer to drop the updates this producer
    queued for the participant that have not started playing.

    Without ``response_id``, this is a complete standalone response. Otherwise,
    ordered messages with the same participant, producer, and ``response_id``
    form one streamed reply: append each message's text until ``final=True``
    closes it. The final message may contain text or be an empty terminator.

    This model validates individual messages, not stream state. Consumers own
    stream tracking and their duplicate-handling policy.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["reply", "update", "withdraw"]
    """``reply`` answers ``request_id``; ``update`` is unprompted information;
    ``withdraw`` drops this producer's not-yet-played updates."""

    text: str = ""
    """Complete response text or one streamed fragment."""

    request_id: str | None = Field(default=None, min_length=1)
    """Request a reply answers. Required for replies and forbidden for updates."""

    urgent: bool = False
    """Updates only: cut current output and play this update next."""

    response_id: str | None = Field(default=None, min_length=1)
    """Identifier shared by the fragments of one streamed reply from one participant and producer."""

    final: bool = True
    """Whether this message completes the response."""

    timestamp_us: int | None = Field(default=None, ge=0)
    """Optional originating input timestamp propagated to the consumer."""

    @model_validator(mode="after")
    def validate_boundary(self) -> ResponseOutput:
        """Require identity for streamed replies and text for finite output."""

        if self.kind == "withdraw":
            if self.text or self.request_id is not None or self.response_id is not None:
                raise ValueError("a withdraw carries no text, request_id, or response_id")
            if not self.final or self.urgent:
                raise ValueError("a withdraw is final and not urgent")
            return self
        if not self.final and self.response_id is None:
            raise ValueError("non-final response output needs a response_id")
        if self.response_id is None and not self.text.strip():
            raise ValueError("complete response output must contain text")
        if self.kind == "reply" and self.request_id is None:
            raise ValueError("a reply needs a request_id")
        if self.kind == "update":
            if self.request_id is not None:
                raise ValueError("an update cannot have a request_id")
            if self.response_id is not None or not self.final:
                raise ValueError("an update must be finite")
        if self.urgent and self.kind != "update":
            raise ValueError("only an update can be urgent")
        return self
