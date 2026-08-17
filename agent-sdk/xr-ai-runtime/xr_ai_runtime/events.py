# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed event contracts for agent communication."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

MessageT = TypeVar("MessageT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Topic(Generic[MessageT]):
    """A stable publish/subscribe name paired with its payload model."""

    name: str
    """Stable name shared by every publisher and subscriber."""

    message_type: type[MessageT]
    """Pydantic model used to validate each publication."""

    telemetry: Literal["full", "none"] = "full"
    """Whether Relay records publication and subscriber-delivery scopes."""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("topic name must not be empty")
        if not issubclass(self.message_type, BaseModel):
            raise TypeError("topic messages must be Pydantic models")
        if self.telemetry not in ("full", "none"):
            raise ValueError("topic telemetry must be 'full' or 'none'")

    def validate(self, message: MessageT | dict[str, Any]) -> MessageT:
        """Validate a message before delivery."""

        return self.message_type.model_validate(message)


@dataclass(frozen=True, slots=True)
class MessageMetadata:
    """Routing and trace context for one published event."""

    message_id: str
    """Unique identifier for this publication."""

    correlation_id: str
    """Identifier shared by publications in one logical operation."""

    participant_id: str | None
    """Participant routing key, or ``None`` for a global event."""

    source: str
    """Runtime-local name of the publisher."""

    parent_message_id: str | None
    """Identifier of the publication that caused this event, when present."""

    timestamp_us: int
    """Publication time as Unix microseconds."""


def subscribe(
    topic: Topic[MessageT],
) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
    """Register an agent method as a typed topic subscriber."""

    def decorate(method: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        topics = (*getattr(method, "__xr_ai_topics__", ()), topic)
        setattr(method, "__xr_ai_topics__", topics)
        return method

    return decorate


__all__ = ["MessageMetadata", "Topic", "subscribe"]
