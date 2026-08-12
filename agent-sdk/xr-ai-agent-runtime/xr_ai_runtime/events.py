# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed event contracts for agent communication."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

MessageT = TypeVar("MessageT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Topic(Generic[MessageT]):
    """A stable publish/subscribe name paired with its payload model."""

    name: str
    message_type: type[MessageT]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("topic name must not be empty")
        if not issubclass(self.message_type, BaseModel):
            raise TypeError("topic messages must be Pydantic models")

    def validate(self, message: MessageT | dict[str, Any]) -> MessageT:
        """Validate a message before delivery."""

        return self.message_type.model_validate(message)


@dataclass(frozen=True, slots=True)
class MessageMetadata:
    """Routing and trace context for one published event."""

    message_id: str
    correlation_id: str
    participant_id: str
    source: str
    parent_message_id: str | None
    timestamp_us: int


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
