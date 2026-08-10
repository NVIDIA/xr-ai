# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral event values for NAT application functions."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

PayloadT = TypeVar("PayloadT", bound=BaseModel)


class EventEnvelope(BaseModel):
    """One participant-scoped event with traceable ancestry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str = Field(min_length=1)
    participant_id: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    payload: dict[str, Any]
    event_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    parent_event_id: str | None = None
    timestamp_us: int


@dataclass(frozen=True, slots=True)
class EventTopic(Generic[PayloadT]):
    """A stable event name paired with its validated payload model."""

    name: str
    payload_type: type[PayloadT]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("event topic name must not be empty")

    def envelope(
        self,
        *,
        participant_id: str,
        producer: str,
        payload: PayloadT | dict[str, Any],
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
        timestamp_us: int | None = None,
    ) -> EventEnvelope:
        value = self.payload_type.model_validate(payload)
        event_id = uuid.uuid4().hex
        return EventEnvelope(
            topic=self.name,
            participant_id=participant_id,
            producer=producer,
            payload=value.model_dump(mode="json"),
            event_id=event_id,
            correlation_id=correlation_id or event_id,
            parent_event_id=parent_event_id,
            timestamp_us=timestamp_us if timestamp_us is not None else time.time_ns() // 1_000,
        )

    def payload_from(self, event: EventEnvelope) -> PayloadT:
        if event.topic != self.name:
            raise ValueError(f"expected event topic {self.name!r}, got {event.topic!r}")
        return self.payload_type.model_validate(event.payload)


__all__ = ["EventEnvelope", "EventTopic", "PayloadT"]
