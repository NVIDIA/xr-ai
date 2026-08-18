# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded thread-safe storage for live web events."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from xr_ai_runtime import MessageMetadata

from ._models import WebEvent


@dataclass(frozen=True, slots=True)
class _StoredEvent:
    sequence: int
    event: WebEvent
    metadata: MessageMetadata

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "topic": self.event.topic,
            "title": self.event.title,
            "participant_id": self.metadata.participant_id,
            "source": self.metadata.source,
            "timestamp_us": self.metadata.timestamp_us,
            "message_id": self.metadata.message_id,
            "correlation_id": self.metadata.correlation_id,
            "parent_message_id": self.metadata.parent_message_id,
            "payload": self.event.payload,
        }


class _EventStore:
    def __init__(self, capacity: int) -> None:
        self._events: deque[_StoredEvent] = deque(maxlen=capacity)
        self._next_sequence = 1
        self._lock = threading.Lock()

    def append(self, event: WebEvent, metadata: MessageMetadata) -> None:
        with self._lock:
            self._events.append(
                _StoredEvent(
                    sequence=self._next_sequence,
                    event=event.model_copy(deep=True),
                    metadata=metadata,
                )
            )
            self._next_sequence += 1

    def after(self, sequence: int) -> dict[str, Any]:
        with self._lock:
            latest = self._next_sequence - 1
            oldest = self._events[0].sequence if self._events else self._next_sequence
            reset = sequence > latest or sequence < oldest - 1
            threshold = oldest - 1 if reset else sequence
            events = [item.as_dict() for item in self._events if item.sequence > threshold]
            return {
                "events": events,
                "cursor": latest,
                "oldest": oldest,
                "reset": reset,
            }
