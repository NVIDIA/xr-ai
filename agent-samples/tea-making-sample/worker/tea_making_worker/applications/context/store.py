# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded in-memory context records; durable detail remains in application logs."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from itertools import islice

from .models import ContextItem, ContextPublishRequest, ContextQueryRequest


class ApplicationContextStore:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: dict[str, deque[ContextItem]] = {}
        self._sequences: defaultdict[str, int] = defaultdict(int)

    def publish(self, participant_id: str, request: ContextPublishRequest) -> ContextItem:
        sequence = self._sequences[participant_id] + 1
        self._sequences[participant_id] = sequence
        item = ContextItem(
            sequence=sequence,
            producer=request.producer,
            topic=request.topic,
            summary=request.summary.strip(),
            observed_at_us=time.time_ns() // 1_000,
            source_ref=request.source_ref,
        )
        self._items.setdefault(participant_id, deque(maxlen=self._capacity)).append(item)
        return item

    def query(self, participant_id: str, request: ContextQueryRequest) -> tuple[ContextItem, ...]:
        cutoff = time.time_ns() // 1_000 - int(request.max_age_s * 1_000_000)
        topics = set(request.topics)
        matches = (
            item
            for item in reversed(self._items.get(participant_id, ()))
            if item.observed_at_us >= cutoff and (not topics or item.topic in topics)
        )
        return tuple(reversed(tuple(islice(matches, request.max_items))))

    def clear(self, participant_id: str) -> int:
        removed = len(self._items.pop(participant_id, ()))
        self._sequences.pop(participant_id, None)
        return removed


__all__ = ["ApplicationContextStore"]
