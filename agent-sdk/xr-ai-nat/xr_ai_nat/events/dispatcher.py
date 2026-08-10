# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin typed delivery adapter for NAT functions."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from typing import Any

from nat.builder.function import Function

from .models import EventEnvelope, EventTopic, PayloadT

EventObserver = Callable[[EventEnvelope, tuple[str, ...]], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class _Subscription:
    subscriber_id: str
    function: Function


class EventDispatcher:
    """Validate events and deliver them to explicitly registered NAT functions."""

    def __init__(self, observer: EventObserver | None = None) -> None:
        self._topics: dict[str, EventTopic[Any]] = {}
        self._subscriptions: dict[str, list[_Subscription]] = {}
        self._observer = observer

    def subscribe(
        self,
        topic: EventTopic[PayloadT],
        *,
        subscriber_id: str,
        function: Function,
    ) -> None:
        self._register_topic(topic)
        subscriptions = self._subscriptions.setdefault(topic.name, [])
        if any(item.subscriber_id == subscriber_id for item in subscriptions):
            raise ValueError(f"subscriber {subscriber_id!r} already handles {topic.name!r}")
        subscriptions.append(_Subscription(subscriber_id, function))

    async def publish(
        self,
        topic: EventTopic[PayloadT],
        *,
        participant_id: str,
        producer: str,
        payload: PayloadT | dict[str, Any],
        subscribers: Collection[str] | None = None,
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
        timestamp_us: int | None = None,
    ) -> tuple[Any, ...]:
        self._register_topic(topic)
        event = topic.envelope(
            participant_id=participant_id,
            producer=producer,
            payload=payload,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            timestamp_us=timestamp_us,
        )
        return await self.deliver(event, subscribers=subscribers)

    async def deliver(
        self,
        event: EventEnvelope,
        *,
        subscribers: Collection[str] | None = None,
    ) -> tuple[Any, ...]:
        topic = self._topics.get(event.topic)
        if topic is None:
            return ()
        topic.payload_from(event)
        selected = set(subscribers) if subscribers is not None else None
        subscriptions = tuple(
            subscription
            for subscription in self._subscriptions.get(event.topic, ())
            if selected is None or subscription.subscriber_id in selected
        )
        if self._observer is not None:
            observed = self._observer(
                event,
                tuple(subscription.subscriber_id for subscription in subscriptions),
            )
            if inspect.isawaitable(observed):
                await observed
        results: list[Any] = []
        for subscription in subscriptions:
            results.append(await subscription.function.ainvoke(event))
        return tuple(results)

    def _register_topic(self, topic: EventTopic[PayloadT]) -> None:
        known = self._topics.setdefault(topic.name, topic)
        if known.payload_type is not topic.payload_type:
            raise ValueError(f"event topic {topic.name!r} already uses another payload type")


__all__ = ["EventDispatcher", "EventObserver"]
