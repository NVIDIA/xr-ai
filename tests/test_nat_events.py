# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for typed NAT-function event delivery."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from xr_ai_nat.events import EventDispatcher, EventEnvelope, EventTopic, PeriodicEventSource


class _Notice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class _Function:
    def __init__(self, result: str) -> None:
        self.result = result
        self.events: list[EventEnvelope] = []

    async def ainvoke(self, event: EventEnvelope) -> str:
        self.events.append(event)
        return self.result


async def test_dispatcher_validates_once_and_invokes_selected_nat_functions() -> None:
    topic = EventTopic("user.output", _Notice)
    dispatcher = EventDispatcher()
    voice = _Function("voice")
    text = _Function("text")
    dispatcher.subscribe(topic, subscriber_id="voice", function=voice)  # type: ignore[arg-type]
    dispatcher.subscribe(topic, subscriber_id="text", function=text)  # type: ignore[arg-type]

    result = await dispatcher.publish(
        topic,
        participant_id="alice",
        producer="guide",
        payload={"text": "Water is heating."},
        subscribers={"voice"},
        correlation_id="turn-7",
        parent_event_id="observation-2",
        timestamp_us=42,
    )

    assert result == ("voice",)
    assert text.events == []
    assert len(voice.events) == 1
    event = voice.events[0]
    assert event.participant_id == "alice"
    assert event.correlation_id == "turn-7"
    assert event.parent_event_id == "observation-2"
    assert event.timestamp_us == 42
    assert topic.payload_from(event) == _Notice(text="Water is heating.")


async def test_dispatcher_rejects_invalid_payload_before_delivery() -> None:
    topic = EventTopic("user.output", _Notice)
    dispatcher = EventDispatcher()
    function = _Function("unused")
    dispatcher.subscribe(topic, subscriber_id="voice", function=function)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        await dispatcher.publish(
            topic,
            participant_id="alice",
            producer="guide",
            payload={"unexpected": "value"},
        )

    assert function.events == []


async def test_dispatcher_preserves_subscription_order_for_shared_state() -> None:
    topic = EventTopic("user.output", _Notice)
    dispatcher = EventDispatcher()
    first = _Function("first")
    second = _Function("second")
    dispatcher.subscribe(topic, subscriber_id="first", function=first)  # type: ignore[arg-type]
    dispatcher.subscribe(topic, subscriber_id="second", function=second)  # type: ignore[arg-type]

    result = await dispatcher.publish(
        topic,
        participant_id="alice",
        producer="guide",
        payload=_Notice(text="Done."),
    )

    assert result == ("first", "second")


async def test_dispatcher_observer_sees_selected_delivery_and_trace_metadata() -> None:
    observed: list[tuple[EventEnvelope, tuple[str, ...]]] = []
    dispatcher = EventDispatcher(lambda event, subscribers: observed.append((event, subscribers)))
    topic = EventTopic("user.output", _Notice)
    function = _Function("ok")
    dispatcher.subscribe(topic, subscriber_id="voice", function=function)  # type: ignore[arg-type]

    await dispatcher.publish(
        topic,
        participant_id="alice",
        producer="guide",
        payload=_Notice(text="Done."),
        correlation_id="turn-1",
    )

    assert observed[0][0].correlation_id == "turn-1"
    assert observed[0][1] == ("voice",)


def test_dispatcher_rejects_conflicting_topic_schemas_and_duplicate_subscribers() -> None:
    class Other(BaseModel):
        value: int

    topic = EventTopic("same", _Notice)
    dispatcher = EventDispatcher()
    function = _Function("ok")
    dispatcher.subscribe(topic, subscriber_id="one", function=function)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="another payload type"):
        dispatcher.subscribe(
            EventTopic("same", Other),
            subscriber_id="two",
            function=function,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="already handles"):
        dispatcher.subscribe(topic, subscriber_id="one", function=function)  # type: ignore[arg-type]


async def test_periodic_source_runs_only_between_explicit_start_and_stop() -> None:
    topic = EventTopic("monitor.tick", _Notice)
    dispatcher = EventDispatcher()
    function = _Function("ok")
    dispatcher.subscribe(topic, subscriber_id="monitor", function=function)  # type: ignore[arg-type]
    source = PeriodicEventSource(
        dispatcher,
        topic,
        payload=lambda _participant_id: _Notice(text="tick"),
        producer="monitor.clock",
        subscriber_id="monitor",
        interval_s=0.01,
    )
    supervisor = asyncio.create_task(source.run())

    assert source.start("alice")
    assert not source.start("alice")
    for _ in range(20):
        if len(function.events) >= 2:
            break
        await asyncio.sleep(0.01)
    assert len(function.events) >= 2
    assert all(event.participant_id == "alice" for event in function.events)

    assert source.start("bob")
    for _ in range(20):
        if any(event.participant_id == "bob" for event in function.events):
            break
        await asyncio.sleep(0.01)
    assert any(event.participant_id == "bob" for event in function.events)

    assert await source.stop("alice")
    alice_count = sum(event.participant_id == "alice" for event in function.events)
    bob_count = sum(event.participant_id == "bob" for event in function.events)
    await asyncio.sleep(0.03)
    assert sum(event.participant_id == "alice" for event in function.events) == alice_count
    assert sum(event.participant_id == "bob" for event in function.events) > bob_count
    assert await source.stop("bob")
    await source.close()
    await supervisor


async def test_periodic_source_propagates_subscriber_failure() -> None:
    class BrokenFunction:
        async def ainvoke(self, _event: EventEnvelope) -> None:
            raise RuntimeError("subscriber failed")

    topic = EventTopic("monitor.tick", _Notice)
    dispatcher = EventDispatcher()
    dispatcher.subscribe(
        topic,
        subscriber_id="monitor",
        function=BrokenFunction(),  # type: ignore[arg-type]
    )
    source = PeriodicEventSource(
        dispatcher,
        topic,
        payload=lambda _participant_id: _Notice(text="tick"),
        producer="monitor.clock",
        subscriber_id="monitor",
        interval_s=1,
    )
    source.start("alice")

    with pytest.raises(RuntimeError, match="subscriber failed"):
        await asyncio.wait_for(source.run(), timeout=1)
