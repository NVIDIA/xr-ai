# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from unittest.mock import ANY

from nat.builder.workflow_builder import WorkflowBuilder
from tea_making_worker.applications.events import (
    USER_OUTPUT,
    AudioTier,
    OutputDestination,
    OutputTiming,
    UserOutput,
)
from tea_making_worker.applications.output import UserOutputDelivery
from xr_ai_nat.events import EventDispatcher


class _Transport:
    def __init__(self) -> None:
        self.messages = []

    async def send_return_data(self, message) -> None:
        self.messages.append(message)


class _Session:
    def __init__(self) -> None:
        self.transport = _Transport()
        self.responses = []

    async def enqueue_response(self, participant_id, text, **kwargs) -> None:
        self.responses.append((participant_id, text, kwargs))


class _VoicePolicy:
    async def ainvoke(self, output, *, to_type):
        return output.model_copy(update={"text": f"Policy: {output.text}", "audio_tier": AudioTier.URGENT})


class UserOutputDeliveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_turn_reply_returns_to_existing_voice_turn(self) -> None:
        session = _Session()
        async with WorkflowBuilder() as builder:
            output = UserOutputDelivery(EventDispatcher(), session)  # type: ignore[arg-type]
            await output.build(builder)

            result = await output.publish(
                "alice",
                "assistant",
                UserOutput(text="Here is the answer.", timing=OutputTiming.REPLY),
            )

        self.assertEqual(result, "Here is the answer.")
        self.assertEqual(session.responses, [])

    async def test_notification_queues_voice_and_urgent_output_can_interrupt(self) -> None:
        session = _Session()
        async with WorkflowBuilder() as builder:
            output = UserOutputDelivery(EventDispatcher(), session)  # type: ignore[arg-type]
            await output.build(builder)

            result = await output.publish(
                "alice",
                "safety monitor",
                UserOutput(text="Move your hand away.", audio_tier=AudioTier.URGENT),
            )

        self.assertEqual(result, "")
        self.assertEqual(
            session.responses,
            [("alice", "Move your hand away.", {"interrupt": True, "pts_us": ANY})],
        )

    async def test_text_destination_uses_application_label_without_voice(self) -> None:
        session = _Session()
        async with WorkflowBuilder() as builder:
            output = UserOutputDelivery(EventDispatcher(), session)  # type: ignore[arg-type]
            await output.build(builder)

            await output.publish(
                "alice",
                "visual monitor",
                UserOutput(
                    text="A parcel moved.",
                    label="Visual monitor",
                    destinations=(OutputDestination.TEXT,),
                ),
            )

        self.assertEqual(session.responses, [])
        self.assertEqual(len(session.transport.messages), 1)
        self.assertEqual(session.transport.messages[0].topic, "Visual monitor")
        self.assertEqual(session.transport.messages[0].data.decode(), "A parcel moved.")

    async def test_text_only_event_is_safe_when_broadcast(self) -> None:
        session = _Session()
        events = EventDispatcher()
        async with WorkflowBuilder() as builder:
            output = UserOutputDelivery(events, session)  # type: ignore[arg-type]
            await output.build(builder)

            await events.publish(
                USER_OUTPUT,
                participant_id="alice",
                producer="visual monitor",
                payload=UserOutput(
                    text="A parcel moved.",
                    label="Visual monitor",
                    destinations=(OutputDestination.TEXT,),
                ),
            )

        self.assertEqual(session.responses, [])
        self.assertEqual(len(session.transport.messages), 1)

    async def test_voice_policy_is_a_replaceable_nat_function(self) -> None:
        session = _Session()
        async with WorkflowBuilder() as builder:
            output = UserOutputDelivery(
                EventDispatcher(),
                session,  # type: ignore[arg-type]
                voice_policy=_VoicePolicy(),  # type: ignore[arg-type]
            )
            await output.build(builder)

            await output.publish("alice", "monitor", UserOutput(text="Check the scene."))

        self.assertEqual(
            session.responses,
            [("alice", "Policy: Check the scene.", {"interrupt": True, "pts_us": ANY})],
        )


if __name__ == "__main__":
    unittest.main()
