# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable text and serialized voice consumers for user output events."""

from __future__ import annotations

import time

from nat.builder.function import Function
from nat.plugin_api import Builder
from xr_ai_hub import DataMessage
from xr_ai_nat.events import EventDispatcher, EventEnvelope, add_event_handler
from xr_ai_voice import VoiceSession

from ..runtime.events import emit
from .events import (
    USER_OUTPUT,
    AudioTier,
    OutputDestination,
    OutputTiming,
    UserOutput,
)


class UserOutputDelivery:
    """Publish once, then select independent text and voice NAT consumers."""

    TEXT_SUBSCRIBER = "output.text"
    VOICE_SUBSCRIBER = "output.voice"

    def __init__(
        self,
        dispatcher: EventDispatcher,
        session: VoiceSession,
        *,
        voice_policy: Function | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.session = session
        self.text = TextOutputBridge(session)
        self.voice_policy = voice_policy

    async def build(self, builder: Builder) -> None:
        text = await add_event_handler(
            builder,
            name="output__text",
            handler=self._send_text,
            description="Display labeled application output as text.",
        )
        voice = await add_event_handler(
            builder,
            name="output__voice",
            handler=self._send_voice,
            description="Return or queue one participant's serialized voice output.",
        )
        self.dispatcher.subscribe(USER_OUTPUT, subscriber_id=self.TEXT_SUBSCRIBER, function=text)
        self.dispatcher.subscribe(USER_OUTPUT, subscriber_id=self.VOICE_SUBSCRIBER, function=voice)

    async def publish(
        self,
        participant_id: str,
        producer: str,
        output: UserOutput,
        *,
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> str:
        subscribers = []
        if OutputDestination.TEXT in output.destinations:
            subscribers.append(self.TEXT_SUBSCRIBER)
        if OutputDestination.VOICE in output.destinations:
            subscribers.append(self.VOICE_SUBSCRIBER)
        results = await self.dispatcher.publish(
            USER_OUTPUT,
            participant_id=participant_id,
            producer=producer,
            payload=output,
            subscribers=subscribers,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )
        return next((result for result in results if isinstance(result, str) and result), "")

    async def _send_text(self, event: EventEnvelope) -> None:
        output = USER_OUTPUT.payload_from(event)
        if OutputDestination.TEXT not in output.destinations:
            return
        await self.text.send(event.participant_id, output.label, output.text)

    async def _send_voice(self, event: EventEnvelope) -> str:
        output = USER_OUTPUT.payload_from(event)
        if OutputDestination.VOICE not in output.destinations:
            return ""
        if self.voice_policy is not None:
            output = await self.voice_policy.ainvoke(output, to_type=UserOutput)
        if output.timing == OutputTiming.REPLY:
            return output.text
        await self.session.enqueue_response(
            event.participant_id,
            output.text,
            interrupt=output.audio_tier == AudioTier.URGENT,
            pts_us=event.timestamp_us,
        )
        return ""


class TextOutputBridge:
    """Send labeled application text directly to the participant data channel."""

    def __init__(self, session: VoiceSession) -> None:
        self._session = session

    async def send(self, participant_id: str, label: str, message: str) -> None:
        text = message.strip()
        if not text:
            return
        await self._session.transport.send_return_data(
            DataMessage(
                participant_id=participant_id,
                topic=label,
                pts_us=time.time_ns() // 1_000,
                data=text.encode(),
            )
        )
        emit(
            "application.text_output",
            participant_id=participant_id,
            application=label,
            message=text,
        )


__all__ = ["TextOutputBridge", "UserOutputDelivery"]
