# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt selected sample topics for the live web-events viewer."""

from __future__ import annotations

from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import (
    VOICE_TRANSCRIPT_TOPIC,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
    VoiceTranscript,
)
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent

from .events import (
    _INSTRUMENT_TRACKING_TOPIC,
    FOREGROUND_RECORD_TOPIC,
    INSTRUMENT_CHANGE_TOPIC,
    INSTRUMENT_LOST_TOPIC,
    INSTRUMENT_STATE_TOPIC,
    MONITOR_RECORD_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    ForegroundRecord,
    InstrumentChange,
    InstrumentLost,
    InstrumentStateSnapshot,
    MonitorRecord,
    _InstrumentTrackingUpdate,
)


class WebEventsAdapterAgent(Agent):
    """Publish an explicit, compact browser projection of application events."""

    def __init__(self) -> None:
        super().__init__()

    @subscribe(MONITOR_RECORD_TOPIC)
    async def monitor_record(self, event: MonitorRecord, ctx: RuntimeContext) -> None:
        await self._publish(ctx, "monitor.observations", "Visual monitoring", event)

    @subscribe(INSTRUMENT_CHANGE_TOPIC)
    async def instrument_change(self, event: InstrumentChange, ctx: RuntimeContext) -> None:
        await self._publish(ctx, "instruments.changes", "Instrument changes", event)

    @subscribe(INSTRUMENT_LOST_TOPIC)
    async def instrument_lost(self, event: InstrumentLost, ctx: RuntimeContext) -> None:
        await self._publish(ctx, "instruments.tracking", "Instrument tracking", event)

    @subscribe(_INSTRUMENT_TRACKING_TOPIC)
    async def instrument_tracking(
        self,
        event: _InstrumentTrackingUpdate,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish(ctx, "instruments.tracking", "Instrument tracking", event)

    @subscribe(INSTRUMENT_STATE_TOPIC)
    async def instrument_state(self, event: InstrumentStateSnapshot, ctx: RuntimeContext) -> None:
        await self._publish(ctx, "instruments.state", "Instrument state", event)

    @subscribe(FOREGROUND_RECORD_TOPIC)
    async def foreground_record(self, event: ForegroundRecord, ctx: RuntimeContext) -> None:
        await self._publish(ctx, "foreground.responses", "Foreground responses", event)

    @subscribe(VOICE_TRANSCRIPT_TOPIC)
    async def transcript(self, event: VoiceTranscript, ctx: RuntimeContext) -> None:
        await self._publish(ctx, "voice.transcripts", "Voice transcripts", event)

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        await ctx.publish(
            WEB_EVENT_TOPIC,
            WebEvent(
                topic="participants.lifecycle",
                title="Participants",
                payload={"status": "joined"},
            ),
        )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        await ctx.publish(
            WEB_EVENT_TOPIC,
            WebEvent(
                topic="participants.lifecycle",
                title="Participants",
                payload={"status": "left"},
            ),
        )

    @staticmethod
    async def _publish(
        ctx: RuntimeContext,
        topic: str,
        title: str,
        event: MonitorRecord
        | InstrumentChange
        | InstrumentLost
        | InstrumentStateSnapshot
        | _InstrumentTrackingUpdate
        | ForegroundRecord
        | VoiceTranscript,
    ) -> None:
        await ctx.publish(
            WEB_EVENT_TOPIC,
            WebEvent(
                topic=topic,
                title=title,
                payload=event.model_dump(mode="json"),
            ),
        )


__all__ = ["WebEventsAdapterAgent"]
