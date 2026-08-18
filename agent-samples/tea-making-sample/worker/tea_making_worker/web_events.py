# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit tea-making event projection for the live browser viewer."""

from __future__ import annotations

from pydantic import BaseModel, JsonValue
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_voice import VoiceParticipantJoined, VoiceParticipantLeft
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent

from .events import (
    BACKGROUND_FACT_TOPIC,
    CHANGE_WATCH_RECORD_TOPIC,
    FOREGROUND_RECORD_TOPIC,
    GUIDANCE_NOTICE_TOPIC,
    GUIDANCE_RECORD_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    TRANSCRIPT_RECORD_TOPIC,
    VIDEO_LOG_RECORD_TOPIC,
    BackgroundFact,
    ChangeWatchRecord,
    ForegroundRecord,
    GuidanceNotice,
    GuidanceRecord,
    TranscriptRecord,
    VideoLogRecord,
)


class TeaWebEventsAgent(Agent):
    """Project selected typed tea events into compact presentation events."""

    def __init__(self) -> None:
        super().__init__()

    @subscribe(FOREGROUND_RECORD_TOPIC)
    async def foreground(
        self,
        record: ForegroundRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish("foreground", "Foreground answers", record, ctx)

    @subscribe(GUIDANCE_RECORD_TOPIC)
    async def guidance_record(
        self,
        record: GuidanceRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish("guidance.events", "Guidance events", record, ctx)

    @subscribe(GUIDANCE_NOTICE_TOPIC)
    async def guidance_notice(
        self,
        notice: GuidanceNotice,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish("guidance.notices", "Guidance notices", notice, ctx)

    @subscribe(BACKGROUND_FACT_TOPIC)
    async def background_fact(
        self,
        fact: BackgroundFact,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish("background.facts", "Background facts", fact, ctx)

    @subscribe(CHANGE_WATCH_RECORD_TOPIC)
    async def change_watch(
        self,
        record: ChangeWatchRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish(
            "background.change-watch",
            "Visual change watch",
            record,
            ctx,
        )

    @subscribe(TRANSCRIPT_RECORD_TOPIC)
    async def transcript(
        self,
        record: TranscriptRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish("background.transcript", "Transcript", record, ctx)

    @subscribe(VIDEO_LOG_RECORD_TOPIC)
    async def video_log(
        self,
        record: VideoLogRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish("background.video-log", "Video log", record, ctx)

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish_payload(
            "participant.lifecycle",
            "Participant lifecycle",
            {"event": "joined"},
            ctx,
        )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish_payload(
            "participant.lifecycle",
            "Participant lifecycle",
            {"event": "left"},
            ctx,
        )

    async def _publish(
        self,
        topic: str,
        title: str,
        record: BaseModel,
        ctx: RuntimeContext,
    ) -> None:
        await self._publish_payload(
            topic,
            title,
            record.model_dump(
                mode="json",
                exclude_defaults=True,
                exclude_none=True,
            ),
            ctx,
        )

    @staticmethod
    async def _publish_payload(
        topic: str,
        title: str,
        payload: dict[str, JsonValue],
        ctx: RuntimeContext,
    ) -> None:
        await ctx.publish(
            WEB_EVENT_TOPIC,
            WebEvent(topic=topic, title=title, payload=payload),
        )


__all__ = ["TeaWebEventsAgent"]
