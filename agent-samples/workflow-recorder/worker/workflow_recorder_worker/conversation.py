# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared conversation with deterministic SOP controls and silent narration."""

from __future__ import annotations

import asyncio

from xr_ai_runtime import RuntimeContext, subscribe
from xr_ai_sample_agents.front_end import (
    FRONT_END_QUERY_TOPIC,
    ConversationApplication,
    ConversationFrontEnd,
    QuickConversation,
)
from xr_ai_voice import UserQuery, VoiceAggregationAgent, VoiceInterrupted, VoiceParticipantLeft

from ._workflow_engine import SopEngineAgent
from .events import INTERRUPTED_TOPIC, PARTICIPANT_LEFT_TOPIC
from .recorder import RecorderAgent


class SopConversationFrontEnd(ConversationFrontEnd):
    """Keep recording controls outside cancellable model-routing turns."""

    def __init__(
        self, conversation: QuickConversation, *, application: ConversationApplication,
        engine: SopEngineAgent, recorder: RecorderAgent, aggregation: VoiceAggregationAgent,
    ) -> None:
        super().__init__(conversation, applications=(application,))
        self._engine = engine
        self._recorder = recorder
        self._aggregation = aggregation
        self._input_locks: dict[str, asyncio.Lock] = {}

    @subscribe(FRONT_END_QUERY_TOPIC)
    async def receive(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("conversation turns require a participant")
        async with self._input_locks.setdefault(participant_id, asyncio.Lock()):
            if not self._engine.is_connected(participant_id):
                return
            control = self._engine.is_control(query.text)
            if self._recorder.is_recording(participant_id) and not control:
                # Main's capture service already owns this narration. It must
                # not become a conversation turn or an unsolicited answer.
                return
            await self.interrupted(participant_id)
            await self._engine.interrupted(participant_id)
            await self._aggregation.release(participant_id)
            # Departure can run while any of the cleanup awaits are suspended.
            if not self._engine.is_connected(participant_id):
                return
            if control:
                # Finalization cannot be cancelled by the next routed query.
                await self._engine.user_query(query, ctx)
            else:
                await super().receive(query, ctx)

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def leave(self, _event: VoiceParticipantLeft, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is not None:
            # Keep the same lock across reconnects and queued input deliveries.
            async with self._input_locks.setdefault(participant_id, asyncio.Lock()):
                await self.participant_left(participant_id)
                await self._aggregation.release(participant_id)

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupt(self, _event: VoiceInterrupted, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        await self.interrupted(participant_id)
        await self._engine.interrupted(participant_id)
        if participant_id is None:
            await self._aggregation.release_all()
        else:
            await self._aggregation.release(participant_id)

    async def stop(self) -> None:
        await super().stop()
        self._input_locks.clear()
