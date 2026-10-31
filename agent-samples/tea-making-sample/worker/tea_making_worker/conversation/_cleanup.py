# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acknowledge SOP producers only after SDK-owned departure has drained them."""

from xr_ai_interaction import ParticipantLeft
from xr_ai_runtime import Agent, RuntimeContext, subscribe

from ..events import PARTICIPANT_CLEANUP_COMPLETE_TOPIC, PARTICIPANT_LEFT_TOPIC, ParticipantCleanupComplete


class _CleanupAcks(Agent):
    def __init__(self, router):
        super().__init__()
        self.router = router

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def acknowledge(self, _event: ParticipantLeft, ctx: RuntimeContext) -> None:
        participant = ctx.metadata.participant_id
        if participant is None:
            return
        # This is the sole left-event adapter so acknowledgment generation stays
        # with the original event. The router owns admission, draining and hooks.
        await self.router.participant_left(participant)
        for producer in ("foreground", "guidance"):
            await ctx.publish(PARTICIPANT_CLEANUP_COMPLETE_TOPIC, ParticipantCleanupComplete(
                generation=ctx.metadata.message_id, producer=producer,
            ))
