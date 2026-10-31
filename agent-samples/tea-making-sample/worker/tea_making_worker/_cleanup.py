# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acknowledge participant cleanup after delegating it to the SDK router.

This is the sole SOP participant-left adapter so a delayed duplicate leave cannot
clean up a newly joined participant. The router cancels/drains work and invokes
its cleanup hook; only then does this bridge acknowledge the original message
generation to the sample's file/image cleanup protocol.
"""

from xr_ai_interaction import ParticipantLeft
from xr_ai_runtime import Agent, RuntimeContext, subscribe

from .events import PARTICIPANT_CLEANUP_COMPLETE_TOPIC, PARTICIPANT_LEFT_TOPIC, ParticipantCleanupComplete


class _CleanupAcks(Agent):
    """Bridge SDK participant cleanup to the sample's correlated cleanup acknowledgments."""

    def __init__(self, router):
        """Retain the public router lifecycle API, not a second task registry."""
        super().__init__()
        self.router = router

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def acknowledge(self, _event: ParticipantLeft, ctx: RuntimeContext) -> None:
        """Await SDK departure cleanup before acknowledging the original message generation."""
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
