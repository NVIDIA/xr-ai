# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped live-image acquisition shared by visual agents."""

from __future__ import annotations

from xr_ai_hub import ProcessorEndpoint
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_voice import VoiceParticipantLeft

from .events import PARTICIPANT_LEFT_TOPIC


class ParticipantImageAgent(Agent):
    """Own current-frame acquisition, image handles, and participant cleanup."""

    def __init__(
        self,
        *,
        endpoint: ProcessorEndpoint,
        frame_max_age_s: float,
        frame_timeout_s: float,
    ) -> None:
        self.images = ImageRegistry()
        self.get_current_frame = CurrentFrameTool(
            endpoint=endpoint,
            images=self.images,
            frame_max_age_s=frame_max_age_s,
            frame_timeout_s=frame_timeout_s,
        )
        super().__init__((self.get_current_frame,))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is not None:
            self.get_current_frame.release(participant_id)

    async def stop(self) -> None:
        """Release every retained image when the application stops."""

        self.images.clear()


__all__ = ["ParticipantImageAgent"]
