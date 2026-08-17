# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared participant-scoped live-image acquisition."""

from __future__ import annotations

from xr_ai_hub import ProcessorEndpoint
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.marker_tracking import MarkerTrackingTool
from xr_ai_voice import VoiceParticipantLeft

from .events import PARTICIPANT_LEFT_TOPIC


class ParticipantImageAgent(Agent):
    """Own live-frame acquisition, image handles, and participant cleanup."""

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
        self.track_markers = MarkerTrackingTool(
            endpoint=endpoint,
            frame_max_age_s=frame_max_age_s,
            frame_timeout_s=frame_timeout_s,
            manage_status=False,
        )
        super().__init__((self.get_current_frame, self.track_markers))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is not None:
            self.get_current_frame.release(participant_id)
            self.track_markers.release(participant_id)

    async def stop(self) -> None:
        """Release every retained image when the application stops."""

        self.images.clear()


__all__ = ["ParticipantImageAgent"]
