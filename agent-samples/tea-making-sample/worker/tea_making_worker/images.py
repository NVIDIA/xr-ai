# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped live-image acquisition shared by visual agents."""

from __future__ import annotations

from xr_ai_hub import ProcessorEndpoint
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry

from .events import (
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    ParticipantCleanupComplete,
)

_CLEANUP_PRODUCERS = frozenset(
    {"guidance", "foreground", "change_watch", "transcript", "video_log"}
)


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
        self._cleanup: dict[str, set[str]] = {}
        super().__init__((self.get_current_frame,))

    @subscribe(PARTICIPANT_CLEANUP_COMPLETE_TOPIC)
    async def participant_cleanup_complete(
        self,
        event: ParticipantCleanupComplete,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            return
        completed = self._cleanup.setdefault(participant_id, set())
        completed.add(event.producer)
        if _CLEANUP_PRODUCERS <= completed:
            self._cleanup.pop(participant_id, None)
            self.get_current_frame.release(participant_id)

    async def stop(self) -> None:
        """Release every retained image when the application stops."""

        self._cleanup.clear()
        self.images.clear()


__all__ = ["ParticipantImageAgent"]
