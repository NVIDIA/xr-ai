# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped live-image acquisition shared by visual agents."""

from __future__ import annotations

from xr_ai_hub import ProcessorEndpoint
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_voice import VoiceParticipantJoined, VoiceParticipantLeft

from .events import (
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
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
        self._cleanup: dict[str, dict[str, set[str]]] = {}
        self._leaving_generation: dict[str, str] = {}
        super().__init__((self.get_current_frame,))

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            return
        self._cleanup[participant_id] = {}
        self._leaving_generation.pop(participant_id, None)

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            return
        generation = ctx.metadata.message_id
        self._leaving_generation[participant_id] = generation
        completed = self._cleanup.setdefault(participant_id, {}).get(
            generation, set()
        )
        if _CLEANUP_PRODUCERS <= completed:
            self._release(participant_id)

    @subscribe(PARTICIPANT_CLEANUP_COMPLETE_TOPIC)
    async def participant_cleanup_complete(
        self,
        event: ParticipantCleanupComplete,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            return
        generations = self._cleanup.setdefault(participant_id, {})
        completed = generations.setdefault(event.generation, set())
        completed.add(event.producer)
        if (
            self._leaving_generation.get(participant_id) == event.generation
            and _CLEANUP_PRODUCERS <= completed
        ):
            self._release(participant_id)

    def _release(self, participant_id: str) -> None:
        self._cleanup.pop(participant_id, None)
        self._leaving_generation.pop(participant_id, None)
        self.get_current_frame.release(participant_id)

    async def stop(self) -> None:
        """Release every retained image when the application stops."""

        self._cleanup.clear()
        self._leaving_generation.clear()
        self.images.clear()


__all__ = ["ParticipantImageAgent"]
