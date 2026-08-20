# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded participant-local facts produced by background agents."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from xr_ai_runtime import Agent, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_voice import VoiceParticipantJoined, VoiceParticipantLeft

from .events import (
    BACKGROUND_FACT_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    BackgroundFact,
)

BackgroundApplication = Literal["change_watch", "transcript", "video_log"]


class BackgroundContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applications: tuple[BackgroundApplication, ...] = Field(
        default=(),
        description="Only needed background applications. Empty means all.",
    )
    max_items: int = Field(default=3, ge=1, le=10)
    max_age_s: float = Field(default=120, gt=0, le=86_400)


class BackgroundContextResult(BaseModel):
    facts: list[BackgroundFact] = Field(default_factory=list)


class BackgroundContextAgent(Agent):
    """Retain a bounded set of recent background facts for foreground turns."""

    def __init__(self, *, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._facts: dict[str, deque[BackgroundFact]] = {}
        self._lock = asyncio.Lock()
        self._closed: set[str] = set()
        super().__init__()

    def participant_tools(self, participant_id: str) -> ToolSet:
        """Return the foreground query tool bound to one participant."""

        async def query(
            request: BackgroundContextRequest,
        ) -> BackgroundContextResult:
            return await self.query(participant_id, request)

        return ToolSet(
            (
                Tool(
                    "application_context__query",
                    (
                        "Read recent facts produced by active background applications. "
                        "Use only when those facts help answer the current request."
                    ),
                    BackgroundContextRequest,
                    BackgroundContextResult,
                    query,
                ),
            )
        )

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        async with self._lock:
            self._closed.discard(participant_id)
            self._facts.pop(participant_id, None)

    @subscribe(BACKGROUND_FACT_TOPIC)
    async def record(self, fact: BackgroundFact, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        async with self._lock:
            if participant_id in self._closed:
                return
            self._facts.setdefault(
                participant_id,
                deque(maxlen=self._capacity),
            ).append(fact.model_copy(deep=True))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        async with self._lock:
            self._closed.add(participant_id)
            self._facts.pop(participant_id, None)

    async def query(
        self,
        participant_id: str,
        request: BackgroundContextRequest,
    ) -> BackgroundContextResult:
        cutoff_us = time.time_ns() // 1_000 - int(request.max_age_s * 1_000_000)
        applications = set(request.applications)
        async with self._lock:
            matches = [
                fact.model_copy(deep=True)
                for fact in reversed(self._facts.get(participant_id, ()))
                if fact.timestamp_us >= cutoff_us
                and (not applications or fact.application in applications)
            ][: request.max_items]
        matches.reverse()
        return BackgroundContextResult(facts=matches)

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("background context requires a participant")
        return participant_id


__all__ = [
    "BackgroundContextAgent",
    "BackgroundContextRequest",
    "BackgroundContextResult",
]
