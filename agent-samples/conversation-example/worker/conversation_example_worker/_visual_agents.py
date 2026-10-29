# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the direct current-view observation route agent.

Only the worker app, sample evaluator, and repository tests import this module.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from xr_ai_hub import FrameUnavailable
from xr_ai_interaction import ResponseOutput, ResponseOutputs
from xr_ai_routing import (
    Admission,
    AgentTurn,
    DestinationResult,
    RoutableAgent,
    RoutingAid,
)
from xr_ai_runtime import AgentRuntime, RuntimeClosedError, Topic
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.vision import ImageQueryRequest, StreamingImageQueryTool

_NO_CAMERA = "I couldn't get a current camera frame. Please try again."
_NO_VISUAL_EVIDENCE = "I couldn't inspect the current view right now. Please try again."


class VisualObservationAgent(RoutableAgent):
    """Answer visible-fact questions by sending the current image directly to Cosmos."""

    def __init__(
        self,
        frames: CurrentFrameTool,
        vision: StreamingImageQueryTool,
        *,
        runtime: AgentRuntime,
        output_topic: Topic[ResponseOutput],
        effects_started: Callable[[str, str], None] | None = None,
    ) -> None:
        self._outputs = ResponseOutputs(
            runtime, output_topic, source="observation", closed_errors=(RuntimeClosedError,)
        )
        self._effects_started = effects_started
        self._frames = frames
        self._vision = vision

    # RoutableAgent contract: answer uses one captured current frame per request.
    def routing_aid(self, _participant_id: str) -> RoutingAid:
        return RoutingAid(
            name="observation",
            description=(
                "Identify or describe visible objects, read visible text, and report observable colors, "
                "counts, or spatial relations from the current camera frame."
            ),
        )

    async def admit(self, _turn: AgentTurn, *, focused: bool) -> Admission:
        del focused
        return Admission("accept")

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[None]]) -> DestinationResult:
        try:
            async with self._outputs.reply(
                turn.participant_id, request_id=turn.turn_id, timestamp_us=turn.timestamp_us, started=started
            ) as reply:
                return await self._answer(turn, reply.send)
        except Exception:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError from None
            raise

    async def _answer(self, turn: AgentTurn, emit: Callable[[str], Awaitable[None]]) -> DestinationResult:
        if self._effects_started is not None:
            self._effects_started(turn.participant_id, turn.turn_id)
        try:
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=turn.participant_id))
        except FrameUnavailable:
            await emit(_NO_CAMERA)
            return DestinationResult("answer", _NO_CAMERA)

        query = turn.query
        if frame.timestamp_us is not None:
            query = f"Frame source time: {_frame_time_label(frame.timestamp_us)}.\n{query}"
        fragments: list[str] = []
        stream = self._vision.stream(ImageQueryRequest(image=frame.image, query=query))
        try:
            async for chunk in stream:
                if chunk.text:
                    fragments.append(chunk.text)
                    await emit(chunk.text)
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        answer = "".join(fragments).strip()
        if not answer:
            answer = _NO_VISUAL_EVIDENCE
            await emit(answer)
        return DestinationResult("answer", answer)

    async def release(self, _participant_id: str) -> None:
        """Keep shared frame state while another destination may use it."""

    async def participant_left(self, _participant_id: str) -> None:
        """The application releases shared frame state on participant departure."""


# Timestamp labels accompany the selected frame, not the decision clock.
def _frame_time_label(timestamp_us: int) -> str:
    timestamp = datetime.fromtimestamp(timestamp_us / 1_000_000, UTC).replace(second=0, microsecond=0)
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


__all__ = ["VisualObservationAgent"]
