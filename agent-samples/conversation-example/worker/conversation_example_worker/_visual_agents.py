# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the direct current-view observation route agent.

Only the worker app, sample evaluator, and repository tests import this module.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime

from xr_ai_conversation import ConversationContextUpdate
from xr_ai_hub import FrameUnavailable
from xr_ai_interaction import ResponseOutput
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
ContextPublisher = Callable[[str, ConversationContextUpdate], Awaitable[None]]
OBSERVATION_CONTEXT_TOPIC = Topic("conversation-example.observation-capability", ConversationContextUpdate)
_OBSERVATION_CAPABILITY = (
    "Capability: answer questions about objects, visible text, colors, counts, and spatial relations by "
    "requesting a fresh camera frame for each visual question. This route does not monitor continuously "
    "or control applications."
)


class VisualObservationAgent(RoutableAgent):
    """Answer visible-fact questions by sending the current image directly to Cosmos."""

    def __init__(
        self,
        frames: CurrentFrameTool,
        vision: StreamingImageQueryTool,
        publish_context: ContextPublisher | None = None,
        *,
        runtime: AgentRuntime,
        output_topic: Topic[ResponseOutput],
    ) -> None:
        self._runtime = runtime
        self._output_topic = output_topic
        self._frames = frames
        self._vision = vision
        self._publish_context = publish_context

    async def participant_joined(self, participant_id: str) -> None:
        """Publish this route's capability without claiming the camera is ready."""

        if self._publish_context is not None:
            await self._publish_context(
                participant_id,
                ConversationContextUpdate(text=_OBSERVATION_CAPABILITY),
            )

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

    async def answer(self, turn: AgentTurn, *, started: Callable[[], Awaitable[bool]]) -> DestinationResult:
        opened = False
        publication_failed = False
        interrupt = False

        async def emit(text: str) -> None:
            nonlocal opened, publication_failed, interrupt
            if not opened:
                interrupt = await started()
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError
            try:
                await self._runtime.publish(
                    self._output_topic,
                    ResponseOutput(
                        text=text,
                        response_id=turn.turn_id,
                        final=False,
                        interrupt=interrupt and not opened,
                        timestamp_us=turn.timestamp_us,
                    ),
                    participant_id=turn.participant_id,
                    source="observation",
                )
            except Exception:
                publication_failed = True
                raise
            opened = True

        try:
            return await self._answer(turn, emit)
        except Exception:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError from None
            raise
        finally:
            # Close an opened stream even when canceled so later output does not queue behind it.
            if opened and not publication_failed:
                with suppress(RuntimeClosedError):
                    await self._runtime.publish(
                        self._output_topic,
                        ResponseOutput(
                            response_id=turn.turn_id,
                            timestamp_us=turn.timestamp_us,
                        ),
                        participant_id=turn.participant_id,
                        source="observation",
                    )

    async def _answer(self, turn: AgentTurn, emit: Callable[[str], Awaitable[None]]) -> DestinationResult:
        try:
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=turn.participant_id))
        except (FrameUnavailable, RuntimeError) as error:
            if _frame_unavailable(error):
                await emit(_NO_CAMERA)
                return DestinationResult("answer", _NO_CAMERA)
            raise

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


# Private timestamp and unavailable-frame classifiers support the public route above.
def _frame_time_label(timestamp_us: int) -> str:
    timestamp = datetime.fromtimestamp(timestamp_us / 1_000_000, UTC).replace(second=0, microsecond=0)
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def _frame_unavailable(error: BaseException) -> bool:
    relay_prefix = "internal error: FrameUnavailable:"
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, FrameUnavailable):
            return True
        if isinstance(current, RuntimeError) and str(current).startswith(relay_prefix):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


__all__ = ["VisualObservationAgent"]
