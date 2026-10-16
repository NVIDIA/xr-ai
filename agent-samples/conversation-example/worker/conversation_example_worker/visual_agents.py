# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-local direct current-view observation agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from xr_ai_hub import FrameUnavailable
from xr_ai_routing import (
    Admission,
    AgentTurn,
    DestinationResult,
    RoutableAgent,
    RoutingAid,
)
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.vision import ImageQueryRequest, StreamingImageQueryTool

_NO_CAMERA = "I couldn't get a current camera frame. Please try again."
_NO_VISUAL_EVIDENCE = "I couldn't inspect the current view right now. Please try again."
Emit = Callable[[str, str], Awaitable[None]]


class VisualObservationAgent(RoutableAgent):
    """Answer visible-fact questions by sending the current image directly to Cosmos."""

    def __init__(
        self,
        frames: CurrentFrameTool,
        vision: StreamingImageQueryTool,
    ) -> None:
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

    async def answer(self, turn: AgentTurn, *, emit: Emit) -> DestinationResult:
        try:
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=turn.participant_id))
        except (FrameUnavailable, RuntimeError) as error:
            if _frame_unavailable(error):
                await emit("observation", _NO_CAMERA)
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
                    await emit("observation", chunk.text)
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        answer = "".join(fragments).strip()
        if not answer:
            answer = _NO_VISUAL_EVIDENCE
            await emit("observation", answer)
        return DestinationResult("answer", answer)

    async def release(self, _participant_id: str) -> None:
        """Keep shared frame state while another destination may use it."""

    async def participant_left(self, _participant_id: str) -> None:
        """The sample adapter releases shared frame state on participant departure."""


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
