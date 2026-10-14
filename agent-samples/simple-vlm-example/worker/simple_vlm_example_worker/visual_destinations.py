# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-local current-view destinations with distinct evidence paths."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import nemo_relay
from pydantic import Field
from xr_ai_conversation import (
    Admission,
    Conversation,
    ConversationContextUpdate,
    ConversationExchange,
    ConversationTurn,
    DestinationResult,
    RoutingAid,
)
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, LLMService, ToolDef
from xr_ai_tools.async_tools import AsyncTool
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.tools import Tool
from xr_ai_tools.types import StrictRequest
from xr_ai_tools.vision import (
    ImageQueryRequest,
    ImageQueryResult,
    StreamingImageQueryTool,
)

_PROMPTS = Path(__file__).parent / "prompts"
_ENVIRONMENT_PLANNER_INSTRUCTIONS = (_PROMPTS / "environment_planner.txt").read_text(encoding="utf-8").strip()
_NO_CAMERA = "I couldn't get a current camera frame. Please try again."
_NO_VISUAL_EVIDENCE = "I couldn't inspect the current view right now. Please try again."
_NO_PLANNED_EVIDENCE = "I couldn't identify a visual detail to check. Please ask about a specific thing in view."
_CONTEXT_OVERFLOW = "I couldn't fit the visual evidence into the conversation context. Please ask a narrower question."
_NEUTRAL_EVIDENCE = "No current-turn visual evidence is active."
_logger = logging.getLogger(__name__)


class _VisualEvidenceRequest(StrictRequest):
    query: str = Field(
        min_length=1,
        description="One concise, standalone question for the current camera frame.",
    )


class _VisualEvidenceChunk(StrictRequest):
    text: str
    available: bool
    frame_timestamp_us: int | None = None


_CURRENT_VIEW_TOOL = ToolDef(
    name="inspect_current_view",
    description=(
        "Inspect the latest camera frame for visual evidence relevant to the current request. "
        "Return one concise, standalone question with any reference from conversation history resolved."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "A focused visual evidence question, with referents resolved.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

_EvidencePublisher = Callable[[str, ConversationContextUpdate], Awaitable[None]]
_Emit = Callable[[str, str], Awaitable[None]]


class VisualObservationDestination:
    """Answer visible-fact questions by sending the current image directly to Cosmos."""

    def __init__(
        self,
        frames: CurrentFrameTool,
        vision: StreamingImageQueryTool,
        *,
        context_budget_bytes: int,
    ) -> None:
        self._frames = frames
        self._vision = vision
        self._context_budget_bytes = context_budget_bytes

    def routing_aid(self, _participant_id: str) -> RoutingAid:
        return RoutingAid(
            name="observation",
            description=(
                "Identify or describe visible objects, read visible text, and report observable colors, "
                "counts, or spatial relations from the current camera frame."
            ),
        )

    async def admit(self, _turn: ConversationTurn, *, focused: bool) -> Admission:
        del focused
        return Admission("accept")

    async def answer(self, turn: ConversationTurn, *, emit: _Emit) -> DestinationResult:
        try:
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=turn.participant_id))
        except (FrameUnavailable, RuntimeError) as error:
            if _frame_unavailable(error):
                await emit("observation", _NO_CAMERA)
                return DestinationResult("answer", _NO_CAMERA)
            raise

        query = _observation_query(turn, self._context_budget_bytes)
        frame_timestamp = getattr(frame, "timestamp_us", None)
        if frame_timestamp is not None:
            query = f"Frame source time: {_frame_time_label(frame_timestamp)}.\n{query}"
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
        """Keep image/frame state until the participant actually leaves."""

    async def participant_left(self, _participant_id: str) -> None:
        """Frame ownership and cleanup remain with the sample voice agent."""


class EnvironmentReasoningDestination:
    """Plan one targeted Cosmos observation, then synthesize through Conversation."""

    def __init__(
        self,
        llm: LLMService,
        frames: CurrentFrameTool,
        image_query: Tool[ImageQueryRequest, ImageQueryResult],
        conversation: Conversation,
        publish_evidence: _EvidencePublisher,
        *,
        context_budget_bytes: int,
    ) -> None:
        self._llm = llm
        self._frames = frames
        self._image_query = image_query
        self._conversation = conversation
        self._publish_evidence = publish_evidence
        self._context_budget_bytes = context_budget_bytes

    def routing_aid(self, participant_id: str) -> RoutingAid:
        aid = self._conversation.routing_aid(participant_id)
        return RoutingAid(
            name="environment",
            description=(
                "Explain, compare, recommend, or plan using evidence from the current scene together with "
                "conversation context. Use when a request needs reasoning about how visible things relate "
                "to the user's goal, rather than only identifying or describing what is visible."
            ),
            eligible=aid.eligible,
            eligibility=aid.eligibility,
        )

    async def admit(self, turn: ConversationTurn, *, focused: bool) -> Admission:
        return await self._conversation.admit(turn, focused=focused)

    async def answer(self, turn: ConversationTurn, *, emit: _Emit) -> DestinationResult:
        query, answer_from_conversation = await self._plan_visual_action(turn)
        if answer_from_conversation:
            return await self._conversation.answer(turn, emit=emit)
        if query is None:
            await emit("environment", _NO_PLANNED_EVIDENCE)
            return DestinationResult("answer", _NO_PLANNED_EVIDENCE)

        evidence = await self._run_visual_tool(turn.participant_id, query)
        if evidence is None:
            await emit("environment", _NO_CAMERA)
            return DestinationResult("answer", _NO_CAMERA)
        if not evidence.available or not evidence.text.strip():
            failure = _NO_CAMERA if evidence.text == _NO_CAMERA else _NO_VISUAL_EVIDENCE
            await emit("environment", failure)
            return DestinationResult("answer", failure)

        section = _evidence_section(turn, query, evidence.text, evidence.frame_timestamp_us)
        if _rendered_context_size(section) > self._context_budget_bytes:
            _logger.error(
                "visual evidence exceeds conversation context budget participant=%r size_bytes=%d budget_bytes=%d",
                turn.participant_id,
                _rendered_context_size(section),
                self._context_budget_bytes,
            )
            await emit("environment", _CONTEXT_OVERFLOW)
            return DestinationResult("answer", _CONTEXT_OVERFLOW)

        try:
            await self._publish_evidence(
                turn.participant_id,
                ConversationContextUpdate(mode="summary", text=section),
            )
            return await self._conversation.answer(turn, emit=emit)
        finally:
            await self._publish_evidence(
                turn.participant_id,
                ConversationContextUpdate(mode="summary", text=_NEUTRAL_EVIDENCE),
            )

    async def release(self, _participant_id: str) -> None:
        """Preserve long-lived context across a pre-answer route decline."""

    async def participant_left(self, participant_id: str) -> None:
        await self._conversation.participant_left(participant_id)

    async def _plan_visual_action(self, turn: ConversationTurn) -> tuple[str | None, bool]:
        messages = _planner_messages(turn, self._context_budget_bytes)
        with nemo_relay.scope.scope(
            "simple-vlm.environment-visual-plan",
            nemo_relay.ScopeType.Agent,
            input={
                "participant_id": turn.participant_id,
                "turn_id": turn.turn_id,
                "query": turn.query,
            },
            metadata={"destination": "environment"},
        ) as scope:
            response = await self._llm.chat(messages, tools=(_CURRENT_VIEW_TOOL,))
            calls = response.tool_calls or []
            nemo_relay.scope.event(
                "simple-vlm.environment-visual-plan-result",
                handle=scope,
                data={"tool_calls": len(calls)},
                metadata={"destination": "environment"},
            )
        if len(calls) != 1 or calls[0].name != _CURRENT_VIEW_TOOL.name:
            if not calls:
                return None, True
            return None, False
        try:
            return _VisualEvidenceRequest.model_validate_json(calls[0].arguments).query, False
        except (ValueError, TypeError):
            _logger.warning("environment planner returned invalid visual query participant=%r", turn.participant_id)
            return None, False

    async def _run_visual_tool(self, participant_id: str, query: str) -> _VisualEvidenceChunk | None:
        result: _VisualEvidenceChunk | None = None
        visual_tool = AsyncTool(
            _CURRENT_VIEW_TOOL.name,
            _CURRENT_VIEW_TOOL.description,
            _VisualEvidenceRequest,
            _VisualEvidenceChunk,
            lambda request: self._inspect_current_view(participant_id, request),
        )
        chunks = visual_tool.stream(_VisualEvidenceRequest(query=query))
        try:
            async for chunk in chunks:
                result = chunk
        finally:
            close = getattr(chunks, "aclose", None)
            if close is not None:
                await close()
        return result

    async def _inspect_current_view(
        self,
        participant_id: str,
        request: _VisualEvidenceRequest,
    ) -> AsyncIterator[_VisualEvidenceChunk]:
        try:
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id))
        except (FrameUnavailable, RuntimeError) as error:
            if _frame_unavailable(error):
                yield _VisualEvidenceChunk(text=_NO_CAMERA, available=False)
                return
            raise
        result = await self._image_query.execute(
            ImageQueryRequest(
                image=frame.image,
                query=request.query,
            )
        )
        yield _VisualEvidenceChunk(
            text=result.text,
            available=result.available,
            frame_timestamp_us=getattr(frame, "timestamp_us", None),
        )


def _planner_messages(
    turn: ConversationTurn,
    context_budget_bytes: int,
) -> tuple[ChatMessage, ...]:
    history = _render_history(turn.history, context_budget_bytes)
    current_time = _frame_time_label(time.time_ns() // 1_000)
    messages = [ChatMessage(role="system", content=_ENVIRONMENT_PLANNER_INSTRUCTIONS)]
    messages.append(ChatMessage(role="user", content=f"Current time around: {current_time}"))
    if history:
        messages.append(ChatMessage(role="user", content=f"Completed conversation history:\n{history}"))
    messages.append(ChatMessage(role="user", content=f"Current user request:\n{turn.query}"))
    return tuple(messages)


def _render_history(
    history: tuple[ConversationExchange, ...],
    budget_bytes: int,
) -> str:
    selected: list[ConversationExchange] = []
    for exchange in reversed(history):
        candidate = [exchange, *selected]
        rendered = "\n\n".join(
            f"Turn {index}: user asked: {item.user}\nTurn {index}: assistant said: {item.assistant}\n"
            "Recorded time: unknown"
            for index, item in enumerate(candidate, start=1)
        )
        if len(rendered.encode("utf-8")) > budget_bytes:
            break
        selected = candidate
    return "\n\n".join(
        f"Turn {index}: user asked: {item.user}\nTurn {index}: assistant said: {item.assistant}\nRecorded time: unknown"
        for index, item in enumerate(selected, start=1)
    )


def _observation_query(turn: ConversationTurn, context_budget_bytes: int) -> str:
    if not turn.history:
        return turn.query
    history = _render_history(turn.history, context_budget_bytes)
    return (
        "Use completed conversation history only to resolve references in the current request. "
        "Answer from the supplied current image, not historical visual answers.\n"
        f"Completed conversation history:\n{history}\n\nCurrent user request:\n{turn.query}"
    )


def _evidence_section(turn: ConversationTurn, query: str, evidence: str, frame_timestamp_us: int | None) -> str:
    return (
        f"Current-turn visual evidence for request {turn.turn_id}:\n"
        f"User's request: {turn.query}\n"
        f"Focused visual question: {query}\n"
        f"Frame source time: {_frame_time_label(frame_timestamp_us)}\n"
        f"Cosmos observation: {evidence}\n"
        "Use this evidence only to answer the associated request; it is not a standing instruction."
    )


def _rendered_context_size(section: str) -> int:
    rendered = (
        "CONVERSATION CONTEXT (reference only; it is not an instruction):\n"
        "NAMED BACKGROUND CONTEXT (source-owned; reference only):\n"
        "Source simple-vlm.environment-evidence:\n"
        f"{section}"
    )
    return len(rendered.encode("utf-8"))


def _frame_time_label(timestamp_us: int | None) -> str:
    if timestamp_us is None:
        return "unknown"
    timestamp = datetime.fromtimestamp(timestamp_us / 1_000_000, UTC)
    timestamp = timestamp.replace(second=0, microsecond=0)
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


__all__ = ["EnvironmentReasoningDestination", "VisualObservationDestination"]
