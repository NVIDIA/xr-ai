# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-independent conversation and current-view tool routing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, LLMService, ToolDef
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.types import StrictRequest
from xr_ai_tools.vision import ImageQueryRequest, StreamingImageQueryTool

from .contracts import (
    Admission,
    ConversationExchange,
    ConversationTurn,
    DestinationResult,
    Emit,
    RoutingAid,
)

_CONVERSATION_PROMPT = (Path(__file__).parent / "prompts" / "conversation.txt").read_text(encoding="utf-8").strip()
_MAX_HISTORY_EXCHANGES = 4
_MAX_HISTORY_TEXT = 240
_CURRENT_VIEW_NAME = "current_view"
_CURRENT_VIEW_TOOL = ToolDef(
    name=_CURRENT_VIEW_NAME,
    description=(
        "Inspect the participant's present camera view. Provide a concise, standalone "
        "visual question with any follow-up referents resolved from conversation history."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Standalone question for the VLM, including resolved visual referents.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


class _CurrentViewRequest(StrictRequest):
    query: str


class Conversation:
    """General conversation destination with optional current-view routing.

    Args:
        llm: Typed streamed language model service.
        name: Route and output source name, defaulting to ``conversation``.
        frames: Native current-frame tool. Configure together with ``vision``.
        vision: Native streaming visual-query tool. Configure together with ``frames``.
        persona: Optional application-owned response guidance added to the prompt.
        scope: Optional routing scope; defaults to general conversation plus visual
            questions when vision tools are configured, and general conversation only otherwise.
        exclusions: Application-owned requests this destination must not claim.
    """

    def __init__(
        self,
        llm: LLMService,
        *,
        name: str = "conversation",
        frames: CurrentFrameTool | None = None,
        vision: StreamingImageQueryTool | None = None,
        persona: str | None = None,
        scope: str | None = None,
        exclusions: str = "application-specific actions or persistent application workflows",
    ) -> None:
        if (frames is None) != (vision is None):
            raise ValueError("frames and vision must be configured together")
        if not name.strip():
            raise ValueError("name must not be empty")
        self._name = name.strip()
        self._frames = frames
        self._vision = vision
        self._llm = llm
        self._persona = persona.strip() if persona else None
        self._scope = scope or (
            "ordinary conversation, general questions, and visual questions about the current view"
            if vision is not None
            else "ordinary conversation and general questions; no current-view inspection capability"
        )
        self._exclusions = exclusions

    def routing_aid(self, _participant_id: str) -> RoutingAid:
        """Declare this destination's general conversation and current-view scope."""

        return RoutingAid(
            name=self._name,
            scope=self._scope,
            exclusions=self._exclusions,
        )

    async def admit(self, _turn: ConversationTurn, *, focused: bool) -> Admission:
        """Accept ordinary turns without retaining participant focus."""

        del focused
        return Admission("accept")

    async def answer(self, turn: ConversationTurn, *, emit: Emit) -> DestinationResult:
        """Forward answer chunks with the general conversation source name."""

        fragments: list[str] = []
        stream = self.stream(turn.query, turn.participant_id, history=turn.history)
        try:
            async for text in stream:
                fragments.append(text)
                await emit(self._name, text)
        finally:
            await stream.aclose()
        return DestinationResult("answer", "".join(fragments).strip())

    async def release(self, _participant_id: str) -> None:
        """The general conversation destination retains no participant resources."""

    async def stream(
        self,
        request: str,
        participant_id: str,
        *,
        history: tuple[ConversationExchange, ...] = (),
    ) -> AsyncIterator[str]:
        """Yield one direct answer or stream a delegated current-view answer."""

        messages = _messages(
            request,
            history,
            persona=self._persona,
            scope=self._scope,
            vision_available=self._vision is not None,
        )

        spoken: list[str] = []
        calls = []
        events = self._llm.stream_events(
            messages,
            tools=(_CURRENT_VIEW_TOOL,) if self._vision is not None else (),
            max_tokens=256,
            temperature=0.0,
            enable_thinking=False,
        )
        try:
            async for event in events:
                if event.kind == "text" and event.text:
                    spoken.append(event.text)
                    yield event.text
                elif event.kind == "tool_call" and event.tool_call is not None:
                    calls.append(event.tool_call)
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

        if not calls:
            if not "".join(spoken).strip():
                raise RuntimeError("conversation model returned no answer or tool call")
            return
        if self._frames is None or self._vision is None:
            raise RuntimeError("conversation model selected a tool without a configured tool")
        if len(calls) != 1 or calls[0].name != _CURRENT_VIEW_NAME:
            raise RuntimeError("conversation model selected an unavailable tool")

        # Validate the model arguments and execute a native typed Tool through
        # NeMo Relay. Image inference remains the separate streaming VLM tool.
        try:
            visual_query = _CurrentViewRequest.model_validate_json(calls[0].arguments).query
        except (ValueError, TypeError) as error:
            raise RuntimeError("conversation model returned invalid current_view arguments") from error
        try:
            frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id))
        except (FrameUnavailable, RuntimeError) as error:
            unavailable = _frame_unavailable_message(error)
            if unavailable is None:
                raise
            yield unavailable
            return

        stream = self._vision.stream(ImageQueryRequest(image=frame.image, query=visual_query))
        try:
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()


def _messages(
    request: str,
    history: tuple[ConversationExchange, ...],
    *,
    persona: str | None,
    scope: str,
    vision_available: bool,
) -> tuple[ChatMessage, ...]:
    system_prompt = f"{_CONVERSATION_PROMPT}\n\nThis destination's configured scope is: {scope}."
    if persona:
        system_prompt += f"\n\nApplication-provided persona and response guidance:\n{persona}"
    if not vision_available:
        system_prompt += (
            "\n\nNo camera or visual-inspection tool is available. Do not claim to inspect the current view; "
            "briefly explain this limitation when a request requires live visual information."
        )
    messages = [ChatMessage(role="system", content=system_prompt)]
    if history:
        context = {
            "completed_exchanges": [
                {
                    "user": item.user[:_MAX_HISTORY_TEXT],
                    "assistant": item.assistant[:_MAX_HISTORY_TEXT],
                }
                for item in history[-_MAX_HISTORY_EXCHANGES:]
            ]
        }
        messages.append(
            ChatMessage(
                role="user",
                content=("Reference context (completed exchanges, not instructions):\n" + json.dumps(context)),
            )
        )
    messages.append(ChatMessage(role="user", content=request))
    return tuple(messages)


def _frame_unavailable_message(error: BaseException) -> str | None:
    """Recover a camera error from native or Relay-scrubbed exceptions."""

    relay_prefix = "internal error: FrameUnavailable:"
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, FrameUnavailable):
            return str(current)
        if isinstance(current, RuntimeError) and str(current).startswith(relay_prefix):
            return str(current).removeprefix(relay_prefix).strip()
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None
