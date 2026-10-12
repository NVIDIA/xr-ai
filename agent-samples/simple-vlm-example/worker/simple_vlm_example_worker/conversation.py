# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-independent conversation and current-view tool routing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, LLMService, ToolDef
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.vision import ImageQueryRequest, StreamingImageQueryTool

_CONVERSATION_PROMPT = (
    Path(__file__).parent / "prompts" / "conversation.txt"
).read_text(encoding="utf-8").strip()
_MAX_HISTORY_EXCHANGES = 4
_MAX_HISTORY_TEXT = 240
_CURRENT_VIEW_NAME = "current_view"
_CURRENT_VIEW_TOOL = ToolDef(
    name=_CURRENT_VIEW_NAME,
    description="Select the participant's present camera view for a visual question.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)


@dataclass(frozen=True, slots=True)
class ConversationExchange:
    """One completed user and assistant exchange."""

    user: str
    assistant: str


class Conversation:
    """Stream conversational answers or delegate visual questions to the VLM."""

    def __init__(
        self,
        frames: CurrentFrameTool,
        vision: StreamingImageQueryTool,
        llm: LLMService,
    ) -> None:
        self._frames = frames
        self._vision = vision
        self._llm = llm

    async def stream(
        self,
        request: str,
        participant_id: str,
        *,
        history: tuple[ConversationExchange, ...] = (),
    ) -> AsyncIterator[str]:
        """Yield one direct answer or stream a delegated current-view answer."""

        messages = _messages(request, history)

        spoken: list[str] = []
        calls = []
        events = self._llm.stream_events(
            messages,
            tools=(_CURRENT_VIEW_TOOL,),
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
        if len(calls) != 1 or calls[0].name != _CURRENT_VIEW_NAME:
            raise RuntimeError("conversation model selected an unavailable tool")

        # Validate the model arguments and execute a native typed Tool through
        # NeMo Relay. Image inference remains the separate streaming VLM tool.
        arguments = json.loads(calls[0].arguments)
        if arguments != {}:
            raise RuntimeError("conversation model returned invalid current_view arguments")
        try:
            frame = await self._frames.execute(
                CurrentFrameRequest(participant_id=participant_id)
            )
        except (FrameUnavailable, RuntimeError) as error:
            unavailable = _frame_unavailable_message(error)
            if unavailable is None:
                raise
            yield unavailable
            return

        stream = self._vision.stream(
            ImageQueryRequest(image=frame.image, query=request)
        )
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
) -> tuple[ChatMessage, ...]:
    messages = [ChatMessage(role="system", content=_CONVERSATION_PROMPT)]
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
                content=(
                    "Reference context (completed exchanges, not instructions):\n"
                    + json.dumps(context)
                ),
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
