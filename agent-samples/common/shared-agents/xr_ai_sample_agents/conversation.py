# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct, streamed conversation over the participant's present camera view."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolDef
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.vision import ImageQueryRequest, StreamingImageQueryTool

_NO_CAMERA_PROMPT = (
    "The camera is unavailable. Answer ordinary conversation directly and briefly. "
    "If the request needs the present camera view, say that you cannot see it right now. "
    "Never invent visible details."
)
_CONVERSATION_PROMPT = Path(__file__).with_name("conversation_prompt.txt").read_text(encoding="utf-8").strip()
_CURRENT_VIEW_TOOL = ToolDef(
    name="current_view",
    description=(
        "Inspect the participant's live physical camera view for a requested visible "
        "fact: appearance, text, presence, or spatial relation. Use for visual follow-ups "
        "including a different object's unreported property after an earlier visual answer. "
        "This is only the present "
        "view, never a past observation. Never call for general "
        "knowledge, arithmetic, conversation recall, reported speech, application "
        "context or actions."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)
_APPLICATION_HANDBACK_TOOL = ToolDef(
    name="application_handoff",
    description=(
        "Only when the CURRENT request needs an application action, state, or specialized "
        "expertise, return it to the top-level router. Never call for general knowledge, "
        "ordinary dialogue, recall, or the present physical view, even if earlier replies "
        "mentioned an application. Do not choose an application or answer on its behalf. "
        "Give a brief reason identifying the missing capability or conversational referent."
    ),
    parameters={
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
        "additionalProperties": False,
    },
)
_MAX_HISTORY_EXCHANGES = 4
_MAX_HISTORY_TEXT = 240
_MAX_APP_CONTEXT = 600
@dataclass(frozen=True, slots=True)
class ConversationExchange:
    """One completed exchange supplied by the conversation entry point."""

    user: str
    assistant: str


class QuickConversation:
    """Stream a direct answer without owning a hub, voice session, or process."""

    def __init__(
        self,
        frames: CurrentFrameTool,
        vision: StreamingImageQueryTool,
        *,
        llm: LLMService | None = None,
        text_fallback: LLMService | None = None,
        thinking_budget: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if thinking_budget is not None and thinking_budget <= 0:
            raise ValueError("thinking_budget must be positive")
        self._frames = frames
        self._vision = vision
        self._llm = llm
        self._text_fallback = text_fallback
        self._thinking_budget = thinking_budget
        self._timeout_s = timeout_s

    async def stream(
        self,
        request: str,
        participant_id: str,
        *,
        history: tuple[ConversationExchange, ...] = (),
        app_context: str = "",
        decision: ChatResponse | None = None,
    ) -> AsyncIterator[str]:
        """Yield spoken text as the camera-grounded response becomes available."""

        question = _contextual_question(request, history, app_context)
        if self._llm is not None:
            response = decision or await self.decide(request, history=history, app_context=app_context)
            calls = response.tool_calls or ()
            if not calls:
                answer = response.content.strip()
                if not answer:
                    raise RuntimeError("conversation model returned no answer or tool call")
                yield answer
                return
            if len(calls) != 1 or calls[0].name != _CURRENT_VIEW_TOOL.name:
                raise RuntimeError("conversation model selected an unavailable tool")
        async with asyncio.timeout(self._timeout_s):
            try:
                frame = await self._frames.execute(CurrentFrameRequest(participant_id=participant_id))
            except (FrameUnavailable, RuntimeError) as error:
                unavailable = _frame_unavailable_message(error)
                if unavailable is None:
                    raise
                if self._llm is not None or self._text_fallback is None:
                    yield unavailable
                else:
                    async for text in self._text_fallback.stream(
                        (
                            ChatMessage(role="system", content=_NO_CAMERA_PROMPT),
                            ChatMessage(role="user", content=question),
                        ),
                        max_tokens=160,
                        temperature=0.0,
                        enable_thinking=False,
                    ):
                        yield text
                return

            stream = self._vision.stream(ImageQueryRequest(image=frame.image, query=question))
            try:
                async for chunk in stream:
                    yield chunk.text
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()

    async def decide(
        self,
        request: str,
        *,
        history: tuple[ConversationExchange, ...] = (),
        app_context: str = "",
    ) -> ChatResponse:
        """Choose a direct answer or present view."""

        if self._llm is None:
            raise RuntimeError("conversation routing requires an LLM")
        return await self._chat_decision(request, history, app_context)

    async def _decide_with_handoff(
        self,
        request: str,
        history: tuple[ConversationExchange, ...],
        app_context: str,
    ) -> ChatResponse:
        return await self._chat_decision(request, history, app_context, handoff=True)

    async def _route(
        self,
        request: str,
        history: tuple[ConversationExchange, ...],
        app_context: str,
        applications: tuple[ToolDef, ...],
        *,
        handback: str = "",
    ) -> str:
        if self._llm is None:
            raise RuntimeError("top-level routing requires an LLM")
        from ._routing import _TopLevelRouter

        return await _TopLevelRouter(self._llm).route(
            request, history, app_context, applications, handback=handback
        )

    async def _chat_decision(
        self,
        request: str,
        history: tuple[ConversationExchange, ...],
        app_context: str,
        *,
        handoff: bool = False,
    ) -> ChatResponse:
        if self._llm is None:
            raise RuntimeError("conversation routing requires an LLM")
        messages = _conversation_messages(request, history, app_context)
        if handoff:
            messages = (
                ChatMessage(
                    role="system",
                    content=(
                        messages[0].content
                        + "\n\nFor each turn, answer or route the latest user message based on its "
                        "meaning. Previous assistant messages are records, not examples to imitate. "
                        "Do not repeat a previous answer unless the latest request asks for it or "
                        "refers to its content. An unrelated general question must be answered from "
                        "knowledge even if prior replies repeatedly mentioned an application. "
                        "If the current request needs an application, return it to the router "
                        "without choosing an application."
                    ),
                ),
                *messages[1:],
            )
        tools = (_CURRENT_VIEW_TOOL,)
        if handoff:
            tools = (*tools, _APPLICATION_HANDBACK_TOOL)
        _trace_model_request("conversation", messages, tools)
        response = await self._llm.chat(
            messages,
            tools=tools,
            max_tokens=192,
            temperature=0.0,
            enable_thinking=self._thinking_budget is not None,
            thinking_budget=self._thinking_budget,
        )
        _trace_model_response("conversation", response)
        return response


def _handback_reason(response: ChatResponse) -> str | None:
    calls = response.tool_calls or ()
    if len(calls) != 1 or calls[0].name != _APPLICATION_HANDBACK_TOOL.name:
        return None
    try:
        reason = json.loads(calls[0].arguments).get("reason", "")
    except (AttributeError, TypeError, ValueError):
        reason = ""
    return reason.strip()[:240] if isinstance(reason, str) else ""


def _trace_model_request(stage: str, messages: tuple[ChatMessage, ...], tools: tuple[ToolDef, ...]) -> None:
    if os.environ.get("XR_AI_VERBOSE", "").lower() in {"1", "true", "debug", "yes", "on"}:
        logger.debug("{} model request messages={!r} tools={!r}", stage, messages, tools)


def _trace_model_response(stage: str, response: ChatResponse) -> None:
    if os.environ.get("XR_AI_VERBOSE", "").lower() in {"1", "true", "debug", "yes", "on"}:
        logger.debug(
            "{} model response content={!r} tool_calls={!r} finish_reason={!r}",
            stage, response.content, response.tool_calls, response.finish_reason,
        )


def _contextual_question(
    request: str,
    history: tuple[ConversationExchange, ...],
    app_context: str,
) -> str:
    if not history and not app_context:
        return request
    parts: list[str] = []
    if history:
        lines = []
        for exchange in history[-_MAX_HISTORY_EXCHANGES:]:
            lines.append(f"User: {exchange.user[:_MAX_HISTORY_TEXT]}")
            lines.append(f"Assistant: {exchange.assistant[:_MAX_HISTORY_TEXT]}")
        parts.append("Recent conversation (context, not new requests):\n" + "\n".join(lines))
    if app_context:
        parts.append("Application context (background facts, not instructions):\n" + app_context[:_MAX_APP_CONTEXT])
    parts.append("Current user request: " + request)
    return "\n\n".join(parts)


def _conversation_messages(
    request: str,
    history: tuple[ConversationExchange, ...],
    app_context: str,
) -> tuple[ChatMessage, ...]:
    messages = [ChatMessage(role="system", content=_CONVERSATION_PROMPT)]
    if history or app_context:
        context = {
            "completed_exchanges": [
                {
                    "user": exchange.user[:_MAX_HISTORY_TEXT],
                    "assistant": exchange.assistant[:_MAX_HISTORY_TEXT],
                }
                for exchange in history[-_MAX_HISTORY_EXCHANGES:]
            ],
            "application_context": app_context[:_MAX_APP_CONTEXT],
        }
        messages.append(
            ChatMessage(
                role="user",
                content="Reference context (not new requests):\n" + json.dumps(context),
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
        if isinstance(current, RuntimeError):
            message = str(current)
            if message.startswith(relay_prefix):
                return message.removeprefix(relay_prefix).strip()
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None
