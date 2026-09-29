# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private first-tier routing for sample conversation entry points."""

from __future__ import annotations

from xr_ai_models import ChatMessage, LLMService, ToolDef

from .conversation import (
    _MAX_APP_CONTEXT,
    _MAX_HISTORY_EXCHANGES,
    _MAX_HISTORY_TEXT,
    ConversationExchange,
    _trace_model_request,
    _trace_model_response,
)

_CONVERSATION_ROUTE = ToolDef(
    name="conversation",
    description=(
        "General conversation, knowledge, dialogue recall, and the participant's "
        "present physical camera view. This destination can report that a request "
        "needs an application, but does not choose which application."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)
_ROUTER_PROMPT = (
    "Select exactly one destination; do not answer the user. conversation handles "
    "chat, facts, memory of spoken answers, and the live first-person physical camera. "
    "Applications handle only their own capabilities, state, and recorded observations. "
    "The participant asking what you see, what this is, or to read visible text "
    "needs conversation unless they explicitly refer to application state or past evidence. "
    "The current user request controls the choice; completed turns only clarify pronouns."
)


class _TopLevelRouter:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def route(
        self,
        request: str,
        history: tuple[ConversationExchange, ...],
        app_context: str,
        applications: tuple[ToolDef, ...],
        *,
        handback: str = "",
    ) -> str:
        lines = ["Completed conversation (reference only):"]
        for turn in history[-_MAX_HISTORY_EXCHANGES:]:
            lines.append(f"User: {turn.user[:_MAX_HISTORY_TEXT]}")
            lines.append(f"Assistant replied: {turn.assistant[:_MAX_HISTORY_TEXT]}")
        lines.extend(
            (
                f"Application state (background facts): {app_context[:_MAX_APP_CONTEXT]}",
                f"Current user request: {request}",
            )
        )
        if handback:
            lines.append(f"Conversation agent handback (not an answer): {handback[:240]}")
        messages = (
            ChatMessage(role="system", content=_ROUTER_PROMPT),
            ChatMessage(role="user", content="\n\n".join(lines)),
        )
        tools = (_CONVERSATION_ROUTE, *applications)
        _trace_model_request("top-level route", messages, tools)
        response = await self._llm.chat(
            messages,
            tools=tools,
            max_tokens=96,
            temperature=0.0,
            enable_thinking=False,
        )
        _trace_model_response("top-level route", response)
        calls = response.tool_calls or ()
        destinations = {_CONVERSATION_ROUTE.name, *(app.name for app in applications)}
        if len(calls) == 1 and calls[0].name in destinations:
            return calls[0].name
        if not calls:
            return _CONVERSATION_ROUTE.name
        raise RuntimeError("top-level router selected an unavailable destination")
