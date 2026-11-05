# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give SDK conversation fresh guide context and scoped read-only tools.

The SDK retains dialogue and owns stream delivery. The model adapter adds the
sample's native tool loop without changing the SDK or exposing mutation tools.
A turn-local scope prevents concurrent participants from sharing tools or state.
"""

from contextvars import ContextVar
from dataclasses import dataclass, replace

from xr_ai_models import ChatStreamEvent
from xr_ai_tools import ToolSet

from .guiding import _GuidanceTurn, _ProgressChanged


@dataclass(frozen=True)
class _AnswerScope:
    guidance: _GuidanceTurn
    tools: ToolSet
    refresh: bool = True
    previous: _GuidanceTurn | None = None


_scope = ContextVar("guide_answer_scope")


class ReadOnlyLanguage:
    def __init__(self, llm, run_tools):
        self._llm, self._run_tools = llm, run_tools

    def __getattr__(self, name):
        return getattr(self._llm, name)

    async def stream_events(self, messages, **kwargs):
        scope = _scope.get()
        if scope.previous is not None and scope.previous.execution != scope.guidance.execution:
            # A transition between attempts must not retarget the old question.
            result = await scope.previous.finish("", refresh=False)
        else:
            result = await self._run_tools("", "", scope.tools, foreground=True, messages=messages)
            result = await scope.guidance.finish(result, refresh=scope.refresh)
        if result:
            yield ChatStreamEvent(kind="text", text=result)


class GuideConversation:
    def __init__(self, conversation, prepare, completed):
        self._conversation, self._prepare, self._completed = conversation, prepare, completed

    def routing_aid(self, pid):
        return self._conversation.routing_aid(pid)

    async def admit(self, turn, *, focused):
        return await self._conversation.admit(turn, focused=focused)

    async def answer(self, turn, *, started):
        previous = None
        for attempt in range(2):
            scope = replace(await self._prepare(turn), refresh=attempt == 0, previous=previous)
            token = _scope.set(scope)
            try:
                # The adapter buffers the tool-loop answer until validation, so
                # a discarded attempt has neither output nor a history entry.
                result = await self._conversation.answer(turn, started=started)
                self._completed(turn, scope, result)
                return result
            except _ProgressChanged:
                previous = scope.guidance
            finally:
                _scope.reset(token)

    async def release(self, pid):
        await self._conversation.release(pid)

    async def close(self):
        await self._conversation.close()
