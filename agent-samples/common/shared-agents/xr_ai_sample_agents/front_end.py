# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped conversation entry point for sample applications."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import nemo_relay
from loguru import logger
from xr_ai_models import ToolDef
from xr_ai_runtime import Agent, RuntimeContext, Topic, subscribe
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceOutput,
)

from .conversation import ConversationExchange, QuickConversation

FRONT_END_QUERY_TOPIC: Topic[UserQuery] = Topic("sample-conversation.user-query", UserQuery)
_MAX_HISTORY = 4


@dataclass(frozen=True, slots=True)
class ConversationApplication:
    """An app-owned top-level route and its read-only participant projection."""

    name: str
    description: str
    query_topic: Topic[UserQuery]
    has_focus: Callable[[str], bool]
    context: Callable[[str], str]

    def tool(self) -> ToolDef:
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )


class ConversationFrontEnd(Agent):
    """Route once per idle turn; send focused turns straight to their app."""

    def __init__(
        self,
        conversation: QuickConversation,
        applications: tuple[ConversationApplication, ...] = (),
    ) -> None:
        super().__init__()
        if len({app.name for app in applications}) != len(applications):
            raise ValueError("application route names must be unique")
        if any(app.name == "current_view" for app in applications):
            raise ValueError("current_view is reserved for generic conversation")
        self._conversation = conversation
        self._applications = applications
        self._route_tools = tuple(app.tool() for app in applications)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._history: dict[str, deque[ConversationExchange]] = {}
        self._pending: dict[tuple[str, str], str] = {}
        self._spoken: dict[tuple[str, str], list[str]] = {}

    @subscribe(FRONT_END_QUERY_TOPIC)
    async def receive(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("conversation turns require a participant")
        await self._cancel(participant_id)
        for key in tuple(self._pending):
            if key[0] == participant_id:
                self._pending.pop(key, None)
                self._spoken.pop(key, None)
        turn_id = ctx.metadata.correlation_id
        self._pending[(participant_id, turn_id)] = query.text
        task = asyncio.create_task(
            self._run(query, ctx, participant_id, turn_id),
            name=f"conversation-front-end:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(lambda done, pid=participant_id: self._discard(pid, done))

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def observe_speech(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        """Remember final delivered results, excluding acknowledgements."""

        participant_id = ctx.metadata.participant_id
        if participant_id is None or output.turn_id is None or output.kind != "result":
            return
        key = (participant_id, output.turn_id)
        if key not in self._pending:
            return
        if output.text:
            self._spoken.setdefault(key, []).append(output.text)
        if not output.final:
            return
        answer = "".join(self._spoken.pop(key, ())).strip()
        request = self._pending.pop(key)
        if answer:
            self._history.setdefault(participant_id, deque(maxlen=_MAX_HISTORY)).append(
                ConversationExchange(user=request, assistant=answer)
            )

    async def participant_left(self, participant_id: str) -> None:
        """Release one participant's in-flight turn and conversation memory."""

        await self._cancel(participant_id)
        self._history.pop(participant_id, None)
        for key in tuple(self._pending):
            if key[0] == participant_id:
                self._pending.pop(key, None)
                self._spoken.pop(key, None)

    async def interrupted(self, participant_id: str | None) -> None:
        """Cancel turns without releasing application focus or completed history."""

        if participant_id is None:
            for pid in tuple(self._tasks):
                await self._cancel(pid)
            self._pending.clear()
            self._spoken.clear()
        else:
            await self._cancel(participant_id)
            for key in tuple(self._pending):
                if key[0] == participant_id:
                    self._pending.pop(key, None)
                    self._spoken.pop(key, None)

    async def stop(self) -> None:
        await self.interrupted(None)

    async def _run(
        self,
        query: UserQuery,
        ctx: RuntimeContext,
        participant_id: str,
        turn_id: str,
    ) -> None:
        try:
            focused = [app for app in self._applications if app.has_focus(participant_id)]
            if len(focused) > 1:
                raise RuntimeError("multiple applications hold participant focus")
            if focused:
                await ctx.publish(focused[0].query_topic, query)
                return
            app_context = "\n".join(text for app in self._applications if (text := app.context(participant_id)))
            history = tuple(self._history.get(participant_id, ()))
            decision = await self._conversation.decide(
                query.text,
                history=history,
                app_context=app_context,
                applications=self._route_tools,
            )
            calls = decision.tool_calls or ()
            if len(calls) == 1:
                for app in self._applications:
                    if calls[0].name == app.name:
                        await ctx.publish(app.query_topic, query)
                        return
            streaming = len(calls) == 1 and calls[0].name == "current_view"
            fragments = 0
            async for chunk in self._conversation.stream(
                query.text,
                participant_id,
                history=history,
                app_context=app_context,
                decision=decision,
            ):
                if not chunk:
                    continue
                fragments += 1
                await ctx.publish(
                    VOICE_CONTRIBUTION_TOPIC,
                    VoiceOutput(
                        text=chunk,
                        response_id=turn_id if streaming else None,
                        final=not streaming,
                        interrupt=streaming and fragments == 1,
                        timestamp_us=query.timestamp_us,
                        kind="result",
                        turn_id=turn_id,
                    ),
                )
            if streaming and fragments:
                await ctx.publish(
                    VOICE_CONTRIBUTION_TOPIC,
                    VoiceOutput(
                        response_id=turn_id,
                        timestamp_us=query.timestamp_us,
                        kind="result",
                        turn_id=turn_id,
                    ),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).error("conversation front end failed pid={!r}", participant_id)
            await ctx.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(
                    text="I couldn't complete that request. Please try again.",
                    timestamp_us=query.timestamp_us,
                    kind="result",
                    turn_id=turn_id,
                ),
            )

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("conversation turn failed pid={!r}: {!r}", participant_id, error)


__all__ = ["FRONT_END_QUERY_TOPIC", "ConversationApplication", "ConversationFrontEnd"]
