# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guide participants through an approved procedure and verify their progress.

ReplayAgent handles speech and connection events and exposes conversational
guide controls. The replay backend owns verification and progress so model
responses cannot bypass the active step's completion rules.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolDef
from xr_ai_runtime import Agent, AgentRuntime, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import ToolSet
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.tool_calling import run_tool_loop
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    UserQuery,
    VoiceAggregationAgent,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)

from .events import INTERRUPTED_TOPIC, PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC, USER_QUERY_TOPIC
from .guides import SelectedGuide
from .replaying import _cancel, _discard, _Replay

_MAX_TOOL_ROUNDS = 3
_FOREGROUND_MODEL_OPTIONS = {"max_tokens": 1536, "enable_thinking": True, "thinking_budget": 1024}
_FOREGROUND_PROMPT = (Path(__file__).with_name("prompts") / "replay_conversation.txt").read_text(
    encoding="utf-8",
).strip()


class ReplayAgent(Agent):
    """Orchestrate speech and connection events for an approved replay guide."""

    def __init__(
        self,
        *,
        guide: SelectedGuide,
        llm: LLMService,
        current_frame: CurrentFrameTool,
        image_query: ImageQueryTool,
        vision_timeout_s: float,
        aggregation: VoiceAggregationAgent,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._aggregation = aggregation
        self._controls: dict[str, asyncio.Lock] = {}
        self._runtime: AgentRuntime | None = None
        self._connected: set[str] = set()
        self._turns: dict[str, asyncio.Task[None]] = {}
        self._stopping = False
        self._backend = _Replay(
            guide=guide, current_frame=current_frame, image_query=image_query,
            vision_timeout_s=vision_timeout_s, run_model=self._tool_loop,
            announce=self._announce, invalidate=aggregation.release,
            connected=lambda pid: pid in self._connected,
        )

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        if self._runtime is not None and self._runtime is not runtime:
            raise RuntimeError("SOP engine is already bound to another runtime")
        self._runtime = runtime

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(self, _event: VoiceInterrupted, ctx: RuntimeContext) -> None:
        participant_id = ctx.metadata.participant_id
        participants = tuple(self._turns) if participant_id is None else (participant_id,)
        for pid in participants:
            async with self._controls.setdefault(pid, asyncio.Lock()):
                await _cancel(self._turns, pid)
                await self._aggregation.release(pid)

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        async with self._controls.setdefault(participant_id, asyncio.Lock()):
            if self._stopping or participant_id in self._connected:
                return
            self._connected.add(participant_id)
            message = self._backend.join(participant_id)
            await self._say(participant_id, message)

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        self._connected.discard(participant_id)
        async with self._controls.setdefault(participant_id, asyncio.Lock()):
            await _cancel(self._turns, participant_id)
            await self._backend.leave(participant_id)
            await self._aggregation.release(participant_id)

    @subscribe(USER_QUERY_TOPIC)
    async def user_query(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        # Controls must finish before the next utterance can start another turn.
        async with self._controls.setdefault(participant_id, asyncio.Lock()):
            if participant_id not in self._connected:
                return
            await _cancel(self._turns, participant_id)
            await self._aggregation.release(participant_id)
            if participant_id not in self._connected:
                return
            await self._start_answer(query, participant_id)

    async def _say(self, participant_id: str, text: str) -> None:
        runtime = self._runtime
        if runtime is not None and runtime.running:
            await runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(text=text, interrupt=True),
                participant_id=participant_id,
                source="sop-replay",
            )

    async def _start_answer(self, query: UserQuery, participant_id: str) -> None:
        await _cancel(self._turns, participant_id)
        task = asyncio.create_task(
            self._answer(query, participant_id),
            name=f"sop-answer:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._turns[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: _discard(self._turns, pid, completed))

    async def stop(self) -> None:
        self._stopping = True
        participants = tuple(self._connected)
        self._connected.clear()
        tasks = tuple(self._turns.values())
        self._turns.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._backend.stop()
        for pid in participants:
            await self._aggregation.release(pid)
        self._controls.clear()

    async def _announce(self, participant_id: str, text: str) -> None:
        runtime = self._runtime
        if runtime is not None and runtime.running:
            try:
                await runtime.publish(
                    VOICE_CONTRIBUTION_TOPIC, VoiceOutput(text=text),
                    participant_id=participant_id, source="sop-replay",
                )
            except RuntimeClosedError:
                pass

    async def _answer(self, query: UserQuery, participant_id: str) -> None:
        try:
            response = await self._route(query.text, participant_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).error("SOP query failed pid={!r}", participant_id)
            response = "I couldn't complete that guide request. Please try again."
        runtime = self._runtime
        if not response or runtime is None or not runtime.running or participant_id not in self._connected:
            return
        try:
            await runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(
                    text=response, interrupt=True, timestamp_us=query.timestamp_us,
                ),
                participant_id=participant_id,
                source="sop-replay",
            )
        except RuntimeClosedError:
            return

    async def _route(self, text: str, participant_id: str) -> str:
        guidance = await self._backend.guidance(participant_id)
        if guidance is None:
            return ""
        tools = _tools_for_query(guidance.tools, text)
        system = f"{_FOREGROUND_PROMPT}\n\n{guidance.instructions}"
        if tools.get("workflow__reset") is None:
            system += (
                "\nThis turn is informational, not authorization to change guide state. "
                "Explain or clarify briefly using only exposed tools for requested live facts. "
                "Do not claim to advance, skip, reset, restart, or record."
            )
        user = json.dumps({"request": text, **guidance.context}, ensure_ascii=False)
        result = await self._tool_loop(system, user, tools, foreground=True)
        return await guidance.finish(result)

    async def _tool_loop(self, system: str, user: str, tools: ToolSet, *, foreground: bool = False) -> str:
        options = _FOREGROUND_MODEL_OPTIONS if foreground else {"max_tokens": 512, "enable_thinking": False}

        async def call_model(messages: tuple[ChatMessage, ...], definitions: tuple[ToolDef, ...]) -> ChatResponse:
            return await self._llm.chat(
                messages,
                tools=definitions,
                temperature=0.0,
                **options,
            )

        result = await run_tool_loop(
            (ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)),
            tools,
            call_model,
            max_iterations=_MAX_TOOL_ROUNDS,
            max_tool_calls=6,
        )
        return result.content.strip() or "Done."

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("SOP engine requires a participant")
        return participant_id


def _tools_for_query(tools: ToolSet, query: str) -> ToolSet:
    """Conservatively filter familiar English phrasing; not an authorization boundary.

    The conversation prompt handles intent outside these cues. State-changing
    tools still enforce session freshness and completion independently of wording.
    """
    text = " ".join(query.casefold().split()).replace("’", "'")
    unsupported_capture = re.search(r"\b(?:record|recording|capture)\b", text)
    informational = re.search(
        r"\b(?:don't|do not|never|not yet|said|says|told|quoted|hypothetical)\b|"
        r"^(?:if|suppose|imagine|why|how|what|should i|would it|can i|could i)\b",
        text,
    )
    if informational or unsupported_capture:
        return ToolSet(tool for name, tool in tools.items() if name not in {
            "workflow__advance", "workflow__restart", "workflow__reset",
        })
    return tools
