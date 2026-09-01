# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic foreground routing with one participant-scoped tool loop."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import nemo_relay
from loguru import logger
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolDef, VLMService
from xr_ai_runtime import Agent, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.tool_calling import ToolLoopIterationLimitError, run_tool_loop
from xr_ai_tools.vision import (
    ImageQueryRequest,
    ImageQueryResult,
    StreamingImageQueryTool,
)
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    UserQuery,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
)

from .background_context import BackgroundContextAgent
from .change_watch import ChangeWatchAgent
from .events import (
    FOREGROUND_RECORD_TOPIC,
    INTERRUPTED_TOPIC,
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ForegroundRecord,
    ParticipantCleanupComplete,
)
from .images import ParticipantImageAgent
from .spec import Workflow
from .transcript import TranscriptAgent
from .video_log import VideoLogAgent
from .workflow import GuidanceAgent
from .workflow_tools import CurrentViewRequest, rag_lookup_tool

_MAX_TOOL_ROUNDS = 4
_PROMPTS = Path(__file__).resolve().parent / "prompts"
_IDLE_PROMPT = (_PROMPTS / "foreground_idle.txt").read_text(encoding="utf-8").strip()
_WORKFLOW_CONTROLS = frozenset(
    {"workflow__advance", "workflow__reset", "workflow__restart", "workflow__status"}
)
_TEA_PROMPT = (
    "Next/continue/advance: workflow__advance(skip=false). Skip: "
    "workflow__advance(skip=true). Exit/stop/reset guide: workflow__reset. "
    "Restart: workflow__restart. Guide status: workflow__status. Questions "
    "using these words are not commands."
)
_VOICE_PROMPT = (
    "Answer in at most two short sentences. Use a tool for requested live "
    "visual or timer facts; if unavailable, say so. Never infer unseen facts."
)
_ACTIVE_POLICY = (
    "While the guide is active, answer tea-making questions from the guide "
    "position and state, and briefly decline every unrelated request without a "
    "tool. For instructions or an overview, summarize the guide order with known "
    "brewing values; do not perform a live check. A procedural, current, next, "
    "or previous-step question is not a live-fact request. Call a step tool only "
    "for a requested live fact that tool can obtain; never substitute another "
    "tool. For a direct guide command, call the exposed workflow tool."
)
_HUMAN_PROMPT = (
    "Use natural spoken language. Rewrite tool/state abbreviations, symbols, "
    "units, and machine notation in words; preserve meaning."
)
_TEA_MANAGEMENT_TOOLS = (
    "workflow__advance",
    "workflow__reset",
    "workflow__restart",
    "workflow__status",
)


@dataclass(frozen=True, slots=True)
class _FocusedAgent:
    """One immutable model-facing agent built from a workflow step."""

    name: str
    system_prompt: str
    tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedTurn:
    agent: _FocusedAgent
    user_message: str
    tools: ToolSet
    route: str


def _requested_workflow_control(query: str) -> str | None:
    """Return only a workflow control explicitly requested by the utterance."""

    text = " ".join(query.casefold().strip(" .!?").split())
    polite = r"(?:(?:please|kindly)\s+)?(?:(?:can|could|would)\s+you\s+)?"
    if re.fullmatch(
        polite
        + r"(?:next|continue|advance|move on|go on)(?:\s+(?:to\s+)?(?:the\s+)?"
        r"(?:next|following)?\s*(?:tea\s+)?(?:step|guide))?",
        text,
    ) or re.fullmatch(
        polite + r"skip(?:\s+(?:(?:this|the|current|next)\s+)?(?:tea\s+)?step)?",
        text,
    ):
        return "workflow__advance"
    if re.fullmatch(
        polite
        + r"(?:end|exit|stop|reset|cancel)(?:\s+(?:(?:the|this|my)\s+)?"
        r"(?:tea(?:-making)?\s+)?(?:guide|guidance|session|demo))",
        text,
    ):
        return "workflow__reset"
    if re.fullmatch(
        polite
        + r"(?:restart|start over|begin again)(?:\s+(?:the\s+)?(?:tea\s+)?"
        r"(?:guide|guidance|instructions?))?",
        text,
    ) or re.fullmatch(
        polite
        + r"begin\s+(?:the\s+)?(?:tea\s+)?(?:guide|guidance|instructions?)"
        r"\s+again(?:\s+from\s+(?:the\s+)?first\s+step)?",
        text,
    ):
        return "workflow__restart"
    if re.fullmatch(
        polite
        + r"(?:(?:what(?:'s| is)\s+)?(?:the\s+)?(?:tea\s+)?guide\s+status|"
        r"(?:what(?:'s| is)\s+)?(?:the\s+)?status\s+of\s+(?:the|my)\s+tea\s+guide|"
        r"report\s+(?:the\s+)?(?:tea\s+)?guide\s+status)",
        text,
    ):
        return "workflow__status"
    return None


def _workflow_tools_for_query(tools: ToolSet, query: str) -> ToolSet:
    requested = _requested_workflow_control(query)
    return ToolSet(
        tool
        for name, tool in tools.items()
        if name not in _WORKFLOW_CONTROLS or name == requested
    )


class ForegroundAgent(Agent):
    """Select the idle root or active tea tool set before invoking the model."""

    def __init__(
        self,
        *,
        llm: LLMService,
        images: ParticipantImageAgent,
        vlm: VLMService,
        rag: RAGTools,
        guidance: GuidanceAgent,
        background_context: BackgroundContextAgent,
        change_watch: ChangeWatchAgent,
        transcript: TranscriptAgent,
        video_log: VideoLogAgent,
        prompt: str,
        vlm_timeout_s: float = 15.0,
    ) -> None:
        if vlm_timeout_s <= 0:
            raise ValueError("vlm_timeout_s must be positive")
        self._vision = StreamingImageQueryTool(images=images.images, vlm=vlm)
        super().__init__((self._vision,))
        self._llm = llm
        self._images = images
        self._rag = rag
        self._guidance = guidance
        self._background_context = background_context
        self._change_watch = change_watch
        self._transcript = transcript
        self._video_log = video_log
        self._prompt = prompt.strip()
        self._root_agent = _FocusedAgent(
            name="foreground_root",
            system_prompt=self._with_route_policy(_IDLE_PROMPT),
            tool_names=(),
        )
        self._tea_agents = _build_tea_agents(guidance.workflow)
        self._vlm_timeout_s = vlm_timeout_s
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @subscribe(USER_QUERY_TOPIC)
    async def answer(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        await self._cancel(participant_id)
        task = asyncio.create_task(
            self._run_turn(query, ctx),
            name=f"tea-foreground:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(
            lambda completed, pid=participant_id: self._discard(pid, completed)
        )

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        await self._cancel(self._participant(ctx))
        await ctx.publish(
            PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
            ParticipantCleanupComplete(
                generation=ctx.metadata.message_id,
                producer="foreground",
            ),
        )

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: VoiceInterrupted,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            await self.stop()
        else:
            await self._cancel(participant_id)

    async def stop(self) -> None:
        """Cancel every participant turn owned by this agent."""

        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_turn(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            try:
                response, tools, spoken = await self._answer(
                    query.text,
                    participant_id,
                    ctx,
                    timestamp_us=query.timestamp_us,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).error(
                    "tea foreground query failed pid={!r}", participant_id
                )
                response = "I couldn't complete that request. Please try again."
                tools = []
                spoken = False
            try:
                if not spoken:
                    await ctx.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(
                            text=response,
                            interrupt=True,
                            timestamp_us=query.timestamp_us,
                        ),
                    )
            except RuntimeClosedError:
                return
            try:
                await ctx.publish(
                    FOREGROUND_RECORD_TOPIC,
                    ForegroundRecord(
                        timestamp_us=query.timestamp_us,
                        query=query.text,
                        response=response,
                        tools=tools,
                    ),
                )
            except RuntimeClosedError:
                return
            except Exception:
                logger.opt(exception=True).error(
                    "tea foreground record failed pid={!r}", participant_id
                )

    async def _answer(
        self,
        query: str,
        participant_id: str,
        ctx: RuntimeContext | None = None,
        *,
        timestamp_us: int | None = None,
    ) -> tuple[str, list[str], bool]:
        turn = self._prepare_turn(
            participant_id,
            query=query,
            ctx=ctx,
            timestamp_us=timestamp_us,
        )

        round_index = 0

        async def call_model(
            messages: tuple[ChatMessage, ...],
            definitions: tuple[ToolDef, ...],
        ) -> ChatResponse:
            nonlocal round_index
            round_index += 1
            response = await self._llm.chat(
                messages,
                tools=definitions,
                max_tokens=512,
                temperature=0.0,
                enable_thinking=False,
            )
            logger.info(
                "tea foreground route pid={!r} route={} agent={} round={} tools={}",
                participant_id,
                turn.route,
                turn.agent.name,
                round_index,
                [call.name for call in response.tool_calls or ()],
            )
            return response

        try:
            result = await run_tool_loop(
                (
                    ChatMessage(role="system", content=turn.agent.system_prompt),
                    ChatMessage(role="user", content=turn.user_message),
                ),
                turn.tools,
                call_model,
                max_iterations=_MAX_TOOL_ROUNDS,
            )
        except ToolLoopIterationLimitError as exc:
            return (
                "I couldn't finish that request within the tool limit.",
                [record.call.name for record in exc.tool_calls],
                False,
            )
        fallback = "Done." if result.return_direct else "I don't have an answer."
        calls = [record.call.name for record in result.tool_calls]
        return (
            result.content.strip() or fallback,
            calls,
            result.return_direct
            and bool(result.tool_calls)
            and result.tool_calls[-1].call.name == "current_view",
        )

    def _prepare_route(
        self,
        participant_id: str,
        *,
        query: str,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> tuple[str, ToolSet, str]:
        """Return the model-facing prompt, tools, and route for route evals."""

        turn = self._prepare_turn(
            participant_id,
            query=query,
            ctx=ctx,
            timestamp_us=timestamp_us,
        )
        return turn.agent.system_prompt, turn.tools, turn.route

    def _prepare_turn(
        self,
        participant_id: str,
        *,
        query: str,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> _PreparedTurn:
        """Select the same focused root-or-step agent shape used by NAT."""

        session = self._guidance.store.find(participant_id)
        if session is None or not session.active or session.step_id is None:
            tools = self._root_tools(
                participant_id,
                ctx=ctx,
                timestamp_us=timestamp_us,
            )
            return _PreparedTurn(
                agent=self._root_agent,
                user_message=_json(request=query),
                tools=tools,
                route="root",
            )

        step = self._guidance.workflow.step(session.step_id)
        agent = self._tea_agents[step.id]
        active_tools = self._guidance.active_tools(participant_id)
        if active_tools is None:
            raise RuntimeError("active tea context has no active tool set")
        tools = _select_tools(active_tools, agent.tool_names)
        tools = _workflow_tools_for_query(tools, query)
        tools = _guide_tools_for_query(tools, query)
        background_tools = self._background_tools_for_query(participant_id, query)
        background_items = tuple(background_tools.items())
        if background_items:
            tools = background_tools
            agent = _FocusedAgent(
                name=f"{agent.name}_background",
                system_prompt=f"{agent.system_prompt}\n{self._prompt}".strip(),
                tool_names=tuple(name for name, _tool in background_items),
            )
        return _PreparedTurn(
            agent=agent,
            user_message=_json(
                request=query,
                state=self._guidance.workflow.project(step, session.state),
            ),
            tools=tools,
            route="tea",
        )

    def _with_route_policy(self, route_prompt: str) -> str:
        return f"{self._prompt}\n\n{route_prompt}"

    def _root_tools(
        self,
        participant_id: str,
        *,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> ToolSet:
        current_view = self._current_view_tool(
            participant_id,
            ctx=ctx,
            timestamp_us=timestamp_us,
        )
        return _merge_tool_sets(
            ToolSet((current_view, rag_lookup_tool(self._rag))),
            self._guidance.root_tools(participant_id),
            self._background_tools(participant_id),
        )

    def _current_view_tool(
        self,
        participant_id: str,
        *,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> Tool[CurrentViewRequest, ImageQueryResult]:
        async def inspect(request: CurrentViewRequest) -> ImageQueryResult:
            if ctx is None:
                raise RuntimeError("current-view streaming requires a runtime context")
            return await self._stream_current_view(
                request.question,
                participant_id,
                ctx,
                timestamp_us=timestamp_us,
            )

        return Tool(
            "current_view",
            "Inspect this participant's current camera frame to answer a question about the current scene.",
            CurrentViewRequest,
            ImageQueryResult,
            inspect,
            return_direct=True,
            render_result=lambda result: result.text,
        )

    async def _stream_current_view(
        self,
        query: str,
        participant_id: str,
        ctx: RuntimeContext,
        *,
        timestamp_us: int | None,
    ) -> ImageQueryResult:
        response_id = ctx.metadata.message_id
        first = True
        opened = False
        cancelled = False
        chunks: list[str] = []
        try:
            async with asyncio.timeout(self._vlm_timeout_s):
                try:
                    frame = await self._images.get_current_frame.execute(
                        CurrentFrameRequest(participant_id=participant_id)
                    )
                except (FrameUnavailable, RuntimeError) as exc:
                    unavailable = _frame_unavailable_message(exc)
                    if unavailable is None:
                        raise
                    await ctx.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(
                            text=unavailable,
                            response_id=response_id,
                            final=False,
                            interrupt=True,
                            timestamp_us=timestamp_us,
                        ),
                    )
                    opened = True
                    return ImageQueryResult(text=unavailable, available=False)
                stream = self._vision.stream(
                    ImageQueryRequest(image=frame.image, query=query)
                )
                try:
                    async for chunk in stream:
                        if first and not chunk.text.strip():
                            continue
                        chunks.append(chunk.text)
                        await ctx.publish(
                            VOICE_CONTRIBUTION_TOPIC,
                            VoiceOutput(
                                text=chunk.text,
                                response_id=response_id,
                                final=False,
                                interrupt=first,
                                timestamp_us=timestamp_us,
                            ),
                        )
                        first = False
                        opened = True
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
                if not chunks:
                    unavailable = (
                        "Unable to inspect the current frame because the vision model "
                        "returned no description."
                    )
                    await ctx.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(
                            text=unavailable,
                            response_id=response_id,
                            final=False,
                            interrupt=True,
                            timestamp_us=timestamp_us,
                        ),
                    )
                    opened = True
                    return ImageQueryResult(text=unavailable, available=False)
            return ImageQueryResult(text="".join(chunks))
        except TimeoutError:
            unavailable = (
                "Unable to inspect the current frame before the vision timeout."
            )
            await ctx.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(
                    text=unavailable,
                    response_id=response_id,
                    final=False,
                    interrupt=first,
                    timestamp_us=timestamp_us,
                ),
            )
            opened = True
            return ImageQueryResult(text=unavailable, available=False)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if opened and not cancelled:
                with suppress(RuntimeClosedError):
                    await ctx.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(
                            response_id=response_id,
                            timestamp_us=timestamp_us,
                        ),
                    )

    def _background_tools(self, participant_id: str) -> ToolSet:
        return _merge_tool_sets(
            self._background_context.participant_tools(participant_id),
            self._change_watch.participant_tools(participant_id),
            self._transcript.participant_tools(participant_id),
            self._video_log.participant_tools(participant_id),
        )

    def _background_tools_for_query(
        self,
        participant_id: str,
        query: str,
    ) -> ToolSet:
        """Expose background tools only to an explicit background request."""

        text = " ".join(query.casefold().split())
        history = re.search(
            r"\b(?:background|monitor|watcher|transcript|video log|activity log)\b"
            r".*\b(?:report\w*|notic\w*|observ\w*|history|recent|earlier|past|"
            r"chang\w*|happen\w*)\b",
            text,
        )
        if history is not None:
            return self._background_context.participant_tools(participant_id)

        catalogs: list[ToolSet] = []
        if re.search(
            r"\b(?:monitor|monitoring|visual changes?|change watch)\b|"
            r"\bbackground\b.*\bwatch(?:ing)?\b|"
            r"\bwatch(?:ing)?\b.*\bbackground\b",
            text,
        ):
            catalogs.append(self._change_watch.participant_tools(participant_id))
        if re.search(r"\b(?:transcript|conversation recording)\b", text):
            catalogs.append(self._transcript.participant_tools(participant_id))
        if re.search(r"\b(?:video log|visual activity log|activity recording)\b", text):
            catalogs.append(self._video_log.participant_tools(participant_id))
        if not catalogs:
            return ToolSet(())
        catalogs.append(self._background_context.participant_tools(participant_id))
        return _merge_tool_sets(*catalogs)

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if task.cancelled():
            return
        with suppress(asyncio.CancelledError):
            if error := task.exception():
                logger.error(
                    "tea foreground stopped pid={!r}: {!r}", participant_id, error
                )

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("foreground queries require a participant")
        return participant_id


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


def _merge_tool_sets(*catalogs: ToolSet) -> ToolSet:
    tools: dict[str, Tool] = {}
    for catalog in catalogs:
        for name, tool in catalog.items():
            if name in tools:
                raise ValueError(f"duplicate participant tool: {name}")
            tools[name] = tool
    return ToolSet(tools)


def _build_tea_agents(workflow: Workflow) -> dict[str, _FocusedAgent]:
    """Build one focused foreground agent for every declarative step."""

    agents: dict[str, _FocusedAgent] = {}
    sequence = ", then ".join(step.title for step in workflow.steps.values())
    for step in workflow.steps.values():
        next_title = (
            workflow.step(step.next_step).title
            if step.next_step is not None
            else "finish the guide"
        )
        agents[step.id] = _FocusedAgent(
            name=f"foreground_tea_{step.id}",
            system_prompt=(
                f"{_TEA_PROMPT}\n{_VOICE_PROMPT}\n"
                f"{workflow.foreground_prompt} Guide order: {sequence}. "
                f"Current step: {step.title}. Next: {next_title}.\n"
                f"{step.voice.prompt}\n{_ACTIVE_POLICY}\n{_HUMAN_PROMPT}"
            ).strip(),
            tool_names=(*_TEA_MANAGEMENT_TOOLS, *step.voice.tools),
        )
    return agents


def _select_tools(tools: ToolSet, names: tuple[str, ...]) -> ToolSet:
    selected: dict[str, Tool] = {}
    for name in names:
        tool = tools.get(name)
        if tool is None:
            raise RuntimeError(f"focused agent requires unavailable tool {name!r}")
        selected[name] = tool
    return ToolSet(selected)


def _guide_tools_for_query(tools: ToolSet, query: str) -> ToolSet:
    """Keep procedural guide questions free of irrelevant live tools."""

    text = " ".join(query.casefold().split())
    procedural = re.search(
        r"\b(?:instructions?|details?|overview|procedure|guide order|"
        r"current step|next step|previous step)\b|"
        r"\bwhat (?:should|do) i do(?: now)?\b|"
        r"\bwhere should i begin\b|\bwhat(?:'s| is) next\b",
        text,
    )
    if procedural is None:
        return tools
    return ToolSet(
        tool for name, tool in tools.items() if name in _WORKFLOW_CONTROLS
    )


def _json(**value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


__all__ = ["ForegroundAgent"]
