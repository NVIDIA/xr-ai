# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-local execution engine for approved declarative SOP guides."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nemo_relay
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, ChatResponse, LLMService, ToolDef
from xr_ai_runtime import Agent, AgentRuntime, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop
from xr_ai_tools.types import EmptyRequest, StrictRequest
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult, ImageQueryTool
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    UserQuery,
    VoiceAggregationAgent,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)

from ._workflow_spec import Step, Workflow, _CurrentViewRequest, _TimerRequest, _TimerResult
from .guides import SelectedGuide
from .replay_events import INTERRUPTED_TOPIC, PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC, USER_QUERY_TOPIC

_POLL_INTERVAL_S = 0.25
_MAX_TOOL_ROUNDS = 3
_FOREGROUND_MODEL_OPTIONS = {"max_tokens": 1536, "enable_thinking": True, "thinking_budget": 1024}
_FOREGROUND_PROMPT = (Path(__file__).with_name("prompts") / "replay_conversation.txt").read_text(
    encoding="utf-8",
).strip()


class _NowResult(BaseModel):
    epoch_us: int


class _AdvanceGuideRequest(StrictRequest):
    skip: bool = Field(default=False, strict=True)


class _GuideResult(BaseModel):
    message: str


class _CommitRequest(StrictRequest):
    model_config = ConfigDict(extra="forbid", strict=True)

    updates: dict[str, bool | int | float | str] = Field(default_factory=dict)
    message: str = Field(default="", max_length=240)


class _CommitResult(BaseModel):
    accepted: bool
    complete: bool
    message: str
    revision: int


@dataclass(slots=True)
class _Session:
    participant_id: str
    guide: SelectedGuide
    state: dict[str, Any]
    step_id: str
    revision: int = 1
    next_tick: float = 0.0
    evidence_hits: int = 0
    notices: list[str] = field(default_factory=list)
    skipped: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def workflow(self) -> Workflow:
        workflow = self.guide.workflow
        if workflow is None:
            raise RuntimeError("a running session lost its pinned workflow")
        return workflow

    @property
    def step(self) -> Step:
        return self.workflow.steps[self.step_id]


class ReplayAgent(Agent):
    """Run one approved guide repeatedly with participant-local progress."""

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
        if not guide.workflow.runnable:
            raise ValueError("replay requires an approved guide")
        if not math.isfinite(vision_timeout_s) or vision_timeout_s <= 0:
            raise ValueError("vision_timeout_s must be finite and positive")
        self._guide = guide
        self._llm = llm
        self._current_frame = current_frame
        self._image_query = image_query
        self._vision_timeout_s = vision_timeout_s
        self._aggregation = aggregation
        self._controls: dict[str, asyncio.Lock] = {}
        self._runtime: AgentRuntime | None = None
        self._connected: set[str] = set()
        self._sessions: dict[str, _Session] = {}
        self._monitors: dict[str, asyncio.Task[None]] = {}
        self._turns: dict[str, asyncio.Task[None]] = {}
        self._stopping = False

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
                await self._cancel(self._turns, pid)
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
            self._sessions.pop(participant_id, None)
            message = self._start(participant_id)
            self._start_monitor(participant_id)
            await self._say(participant_id, f"{self._guide.workflow.name}. {message}")

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        self._connected.discard(participant_id)
        async with self._controls.setdefault(participant_id, asyncio.Lock()):
            await self._cancel(self._turns, participant_id)
            await self._cancel(self._monitors, participant_id)
            self._sessions.pop(participant_id, None)
            await self._aggregation.release(participant_id)
            self._current_frame.release(participant_id)

    @subscribe(USER_QUERY_TOPIC)
    async def user_query(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        # Controls must finish before the next utterance can start another turn.
        async with self._controls.setdefault(participant_id, asyncio.Lock()):
            if participant_id not in self._connected:
                return
            await self._cancel(self._turns, participant_id)
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
        await self._cancel(self._turns, participant_id)
        task = asyncio.create_task(
            self._answer(query, participant_id),
            name=f"sop-answer:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._turns[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(self._turns, pid, completed))

    async def stop(self) -> None:
        self._stopping = True
        participants = tuple(self._connected)
        tasks = tuple((*self._turns.values(), *self._monitors.values()))
        self._turns.clear()
        self._monitors.clear()
        self._connected.clear()
        self._sessions.clear()
        self._controls.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for participant_id in participants:
            self._current_frame.release(participant_id)

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
        session = self._sessions.get(participant_id)
        if session is None:
            return ""
        return await self._answer_step_question(session, text)

    def _guide_tools(self, participant_id: str) -> tuple[ToolSet, list[str]]:
        """Expose application-owned controls, never recording or evidence commits."""
        session = self._sessions.get(participant_id)
        revision = session.revision if session else None
        results: list[str] = []

        async def control(action) -> _GuideResult:
            async with self._controls.setdefault(participant_id, asyncio.Lock()):
                current = self._sessions.get(participant_id)
                if participant_id not in self._connected:
                    message = "Guide controls are unavailable while disconnected."
                elif current is not session or (session is not None and session.revision != revision):
                    message = "The guide changed while I was answering. Ask again for the current step."
                else:
                    message = await action()
                results.append(message)
                return _GuideResult(message=message)

        async def status(_request: EmptyRequest) -> _GuideResult:
            return await control(lambda: self._status(participant_id))

        async def advance(request: _AdvanceGuideRequest) -> _GuideResult:
            return await control(lambda: self._advance(participant_id, skip=request.skip))

        async def reset(_request: EmptyRequest) -> _GuideResult:
            return await control(lambda: self._reset(participant_id))

        async def restart(_request: EmptyRequest) -> _GuideResult:
            return await control(lambda: self._restart(participant_id))

        def tool(name, description, request_type, handler):
            return Tool(
                name, description, request_type, _GuideResult, handler,
                render_result=lambda result: result.message, return_direct=True,
            )

        tools = [
            tool("workflow__status", "Report guide status when requested; do not substitute for a live visual check.",
                 EmptyRequest, status),
        ]
        if session is not None:
            tools.extend((
                tool("workflow__advance",
                     "USE WHEN: the user directly commands the active guide to move forward now; "
                     "always call even when supplied state looks incomplete because the tool alone decides readiness. "
                     "Set skip=false for continue/next/move-on/proceed commands and "
                     "skip=true only for bypass/skip-this-step commands. DO NOT USE WHEN: a question, "
                     "quotation, hypothetical, deliberation, negation, report, or unrelated wording.",
                     _AdvanceGuideRequest, advance),
                tool("workflow__reset",
                     "Reset the selected guide to step one only for a direct present request to stop or reset it. "
                     "Not for questions, quotations, hypotheticals, reports, or negations. "
                     "Does not select another guide.",
                     EmptyRequest, reset),
                tool("workflow__restart",
                     "Restart from step one only when directly requested now; not for questions, quotations, "
                     "reports, hypotheticals, or negations.", EmptyRequest, restart),
            ))
        return ToolSet(tools), results

    def _start(self, participant_id: str) -> str:
        guide = self._guide
        workflow = guide.workflow
        session = _Session(
            participant_id=participant_id,
            guide=guide,
            state=workflow.initial_state(),
            step_id=workflow.start_step,
        )
        self._sessions[participant_id] = session
        logger.info(
            "SOP started pid={!r} guide={} version={} sha256={}",
            participant_id,
            workflow.id,
            workflow.version,
            guide.sha256,
        )
        return session.step.enter_message

    async def _status(self, participant_id: str) -> str:
        session = self._sessions.get(participant_id)
        if session is None:
            return "No guide is active."
        async with session.lock:
            suffix = " Complete; say next when ready." if session.step.is_complete(session.state) else ""
            return f"{session.workflow.name}, current step: {session.step.title}.{suffix}"

    async def _advance(self, participant_id: str, *, skip: bool) -> str:
        session = self._sessions.get(participant_id)
        if session is None:
            return "No guide is active."
        async with session.lock:
            step = session.step
            complete = step.is_complete(session.state)
            if not complete and not skip:
                return f"{step.title} is not complete yet. Say skip to move on anyway."
            skipping = skip and not complete
            if skipping:
                session.skipped = True
                session.state.update(copy.deepcopy(step.state_on_skip))
            next_step = None if skipping and step.complete_on_skip else step.next_step
            session.revision += 1
            session.evidence_hits = 0
            session.next_tick = 0.0
            session.notices.clear()
            if next_step is None:
                message = "Replay ended with skipped steps." if session.skipped else session.workflow.complete_message
                self._reset_progress(session)
                return f"{message} Ready to replay again. {session.step.enter_message}"
            session.step_id = next_step
            if skipping and step.skip_message:
                return f"{step.skip_message} {session.step.enter_message}"
            return session.step.enter_message

    async def _reset(self, participant_id: str) -> str:
        return await self._restart(participant_id)

    async def _restart(self, participant_id: str) -> str:
        session = self._sessions.get(participant_id)
        if session is None:
            return "No guide is active."
        async with session.lock:
            self._reset_progress(session)
            return session.step.enter_message

    @staticmethod
    def _reset_progress(session: _Session) -> None:
        session.state = session.workflow.initial_state()
        session.step_id = session.workflow.start_step
        session.revision += 1
        session.next_tick = 0.0
        session.evidence_hits = 0
        session.notices.clear()
        session.skipped = False

    async def _answer_step_question(self, session: _Session, query: str) -> str:
        async with session.lock:
            step = session.step
            revision = session.revision
            state = session.workflow.project(step, session.state)
            tools = self._named_tools(session.participant_id, step.voice.tools)
            controls, control_results = self._guide_tools(session.participant_id)
            tools = _tools_for_query(ToolSet({**dict(tools.items()), **dict(controls.items())}), query)
            policy = _FOREGROUND_PROMPT if tools.get("workflow__advance") is not None else (
                "This turn is informational, not authorization to change guide state. "
                "Explain or clarify briefly. Do not execute quoted, reported, hypothetical, or negated actions. "
                "Do not claim to advance, skip, reset, restart, or record. "
                "Recording is unsupported in replay mode. Use only the tools actually exposed in this turn "
                "for requested live facts; if none can obtain the requested evidence, say it is unavailable."
            )
            system = (
                f"You are the active {session.workflow.name} SOP guide for this replay connection. "
                "Answer guide questions from the supplied state and guide_order. "
                "Decline unrelated requests without revealing their answer or exiting the guide. "
                "Unqualified visual checks refer to the current step: use current_view for a requested "
                "check of whether this looks correct. Procedural explanations use supplied context.\n\n"
                f"{session.workflow.foreground_prompt}\n"
                f"Current step: {step.title}.\n{step.voice.prompt}\n\n{policy}"
            )
            user = json.dumps(
                {"request": query, "guide_order": [item.title for item in session.workflow.steps.values()],
                 "state": state},
                ensure_ascii=False,
            )
        result = await self._tool_loop(system, user, tools, foreground=True)
        if control_results:
            return control_results[-1]
        async with session.lock:
            if session.revision != revision or self._sessions.get(session.participant_id) is not session:
                return "The guide changed while I was answering. Ask again for the current step."
        return result

    def _start_monitor(self, participant_id: str) -> None:
        if participant_id in self._monitors and not self._monitors[participant_id].done():
            return
        task = asyncio.create_task(
            self._monitor(participant_id),
            name=f"sop-monitor:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._monitors[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(self._monitors, pid, completed))

    async def _monitor(self, participant_id: str) -> None:
        while participant_id in self._connected:
            session = self._sessions.get(participant_id)
            if session is not None:
                try:
                    await self._tick(session)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.opt(exception=True).warning("SOP observation failed pid={!r}", participant_id)
            await asyncio.sleep(_POLL_INTERVAL_S)

    async def _tick(self, session: _Session) -> None:
        async with session.lock:
            if self._sessions.get(session.participant_id) is not session or session.step.is_complete(session.state):
                return
            now = time.monotonic()
            if session.next_tick > now:
                return
            step = session.step
            revision = session.revision
            state = dict(session.state)
            session.next_tick = now + step.trigger.interval_s
        available, observation = await self._trigger(session.participant_id, step, state)
        if not available:
            return
        run_model = True
        async with session.lock:
            if not self._current(session, step, revision):
                return
            evidence = step.evidence
            if evidence is not None:
                value = observation if isinstance(observation, str) else json.dumps(observation, separators=(",", ":"))
                matched = re.fullmatch(evidence.pattern, value.strip()) is not None
                session.evidence_hits = session.evidence_hits + 1 if matched else 0
                if evidence.commit:
                    run_model = False
                    if session.evidence_hits >= evidence.consecutive:
                        self._commit(session, step, evidence.commit, "")
            state = dict(session.state)
        if run_model:
            await self._observation_turn(session, step, revision, state, observation)
        await self._publish_notices(session)

    async def _observation_turn(
        self,
        session: _Session,
        step: Step,
        revision: int,
        state: dict[str, Any],
        observation: Any,
    ) -> None:
        commit = self._commit_tool(session, step, revision)
        named = self._named_tools(session.participant_id, step.agent.tools)
        tools = ToolSet({commit.name: commit, **dict(named.items())})
        writable = {
            name: {
                "type": session.workflow.state_fields[name].type,
                "description": session.workflow.state_fields[name].description,
            }
            for name in step.writes
        }
        prompt = json.dumps(
            {
                "role": "SOP observation controller",
                "instructions": step.agent.prompt,
                "observation": observation,
                "state": session.workflow.project(step, state),
                "writable_state": writable,
                "complete_when": step.complete_when,
                "rules": [
                    "Use workflow__commit for every state change.",
                    "Commit only facts supported by the observation or deterministic tools.",
                    "Do not infer hidden actions or outcomes.",
                ],
            },
            ensure_ascii=False,
        )
        try:
            await self._tool_loop("You update one bounded SOP step.", prompt, tools)
        except ToolLoopError:
            logger.opt(exception=True).warning("SOP observation tool loop failed")

    def _commit_tool(self, session: _Session, step: Step, revision: int) -> Tool[_CommitRequest, _CommitResult]:
        async def commit(request: _CommitRequest) -> _CommitResult:
            async with session.lock:
                if not self._current(session, step, revision):
                    return _CommitResult(
                        accepted=False,
                        complete=False,
                        message="stale workflow revision",
                        revision=session.revision,
                    )
                accepted, complete, message = self._commit(session, step, request.updates, request.message)
                return _CommitResult(
                    accepted=accepted,
                    complete=complete,
                    message=message,
                    revision=session.revision,
                )

        return Tool(
            "workflow__commit",
            "Atomically commit evidence-backed fields writable by the active SOP step.",
            _CommitRequest,
            _CommitResult,
            commit,
            return_direct=True,
        )

    def _commit(
        self,
        session: _Session,
        step: Step,
        updates: dict[str, Any],
        message: str,
    ) -> tuple[bool, bool, str]:
        unknown = updates.keys() - set(step.writes)
        if unknown:
            return False, False, f"fields not writable in this step: {sorted(unknown)}"
        for name, value in updates.items():
            if not session.workflow.state_fields[name].accepts(value):
                return False, False, f"{name} has the wrong type"
        candidate = {**session.state, **updates}
        complete = step.is_complete(candidate)
        if complete and step.trigger.function == "clock__timer":
            arguments = _TimerRequest.model_validate(self._resolve(step.trigger.arguments, session.state))
            if time.time_ns() // 1_000 - arguments.started_at_us < arguments.duration_s * 1_000_000:
                return False, False, "timer has not expired"
        if complete and step.evidence is not None and session.evidence_hits < step.evidence.consecutive:
            return False, False, f"completion evidence {session.evidence_hits}/{step.evidence.consecutive}"
        changes = {name: value for name, value in updates.items() if session.state.get(name) != value}
        if not changes:
            return True, step.is_complete(session.state), "state unchanged"
        session.state.update(copy.deepcopy(changes))
        session.revision += 1
        if complete:
            session.notices.append(step.complete_message)
        elif message.strip():
            session.notices.append(message.strip())
        return True, complete, "state committed"

    async def _trigger(self, participant_id: str, step: Step, state: dict[str, Any]) -> tuple[bool, Any]:
        arguments = self._resolve(step.trigger.arguments, state)
        if step.trigger.function == "current_view":
            result = await self._current_view_tool(participant_id).execute(
                _CurrentViewRequest.model_validate(arguments)
            )
            return result.available, result.text
        if step.trigger.function == "clock__timer":
            result = await self._timer_tool().execute(_TimerRequest.model_validate(arguments))
            value: Any = result.model_dump(mode="json")
            if step.trigger.result_field is not None:
                if step.trigger.result_field not in value:
                    raise ValueError(f"timer has no result field {step.trigger.result_field!r}")
                value = value[step.trigger.result_field]
            return True, value
        raise ValueError(f"unsupported trigger: {step.trigger.function}")

    def _named_tools(self, participant_id: str, names: tuple[str, ...]) -> ToolSet:
        catalog = {
            "current_view": self._current_view_tool(participant_id),
            "clock__now": self._now_tool(),
            "clock__timer": self._timer_tool(),
        }
        return ToolSet({name: catalog[name] for name in names})

    def _current_view_tool(self, participant_id: str) -> Tool[_CurrentViewRequest, ImageQueryResult]:
        async def inspect(request: _CurrentViewRequest) -> ImageQueryResult:
            try:
                async with asyncio.timeout(self._vision_timeout_s):
                    frame = await self._current_frame.execute(CurrentFrameRequest(participant_id=participant_id))
                    return await self._image_query.execute(ImageQueryRequest(image=frame.image, query=request.question))
            except (FrameUnavailable, RuntimeError) as exc:
                return ImageQueryResult(text=f"Current frame unavailable: {exc}", available=False)
            except TimeoutError:
                return ImageQueryResult(text="Current-frame inspection timed out.", available=False)

        return Tool(
            "current_view",
            "Inspect this participant's current camera frame for one visible fact.",
            _CurrentViewRequest,
            ImageQueryResult,
            inspect,
            render_result=lambda result: result.text,
        )

    @staticmethod
    def _now_tool() -> Tool[EmptyRequest, _NowResult]:
        async def now(_request: EmptyRequest) -> _NowResult:
            return _NowResult(epoch_us=time.time_ns() // 1_000)

        return Tool("clock__now", "Return current Unix time in microseconds.", EmptyRequest, _NowResult, now)

    @staticmethod
    def _timer_tool() -> Tool[_TimerRequest, _TimerResult]:
        async def timer(request: _TimerRequest) -> _TimerResult:
            elapsed_us = max(0, time.time_ns() // 1_000 - request.started_at_us)
            duration_us = request.duration_s * 1_000_000
            return _TimerResult(
                elapsed_s=elapsed_us // 1_000_000,
                remaining_s=max(0, math.ceil((duration_us - elapsed_us) / 1_000_000)),
                expired=elapsed_us >= duration_us,
            )

        return Tool(
            "clock__timer",
            "Calculate fresh elapsed and remaining timer values.",
            _TimerRequest,
            _TimerResult,
            timer,
        )

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

    async def _publish_notices(self, session: _Session) -> None:
        async with session.lock:
            notices = tuple(session.notices)
            session.notices.clear()
            runtime = self._runtime
            if (runtime is None or not runtime.running or session.participant_id not in self._connected
                    or self._sessions.get(session.participant_id) is not session):
                return
            # Transitions clear queued notices under this same lock. Keep it
            # until publication so a drained notice cannot race a transition.
            for message in notices:
                try:
                    await runtime.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(text=message),
                        participant_id=session.participant_id,
                        source="sop-replay",
                    )
                except RuntimeClosedError:
                    return

    def _current(self, session: _Session, step: Step, revision: int) -> bool:
        return (
            session.participant_id in self._connected
            and self._sessions.get(session.participant_id) is session
            and session.step_id == step.id
            and session.revision == revision
            and not step.is_complete(session.state)
        )

    @staticmethod
    def _resolve(value: Any, state: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {name: ReplayAgent._resolve(item, state) for name, item in value.items()}
        if isinstance(value, list):
            return [ReplayAgent._resolve(item, state) for item in value]
        if isinstance(value, str) and value.startswith("$state."):
            name = value.removeprefix("$state.")
            if name not in state:
                raise ValueError(f"trigger references missing state: {name}")
            return state[name]
        return value

    @staticmethod
    async def _cancel(tasks: dict[str, asyncio.Task[None]], participant_id: str) -> None:
        task = tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    def _discard(
        tasks: dict[str, asyncio.Task[None]],
        participant_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if tasks.get(participant_id) is task:
            tasks.pop(participant_id, None)
        if task.cancelled():
            return
        with suppress(asyncio.CancelledError):
            if error := task.exception():
                logger.error("SOP task stopped pid={!r}: {!r}", participant_id, error)

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("SOP engine requires a participant")
        return participant_id


def _tools_for_query(tools: ToolSet, query: str) -> ToolSet:
    """Keep non-action speech away from mutating controls, as in the tea frontend."""
    text = " ".join(query.casefold().split()).replace("’", "'")
    if re.search(r"\b(?:record|recording|capture)\b", text):
        return ToolSet(())
    informational = re.search(
        r"\b(?:don't|do not|never|not yet|said|says|told|quoted|hypothetical)\b|"
        r"^(?:if|suppose|imagine|why|how|what|should i|would it|can i|could i)\b",
        text,
    )
    if informational:
        if not re.search(r"\b(?:look|visible|see|camera|time|timer|remaining|elapsed|status)\b", text):
            return ToolSet(())
        return ToolSet(tool for name, tool in tools.items() if name not in {
            "workflow__advance", "workflow__restart", "workflow__reset",
        })
    return tools
