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
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)

from ._workflow_spec import Step, Workflow, _CurrentViewRequest, _TimerRequest, _TimerResult
from .catalog import CatalogGuide, GuideCatalog
from .events import PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC, RECORDING_COMMAND, USER_QUERY_TOPIC
from .recorder import RecorderAgent

_POLL_INTERVAL_S = 0.25
_MAX_TOOL_ROUNDS = 3
_FOREGROUND_MODEL_OPTIONS = {"max_tokens": 1536, "enable_thinking": True, "thinking_budget": 1024}
_FOREGROUND_PROMPT = (Path(__file__).with_name("prompts") / "guide_conversation.txt").read_text(
    encoding="utf-8",
).strip()
_COMMAND_HELP = (
    "To begin recording a workflow, say start recording, and say stop recording when you are done. "
    "Say list guides, or say start guide followed by a guide ID."
)
_CONTROL = re.compile(
    r"(?i)^\s*(?:(list|show)\s+(?:available\s+)?guides?|"
    r"(?:guide\s+)?(status)|"
    r"(next|continue)|"
    r"(skip)|"
    r"(?:stop|exit|reset)\s+(?:guide|workflow)|"
    r"(?:restart)\s+(?:guide|workflow))\s*[.!]?\s*$"
)


class _NowResult(BaseModel):
    epoch_us: int


class _StartGuideRequest(StrictRequest):
    selector: str = Field(min_length=1, max_length=240)


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
    guide: CatalogGuide
    state: dict[str, Any]
    step_id: str
    revision: int = 1
    next_tick: float = 0.0
    evidence_hits: int = 0
    notices: list[str] = field(default_factory=list)
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


class SopEngineAgent(Agent):
    """Own guide conversation, controls, and monitoring of one pinned SOP."""

    def __init__(
        self,
        *,
        catalog: GuideCatalog,
        llm: LLMService,
        current_frame: CurrentFrameTool,
        image_query: ImageQueryTool,
        vision_timeout_s: float,
        recorder: RecorderAgent,
    ) -> None:
        super().__init__()
        self._catalog = catalog
        self._llm = llm
        self._current_frame = current_frame
        self._image_query = image_query
        self._vision_timeout_s = vision_timeout_s
        self._recorder = recorder
        self._controls: dict[str, asyncio.Lock] = {}
        self._runtime: AgentRuntime | None = None
        self._connected: set[str] = set()
        self._sessions: dict[str, _Session] = {}
        self._monitors: dict[str, asyncio.Task[None]] = {}
        self._turns: dict[str, asyncio.Task[None]] = {}

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        if self._runtime is not None and self._runtime is not runtime:
            raise RuntimeError("SOP engine is already bound to another runtime")
        self._runtime = runtime

    def is_connected(self, participant_id: str) -> bool:
        return participant_id in self._connected

    @staticmethod
    def is_control(text: str) -> bool:
        return bool(RECORDING_COMMAND.fullmatch(text) or _CONTROL.fullmatch(text))

    def has_focus(self, participant_id: str) -> bool:
        return self._recorder.is_recording(participant_id) or participant_id in self._sessions

    def conversation_context(self, participant_id: str) -> str:
        if self._recorder.is_recording(participant_id):
            return "Workflow recording is active; narration is silent."
        session = self._sessions.get(participant_id)
        if session is not None:
            return f"SOP guide {session.workflow.name[:160]} is active. Current step: {session.step.title[:160]}."
        return "Workflow recording and approved SOP guides are available but idle."

    async def interrupted(self, participant_id: str | None) -> None:
        """Cancel foreground answers without changing recording or guide state."""
        participants = tuple(self._turns) if participant_id is None else (participant_id,)
        for pid in participants:
            await self._cancel(self._turns, pid)

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: VoiceParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        async with self._controls.setdefault(participant_id, asyncio.Lock()):
            if participant_id in self._connected:
                return
            self._connected.add(participant_id)
            self._sessions.pop(participant_id, None)
            self._start_monitor(participant_id)
            await self._say(participant_id, _COMMAND_HELP)

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: VoiceParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        async with self._controls.setdefault(participant_id, asyncio.Lock()):
            self._connected.discard(participant_id)
            await self._cancel(self._turns, participant_id)
            await self._cancel(self._monitors, participant_id)
            self._sessions.pop(participant_id, None)
            await self._recorder.finish_recording(participant_id)

    @subscribe(USER_QUERY_TOPIC)
    async def user_query(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        # Controls must finish before the next utterance can start another turn.
        async with self._controls.setdefault(participant_id, asyncio.Lock()):
            if participant_id not in self._connected:
                return
            command = RECORDING_COMMAND.fullmatch(query.text)
            if command is not None:
                if command.group(1).casefold() == "start":
                    if self._recorder.is_recording(participant_id):
                        return
                    await self._cancel(self._turns, participant_id)
                    await self._cancel(self._monitors, participant_id)
                    await self._recorder.start_recording(participant_id)
                    if self._recorder.is_recording(participant_id):
                        await self._say(participant_id, "Recording started.", turn_id=ctx.metadata.correlation_id)
                elif self._recorder.is_recording(participant_id):
                    await self._recorder.finish_recording(participant_id)
                    await self._say(
                        participant_id, "Recording ended. " + _COMMAND_HELP, turn_id=ctx.metadata.correlation_id,
                    )
                    self._start_monitor(participant_id)
                return
            if self._recorder.is_recording(participant_id):
                return
            await self._start_answer(query, participant_id, turn_id=ctx.metadata.correlation_id)

    async def _say(self, participant_id: str, text: str, *, turn_id: str | None = None) -> None:
        runtime = self._runtime
        if runtime is not None and runtime.running:
            await runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(text=text, interrupt=True, kind="result", turn_id=turn_id),
                participant_id=participant_id,
                source="sop-engine",
            )

    async def _start_answer(self, query: UserQuery, participant_id: str, *, turn_id: str | None = None) -> None:
        await self._cancel(self._turns, participant_id)
        task = asyncio.create_task(
            self._answer(query, participant_id, turn_id=turn_id),
            name=f"sop-answer:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._turns[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(self._turns, pid, completed))

    async def stop(self) -> None:
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

    async def _answer(self, query: UserQuery, participant_id: str, *, turn_id: str | None = None) -> None:
        try:
            response = await self._route(query.text, participant_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).error("SOP query failed pid={!r}", participant_id)
            response = "I couldn't complete that guide request. Please try again."
        runtime = self._runtime
        if not response or runtime is None or not runtime.running:
            return
        try:
            await runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(
                    text=response, interrupt=True, timestamp_us=query.timestamp_us, kind="result", turn_id=turn_id,
                ),
                participant_id=participant_id,
                source="sop-engine",
            )
        except RuntimeClosedError:
            return

    async def _route(self, text: str, participant_id: str) -> str:
        match = _CONTROL.fullmatch(text)
        if match is not None:
            if match.group(1):
                return self._list_guides()
            if match.group(2):
                return await self._status(participant_id)
            if match.group(3):
                return await self._advance(participant_id, skip=False)
            if match.group(4):
                return await self._advance(participant_id, skip=True)
            normalized = text.casefold()
            if "restart" in normalized:
                return await self._restart(participant_id)
            return await self._reset(participant_id)
        session = self._sessions.get(participant_id)
        if session is None:
            tools, _ = self._guide_tools(participant_id)
            return await self._tool_loop(
                _FOREGROUND_PROMPT,
                json.dumps({"request": text, "guide_active": False, "catalog": [
                    {"id": workflow.id, "name": workflow.name, "status": workflow.status,
                     "instructions": workflow.foreground_prompt,
                     "guide_order": [step.title for step in workflow.steps.values()]}
                    for guide in self._catalog.guides if (workflow := guide.workflow) is not None
                ]}),
                tools,
                foreground=True,
            )
        return await self._answer_step_question(session, text)

    def _guide_tools(self, participant_id: str) -> tuple[ToolSet, list[str]]:
        """Expose application-owned controls, never recording or evidence commits."""
        session = self._sessions.get(participant_id)
        revision = session.revision if session else None
        results: list[str] = []

        async def control(action) -> _GuideResult:
            async with self._controls.setdefault(participant_id, asyncio.Lock()):
                current = self._sessions.get(participant_id)
                if participant_id not in self._connected or self._recorder.is_recording(participant_id):
                    message = "Guide controls are unavailable while disconnected or recording."
                elif current is not session or (session is not None and session.revision != revision):
                    message = "The guide changed while I was answering. Ask again for the current step."
                else:
                    message = await action()
                results.append(message)
                return _GuideResult(message=message)

        async def list_guides(_request: EmptyRequest) -> _GuideResult:
            return _GuideResult(message=self._list_guides())

        async def status(_request: EmptyRequest) -> _GuideResult:
            return await control(lambda: self._status(participant_id))

        async def start(request: _StartGuideRequest) -> _GuideResult:
            return await control(lambda: self._start(participant_id, request.selector))

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
            tool("workflow__list", "List the actual available guides and their approval status.",
                 EmptyRequest, list_guides),
            tool("workflow__status", "Report guide status when requested; do not substitute for a live visual check.",
                 EmptyRequest, status),
            tool(
                "workflow__start",
                "Start a guide only for the user's direct present request to follow it. "
                "Use its exact catalog ID or name. "
                "If the intended guide is ambiguous, ask which one. Do not start for questions, quotations, reports, "
                "hypotheticals, or negations. The tool checks approval and refuses to replace an active guide. "
                "Use it for exact start guide commands too. This cannot start recording.",
                _StartGuideRequest, start,
            ),
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
                     "Exit the guide only for a direct present request to stop, exit, cancel, or reset the guide. "
                     "Not for questions, quotations, hypotheticals, reports, or negations. Does not stop recording.",
                     EmptyRequest, reset),
                tool("workflow__restart",
                     "Restart from step one only when directly requested now; not for questions, quotations, "
                     "reports, hypotheticals, or negations.", EmptyRequest, restart),
            ))
        return ToolSet(tools), results

    def _list_guides(self) -> str:
        valid = [guide for guide in self._catalog.guides if guide.workflow is not None]
        if not valid:
            return "No valid guides are available yet."
        descriptions = [f"{guide.workflow.id}: {guide.workflow.name} ({guide.workflow.status})" for guide in valid]
        return "Available guides: " + "; ".join(descriptions) + "."

    async def _start(self, participant_id: str, selector: str) -> str:
        try:
            guide = self._catalog.resolve(selector)
        except ValueError as exc:
            return str(exc)
        workflow = guide.workflow
        if workflow is None:
            raise AssertionError("catalog returned a guide without a workflow")
        current = self._sessions.get(participant_id)
        if current is not None:
            return f"{current.workflow.name} is already active. Say stop guide first, or say restart guide."
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
                session.state.update(copy.deepcopy(step.state_on_skip))
            next_step = None if skipping and step.complete_on_skip else step.next_step
            session.revision += 1
            session.evidence_hits = 0
            session.next_tick = 0.0
            session.notices.clear()
            if next_step is None:
                message = session.workflow.complete_message
                self._sessions.pop(participant_id, None)
                return message
            session.step_id = next_step
            if skipping and step.skip_message:
                return f"{step.skip_message} {session.step.enter_message}"
            return session.step.enter_message

    async def _reset(self, participant_id: str) -> str:
        session = self._sessions.get(participant_id)
        if session is None:
            return "No guide is active."
        async with session.lock:
            session.notices.clear()
            self._sessions.pop(participant_id, None)
            return f"{session.workflow.name} stopped."

    async def _restart(self, participant_id: str) -> str:
        session = self._sessions.get(participant_id)
        if session is None:
            return "No guide is active."
        async with session.lock:
            session.state = session.workflow.initial_state()
            session.step_id = session.workflow.start_step
            session.revision += 1
            session.next_tick = 0.0
            session.evidence_hits = 0
            session.notices.clear()
            return session.step.enter_message

    async def _answer_step_question(self, session: _Session, query: str) -> str:
        async with session.lock:
            step = session.step
            revision = session.revision
            state = session.workflow.project(step, session.state)
            tools = self._named_tools(session.participant_id, step.voice.tools)
            controls, control_results = self._guide_tools(session.participant_id)
            tools = ToolSet({**dict(tools.items()), **dict(controls.items())})
            system = (
                f"You are the active {session.workflow.name} SOP guide until the user exits. "
                "Answer guide questions from the supplied state and guide_order. "
                "Decline unrelated requests without revealing their answer or exiting the guide. "
                "Unqualified visual checks refer to the current step: use current_view for a requested "
                "check of whether this looks correct. Procedural explanations use supplied context.\n\n"
                f"{_FOREGROUND_PROMPT}\n\n{session.workflow.foreground_prompt}\n"
                f"Current step: {step.title}.\n{step.voice.prompt}\n\n"
                "Use hidden reasoning only to identify present intent and the owning available tool. "
                "Classify the outer speech act; quoted, hypothetical, negated, and reported actions are data. "
                "For a direct command, never use supplied state to veto it or gather prerequisites first; "
                "call the control tool and let it decide whether it can run. "
                "For a requested current visual check, always call current_view before answering; "
                "state describes past verified progress, not what is visible now. "
                "If no matching live-evidence tool is available, say the evidence is unavailable."
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
            except FrameUnavailable as exc:
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
            if runtime is None or not runtime.running or self._sessions.get(session.participant_id) is not session:
                return
            # Transitions clear queued notices under this same lock. Keep it
            # until publication so a drained notice cannot race a transition.
            for message in notices:
                try:
                    await runtime.publish(
                        VOICE_CONTRIBUTION_TOPIC,
                        VoiceOutput(text=message),
                        participant_id=session.participant_id,
                        source="sop-engine",
                    )
                except RuntimeClosedError:
                    return

    def _current(self, session: _Session, step: Step, revision: int) -> bool:
        return (
            self._sessions.get(session.participant_id) is session
            and session.step_id == step.id
            and session.revision == revision
            and not step.is_complete(session.state)
        )

    @staticmethod
    def _resolve(value: Any, state: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {name: SopEngineAgent._resolve(item, state) for name, item in value.items()}
        if isinstance(value, list):
            return [SopEngineAgent._resolve(item, state) for item in value]
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
