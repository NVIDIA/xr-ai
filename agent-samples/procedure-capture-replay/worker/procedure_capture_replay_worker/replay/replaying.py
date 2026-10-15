# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify replay progress and own the lifetime of each participant's procedure.

The backend coordinates observations, evidence, timers, and transitions. Its
session lock keeps verification results and guidance consistent with the active
step while the agent handles conversation and runtime events.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import nemo_relay
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_hub import FrameUnavailable
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.tool_calling import ToolLoopError
from xr_ai_tools.types import EmptyRequest, StrictRequest
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryResult, ImageQueryTool

from ._workflow_spec import Step, Workflow, _CurrentViewRequest, _TimerRequest, _TimerResult
from .guides import SelectedGuide

_POLL_INTERVAL_S = 0.25


class _NowResult(BaseModel):
    epoch_us: int


class _CommitRequest(StrictRequest):
    model_config = ConfigDict(extra="forbid", strict=True)

    updates: dict[str, bool | int | float | str] = Field(default_factory=dict)
    message: str = Field(default="", max_length=240)


class _CommitResult(BaseModel):
    accepted: bool
    complete: bool
    message: str
    revision: int


class _FrameIdentity(BaseModel):
    participant_id: str
    track_id: str
    sequence: int
    timestamp_us: int


class _ViewResult(ImageQueryResult):
    frame: _FrameIdentity | None = None


@dataclass(frozen=True, slots=True)
class _Observation:
    available: bool
    value: Any
    frame: _FrameIdentity | None = None


@dataclass(frozen=True, slots=True)
class _Timer:
    started: float
    duration_s: int

    def read(self) -> _TimerResult:
        elapsed = max(0.0, time.monotonic() - self.started)
        return _TimerResult(
            elapsed_s=math.floor(elapsed),
            remaining_s=max(0, math.ceil(self.duration_s - elapsed)),
            expired=elapsed >= self.duration_s,
        )


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
    # Runtime-owned timing never comes from a model commit or a tool argument.
    timer: _Timer | None = None
    timer_error: str | None = None
    observed_frames: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)
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


class _Replay:
    """Own progress with one observer per participant and locked transitions.

    Foreground controls and the observer share each session lock. Frame checks
    may await inference outside it, but their results must match the same step
    and revision before they can change progress.
    """

    def __init__(
        self,
        *,
        guide: SelectedGuide,
        current_frame: CurrentFrameTool,
        image_query: ImageQueryTool,
        vision_timeout_s: float,
        run_model: Callable[..., Awaitable[str]],
        announce: Callable[[str, str], Awaitable[None]],
        invalidate: Callable[[str], Awaitable[None]],
        connected: Callable[[str], bool],
    ) -> None:
        if not guide.workflow.runnable:
            raise ValueError("replay requires an approved guide")
        if not math.isfinite(vision_timeout_s) or vision_timeout_s <= 0:
            raise ValueError("vision_timeout_s must be finite and positive")
        self._guide = guide
        self._current_frame = current_frame
        self._image_query = image_query
        self._vision_timeout_s = vision_timeout_s
        self._tool_loop = run_model
        self._announce = announce
        self._invalidate = invalidate
        self._connected = connected
        self._sessions: dict[str, _Session] = {}
        self._monitors: dict[str, asyncio.Task[None]] = {}

    async def leave(self, participant_id: str) -> None:
        self._sessions.pop(participant_id, None)
        await _cancel(self._monitors, participant_id)
        self._current_frame.release(participant_id)

    async def stop(self) -> None:
        participants = tuple(self._sessions)
        self._sessions.clear()
        tasks = tuple(self._monitors.values())
        self._monitors.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for participant_id in participants:
            self._current_frame.release(participant_id)

    def _start(self, participant_id: str) -> str:
        guide = self._guide
        workflow = guide.workflow
        session = _Session(
            participant_id=participant_id,
            guide=guide,
            state=workflow.initial_state(),
            step_id=workflow.start_step,
        )
        self._enter_step(session)
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
            suffix = " Complete; say next when ready." if self._is_complete(session) else ""
            return f"{session.workflow.name}, current step: {session.step.title}.{suffix}"

    async def _advance(self, participant_id: str, *, skip: bool) -> str:
        session = self._sessions.get(participant_id)
        if session is None:
            return "No guide is active."
        async with session.lock:
            step = session.step
            complete = self._is_complete(session)
            if not complete and not skip:
                return f"{step.title} is not complete yet. Say skip to move on anyway."
            # Publication only enqueues speech. Clear the delivery queue as well
            # as local notices before changing which step that speech describes.
            await self._invalidate(participant_id)
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
            self._enter_step(session)
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
            await self._invalidate(participant_id)
            self._reset_progress(session)
            return session.step.enter_message

    def _reset_progress(self, session: _Session) -> None:
        session.state = session.workflow.initial_state()
        session.step_id = session.workflow.start_step
        session.revision += 1
        session.next_tick = 0.0
        session.evidence_hits = 0
        session.notices.clear()
        session.skipped = False
        self._enter_step(session)

    def _enter_step(self, session: _Session) -> None:
        session.observed_frames.clear()
        session.timer = None
        session.timer_error = None
        if session.step.trigger.function == "clock__timer":
            try:
                arguments = _TimerRequest.model_validate(self._resolve(session.step.trigger.arguments, session.state))
                session.timer = _Timer(time.monotonic(), arguments.duration_s)
            except ValueError as exc:
                # Missing duration leaves the step blocked, never implicitly expired.
                session.timer_error = str(exc)

    def _start_monitor(self, participant_id: str) -> None:
        if participant_id in self._monitors and not self._monitors[participant_id].done():
            return
        task = asyncio.create_task(
            self._monitor(participant_id),
            name=f"sop-monitor:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._monitors[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: _discard(self._monitors, pid, completed))

    async def _monitor(self, participant_id: str) -> None:
        while self._connected(participant_id):
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
            if self._sessions.get(session.participant_id) is not session or self._is_complete(session):
                return
            now = time.monotonic()
            if session.next_tick > now:
                return
            step = session.step
            revision = session.revision
            state = dict(session.state)
            session.next_tick = now + step.trigger.interval_s
        observation = await self._trigger(session, step, state)
        run_model = True
        async with session.lock:
            if not self._current(session, step, revision):
                return
            if not observation.available:
                session.evidence_hits = 0
                return
            if step.trigger.function == "current_view":
                frame = observation.frame
                if frame is None:
                    session.evidence_hits = 0
                    return
                key = (frame.participant_id, frame.track_id)
                position = (frame.timestamp_us, frame.sequence)
                previous = session.observed_frames.get(key)
                # Re-running inference is not another observation of the action.
                if previous is not None and position <= previous:
                    return
                session.observed_frames[key] = position
            evidence = step.evidence
            if evidence is not None:
                value = observation.value if isinstance(observation.value, str) else json.dumps(
                    observation.value, separators=(",", ":"),
                )
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
        observation: _Observation,
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
                "observation": observation.value,
                "frame": observation.frame.model_dump() if observation.frame is not None else None,
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
            if session.timer is None or not session.timer.read().expired:
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

    async def _trigger(self, session: _Session, step: Step, state: dict[str, Any]) -> _Observation:
        if step.trigger.function == "current_view":
            arguments = self._resolve(step.trigger.arguments, state)
            result = await self._current_view_tool(session.participant_id).execute(
                _CurrentViewRequest.model_validate(arguments)
            )
            return _Observation(result.available, result.text, result.frame)
        if step.trigger.function == "clock__timer":
            if session.timer is None:
                return _Observation(False, session.timer_error)
            result = session.timer.read()
            value: Any = result.model_dump(mode="json")
            if step.trigger.result_field is not None:
                value = value[step.trigger.result_field]
            return _Observation(True, value)
        raise ValueError(f"unsupported trigger: {step.trigger.function}")

    def _named_tools(self, participant_id: str, names: tuple[str, ...]) -> ToolSet:
        catalog = {
            "current_view": self._current_view_tool(participant_id),
            "clock__now": self._now_tool(),
            "clock__timer": self._timer_tool(participant_id),
        }
        return ToolSet({name: catalog[name] for name in names})

    def _current_view_tool(self, participant_id: str) -> Tool[_CurrentViewRequest, _ViewResult]:
        async def inspect(request: _CurrentViewRequest) -> _ViewResult:
            try:
                async with asyncio.timeout(self._vision_timeout_s):
                    frame = await self._current_frame.execute(CurrentFrameRequest(participant_id=participant_id))
                    result = await self._image_query.execute(
                        ImageQueryRequest(image=frame.image, query=request.question),
                    )
                    return _ViewResult(
                        **result.model_dump(),
                        frame=_FrameIdentity(
                            participant_id=frame.participant_id, track_id=frame.track_id,
                            timestamp_us=frame.timestamp_us, sequence=frame.sequence,
                        ),
                    )
            except (FrameUnavailable, RuntimeError) as exc:
                return _ViewResult(text=f"Current frame unavailable: {exc}", available=False)
            except TimeoutError:
                return _ViewResult(text="Current-frame inspection timed out.", available=False)

        return Tool(
            "current_view",
            "Inspect this participant's current camera frame for one visible fact.",
            _CurrentViewRequest,
            _ViewResult,
            inspect,
            render_result=lambda result: result.text,
        )

    @staticmethod
    def _now_tool() -> Tool[EmptyRequest, _NowResult]:
        async def now(_request: EmptyRequest) -> _NowResult:
            return _NowResult(epoch_us=time.time_ns() // 1_000)

        return Tool("clock__now", "Return current Unix time in microseconds.", EmptyRequest, _NowResult, now)

    def _timer_tool(self, participant_id: str) -> Tool[EmptyRequest, _TimerResult]:
        session = self._sessions.get(participant_id)
        timer = session.timer if session is not None else None

        async def read(_request: EmptyRequest) -> _TimerResult:
            if session is None:
                raise ValueError("No active guide timer.")
            async with session.lock:
                if self._sessions.get(participant_id) is not session or timer is not session.timer:
                    raise ValueError("The active timer changed; request the current step again.")
                if timer is None:
                    raise ValueError(session.timer_error or "The current step has no timer.")
                return timer.read()

        return Tool(
            "clock__timer",
            "Read the active step's runtime-owned timer. Takes no arguments.",
            EmptyRequest, _TimerResult, read,
        )

    async def _publish_notices(self, session: _Session) -> None:
        async with session.lock:
            notices = tuple(session.notices)
            session.notices.clear()
            if not self._connected(session.participant_id) or self._sessions.get(session.participant_id) is not session:
                return
            # A transition invalidates already queued speech under this same lock.
            for message in notices:
                await self._announce(session.participant_id, message)

    def _current(self, session: _Session, step: Step, revision: int) -> bool:
        return (
            self._connected(session.participant_id)
            and self._sessions.get(session.participant_id) is session
            and session.step_id == step.id
            and session.revision == revision
            and not self._is_complete(session)
        )

    @staticmethod
    def _is_complete(session: _Session) -> bool:
        # State may be shared with preceding steps. A preexisting completion
        # field cannot authorize advancing past a timer that has not elapsed.
        if session.step.trigger.function == "clock__timer":
            if session.timer is None or not session.timer.read().expired:
                return False
        return session.step.is_complete(session.state)

    @staticmethod
    def _resolve(value: Any, state: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {name: _Replay._resolve(item, state) for name, item in value.items()}
        if isinstance(value, list):
            return [_Replay._resolve(item, state) for item in value]
        if isinstance(value, str) and value.startswith("$state."):
            name = value.removeprefix("$state.")
            if name not in state:
                raise ValueError(f"trigger references missing state: {name}")
            return state[name]
        return value


async def _cancel(tasks: dict[str, asyncio.Task[None]], participant_id: str) -> None:
    task = tasks.pop(participant_id, None)
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


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
