# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run each participant's procedure and verify when its steps are complete.

The guide backend tracks the active step, checks camera evidence or elapsed
time, and applies the guide's completion rules. It handles starting, stopping,
and advancing a run, and supplies current progress and completion notices to
GuideAgent in agent.py. Conversation handling remains in the agent.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import math
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_hub import FrameUnavailable
from xr_ai_models import DecisionService
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.tool_calling import ToolLoopError
from xr_ai_tools.types import EmptyRequest, StrictRequest
from xr_ai_tools.vision import ImageQueryResult, ImageQueryTool

from ._workflow_spec import Step, Workflow, _CurrentViewRequest, _TimerRequest
from .flow import FlowUpdate
from .guides import SelectedGuide
from .policy import load_policy
from .run_trace import RunTrace
from .step_runner import StepRunner
from .timer_flow import TimerFlow
from .visual_check import VisualCheckTrace, caption_image, verify_image
from .visual_verification_flow import (
    FrameIdentity as _FrameIdentity,
)
from .visual_verification_flow import (
    Observation as _Observation,
)
from .visual_verification_flow import (
    VisualProgress,
    VisualVerificationFlow,
)


class _AdvanceGuideRequest(StrictRequest):
    skip: bool = Field(default=False, strict=True)


class _GuideResult(BaseModel):
    message: str


class _ProgressChanged(Exception):
    """A read-only answer needs a fresh snapshot of the same step's progress."""


class _FinishAnswer(Protocol):
    async def __call__(self, answer: str, *, refresh: bool = True) -> str: ...


@dataclass(frozen=True, slots=True)
class _GuidanceTurn:
    """Detached foreground context and tools scoped to the same guide revision."""

    # Progress can change within this session/control scope; transitions cannot.
    execution: tuple[int, int]
    instructions: str
    context: dict[str, Any]
    tools: ToolSet
    finish: _FinishAnswer
    cancel: Callable[[], Awaitable[None]]


class _TimerStatus(BaseModel):
    available: bool
    unavailable_reason: str | None = None
    elapsed_s: int | None = None
    remaining_s: int | None = None
    expired: bool | None = None


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


class _ViewResult(ImageQueryResult):
    frame: _FrameIdentity | None = None


@dataclass(slots=True)
class _Session:
    participant_id: str
    guide: SelectedGuide
    state: dict[str, Any]
    step_id: str
    active: bool = True
    # State changes expire observations and answers; only transitions expire
    # controls so a concurrently completed step can still be advanced or stopped.
    revision: int = 1
    control_revision: int = 1
    evidence_hits: int = 0
    notices: list[str] = field(default_factory=list)
    skipped: bool = False
    run_id: str | None = None
    last_verification: dict | None = None
    observation_history: list[dict] = field(default_factory=list)
    completed_steps: list[dict] = field(default_factory=list)
    previous_run: dict | None = None
    # Runtime-owned timing never comes from a model commit or a tool argument.
    timer: TimerFlow | None = None
    timer_error: str | None = None
    # Each step gets a fresh runner; its flow owns deduplication/confirmation.
    # timer references that same flow so foreground reads never start another wait.
    runner: StepRunner | None = None
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


class _Guide:
    """Own guide progress with one step runner per participant and locked transitions.

    Foreground controls and flow result application share each session lock. Frame checks
    may await inference outside it, but their results must match the same step
    and revision before they can change progress.
    """

    def __init__(
        self,
        *,
        guide: SelectedGuide,
        decisions: DecisionService,
        current_frame: CurrentFrameTool,
        image_query: ImageQueryTool,
        vision_timeout_s: float,
        run_model: Callable[..., Awaitable[str]],
        announce: Callable[[str, str], Awaitable[None]],
        invalidate: Callable[[str], Awaitable[None]],
        connected: Callable[[str], bool],
        decision_timeout_s: float = 15.0,
        measurement_query: ImageQueryTool | None = None,
        traces_dir: Path | None = None,
        policy=None,
    ) -> None:
        if not guide.workflow.runnable:
            raise ValueError("guide mode requires an approved guide")
        if not math.isfinite(vision_timeout_s) or vision_timeout_s <= 0:
            raise ValueError("vision_timeout_s must be finite and positive")
        if not math.isfinite(decision_timeout_s) or decision_timeout_s <= 0:
            raise ValueError("decision_timeout_s must be finite and positive")
        self._guide = guide
        self._decisions = decisions
        self._decision_timeout_s = decision_timeout_s
        self._current_frame = current_frame
        self._image_query = image_query
        self._measurement_query = measurement_query
        self._policy = policy or load_policy()
        self._trace = RunTrace(traces_dir, guide)
        self._vision_timeout_s = vision_timeout_s
        self._tool_loop = run_model
        self._announce = announce
        self._invalidate = invalidate
        self._connected = connected
        self._sessions: dict[str, _Session] = {}
        self._runners: dict[str, StepRunner] = {}
        self._control_tasks: dict[str, set[asyncio.Task[tuple[str, int]]]] = {}

    def join(self, participant_id: str) -> str:
        """Load the selected guide without beginning observations or timers."""
        self._start(participant_id, active=False)
        return f"{self._guide.workflow.name} is ready. Say start guide to begin."

    async def guidance(self, participant_id: str) -> _GuidanceTurn | None:
        """Capture policy, state, and controls at one execution revision.

        The foreground receives detached values, not a mutable session or its
        lock. Controls and answer acceptance recheck this scope inside the
        backend after model inference.
        """
        await self._drain_controls(participant_id)
        session = self._sessions.get(participant_id)
        if session is None:
            return None
        async with session.lock:
            if not self._connected(participant_id) or self._sessions.get(participant_id) is not session:
                return None
            revision = session.revision
            control_revision = session.control_revision
            workflow = session.workflow
            step = session.step
            active_trigger: dict[str, Any] | None = None
            if session.active and step.trigger.function == "clock__timer":
                active_trigger = {"function": step.trigger.function}
                if session.timer is None:
                    active_trigger["unavailable_reason"] = session.timer_error or "No active timer"
                else:
                    active_trigger["duration_s"] = session.timer.duration_s
                    active_trigger["status"] = session.timer.read().model_dump()
            guide_order = []
            step_id = workflow.start_step
            while step_id is not None:
                ordered_step = workflow.steps[step_id]
                guide_order.append(ordered_step.title)
                step_id = ordered_step.next_step
            results: list[tuple[str, int]] = []
            controls = self._guide_tools(session, control_revision, results)
            named = self._named_tools(participant_id, step.voice.tools) if session.active else ToolSet([])
            tools = ToolSet({**dict(named.items()), **dict(controls.items())})
            instructions = f"Selected guide: {workflow.name}.\n{workflow.foreground_prompt}\n"
            if session.active:
                instructions += f"Current step: {step.title}.\n{step.voice.prompt}"
            else:
                instructions += (
                    "The selected guide is inactive. There is no active step or verification. "
                    "Say start guide to begin."
                )
            evidence = session.last_verification
            if evidence is not None:
                evidence = {**evidence, "age_s": max(0.0, time.monotonic() - evidence["observed_monotonic"])}
                evidence.pop("observed_monotonic", None)
            context = {
                "active": session.active,
                "finish_on_complete": step.finish_on_complete if session.active else False,
                "guide_order": guide_order,
                "state": copy.deepcopy(workflow.project(step, session.state)) if session.active else {},
                "active_trigger": active_trigger,
                "execution_status": (session.runner.status if session.active and session.runner is not None
                                     else "blocked" if session.active else "inactive"),
                "last_verification": copy.deepcopy(evidence) if session.active else None,
                "run_id": session.run_id,
                "step_id": step.id if session.active else None,
                "step_revision": session.revision,
                "recent_observations": [_evidence_context(item) for item in session.observation_history],
                "completed_steps": copy.deepcopy(session.completed_steps),
                "historical_previous_run": copy.deepcopy(session.previous_run),
            }

        async def finish(answer: str, *, refresh: bool = True) -> str:
            async with session.lock:
                if not self._connected(participant_id):
                    return ""
                if self._sessions.get(participant_id) is not session:
                    return "The guide changed while I was answering. Ask again for the current step."
                expected_control = results[-1][1] if results else control_revision
                if session.control_revision != expected_control:
                    return "The guide changed while I was answering. Ask again for the current step."
                if not results and session.revision != revision:
                    # Verification updates progress without changing the selected
                    # step. Discard the stale answer, not the user's question.
                    if refresh:
                        raise _ProgressChanged
                    return self._status_message(session)
                return results[-1][0] if results else answer

        async def cancel() -> None:
            # Once accepted, a transition and its guidance belong to execution,
            # even if the model turn waiting for the tool result is interrupted.
            await self._drain_controls(participant_id)
            async with session.lock:
                if (results and results[-1][1] != control_revision
                        and session.control_revision == results[-1][1]
                        and self._sessions.get(participant_id) is session
                        and self._connected(participant_id)):
                    await self._announce(participant_id, results[-1][0])

        return _GuidanceTurn((id(session), control_revision), instructions, context, tools, finish, cancel)

    async def leave(self, participant_id: str) -> None:
        await self._drain_controls(participant_id)
        session = self._sessions.pop(participant_id, None)
        await _cancel(self._runners, participant_id)
        if session is not None:
            self._end_trace(session, "disconnect")
        self._current_frame.release(participant_id)

    async def stop(self) -> None:
        for participant_id in tuple(self._control_tasks):
            await self._drain_controls(participant_id)
        participants = tuple(self._sessions)
        sessions = tuple(self._sessions.values())
        self._sessions.clear()
        runners = tuple(self._runners.values())
        self._runners.clear()
        if runners:
            await asyncio.gather(*(runner.stop() for runner in runners))
        for session in sessions:
            self._end_trace(session, "shutdown")
        await self._trace.close()
        for participant_id in participants:
            self._current_frame.release(participant_id)

    def _start(self, participant_id: str, *, active: bool = True) -> str:
        guide = self._guide
        workflow = guide.workflow
        session = _Session(
            participant_id=participant_id,
            guide=guide,
            state=workflow.initial_state(),
            step_id=workflow.start_step,
            active=active,
        )
        if active:
            self._begin_trace(session)
        self._enter_step(session)
        self._sessions[participant_id] = session
        logger.info(
            "SOP loaded pid={!r} guide={} version={} sha256={}",
            participant_id,
            workflow.id,
            workflow.version,
            guide.sha256,
        )
        return session.step.enter_message

    def _guide_tools(self, session: _Session, revision: int, results: list[tuple[str, int]]) -> ToolSet:
        async def control(action: str, *, skip: bool = False) -> _GuideResult:
            # A new utterance can cancel inference, but not half of an accepted
            # transition. Retain the operation until execution cleanup drains it.
            async def execute() -> tuple[str, int]:
                result = await self._control(session, revision, action, skip=skip)
                if action != "status":
                    results.append(result)
                return result

            task = asyncio.create_task(execute())
            pending = self._control_tasks.setdefault(session.participant_id, set())
            pending.add(task)

            def finished(completed: asyncio.Task[tuple[str, int]]) -> None:
                pending.discard(completed)
                if not completed.cancelled() and (error := completed.exception()):
                    logger.error("Guide control failed pid={!r}: {!r}", session.participant_id, error)

            task.add_done_callback(finished)
            message, _ = await asyncio.shield(task)
            return _GuideResult(message=message)

        async def status(_request: EmptyRequest) -> _GuideResult:
            return await control("status")

        async def advance(request: _AdvanceGuideRequest) -> _GuideResult:
            return await control("advance", skip=request.skip)

        async def reset(_request: EmptyRequest) -> _GuideResult:
            return await control("reset")

        async def restart(_request: EmptyRequest) -> _GuideResult:
            return await control("restart")

        def tool(name, description, request_type, handler):
            return Tool(
                name, description, request_type, _GuideResult, handler,
                render_result=lambda result: result.message, return_direct=True,
            )

        tools = [
            tool("workflow__status", "Report whether the guide is active and its current progress.",
                 EmptyRequest, status),
            tool("workflow__reset",
                 "Cancel the current run and clear progress only for a direct present request "
                 "to stop guide, stop, reset, or cancel. "
                 "The selected guide stays loaded but inactive; this does not restart it. "
                 "Polite action requests count; explanations, quotations, hypotheticals and negations do not.",
                 EmptyRequest, reset),
            tool("workflow__restart",
                 "Begin a fresh run of the command-line-selected guide from step one "
                 "for start guide or a direct restart request. "
                 "Works whether the guide is active or stopped. "
                 "Polite action requests count; explanations, quotations, hypotheticals and negations do not.",
                 EmptyRequest, restart),
        ]
        if session.active:
            tools.append(tool(
                "workflow__advance",
                "USE WHEN: the user directly commands the active guide to move forward now; "
                "always call even when supplied state looks incomplete because the tool alone decides readiness. "
                "Set skip=false for continue/next/move-on/proceed commands and "
                "skip=true only for bypass/skip-this-step requests. Polite requests to act count too. "
                "Do not act on explanations, quotations, hypotheticals, deliberation or negated requests.",
                _AdvanceGuideRequest, advance,
            ))
        return ToolSet(tools)

    async def _drain_controls(self, participant_id: str) -> None:
        pending = tuple(self._control_tasks.get(participant_id, ()))
        if pending:
            await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))
        self._control_tasks.pop(participant_id, None)

    async def _control(
        self, session: _Session, revision: int, action: str, *, skip: bool = False,
    ) -> tuple[str, int]:
        async with session.lock:
            previous_run = session.run_id
            self._trace.emit(previous_run, "control", action=action, skip=skip,
                             step=session.step_id, revision=revision)
            message = await self._apply_control(session, revision, action, skip=skip)
            self._trace.emit(previous_run or session.run_id, "control_result", action=action, message=message,
                             next_run_id=session.run_id,
                             step=session.step_id, revision=session.revision, state=dict(session.state))
            return message, session.control_revision

    async def _apply_control(self, session: _Session, revision: int, action: str, *, skip: bool) -> str:
        pid = session.participant_id
        if not self._connected(pid):
            return "Guide controls are unavailable while disconnected."
        # Observations may complete the same step during foreground inference.
        # Controls inspect that fresh state; only a run/step transition expires them.
        if self._sessions.get(pid) is not session or session.control_revision != revision:
            return "The guide changed while I was answering. Ask again for the current step."
        if action == "status":
            return self._status_message(session)
        if action == "advance":
            if not session.active:
                return "Guide is inactive. Say start guide to begin."
            return await self._advance(session, skip=skip)
        if action not in {"reset", "restart"}:
            raise ValueError(f"unknown guide control: {action}")
        await self._invalidate(pid)
        await _cancel(self._runners, pid)
        self._reset_progress(session, active=action == "restart", reason=action)
        if session.active:
            self._start_step(pid)
            return session.step.enter_message
        return "Guide stopped. Say start guide to begin again."

    def _status_message(self, session: _Session) -> str:
        if not session.active:
            return f"{session.workflow.name} is inactive. Say start guide to begin."
        suffix = " Complete; say next when ready." if self._is_complete(session) else ""
        return f"{session.workflow.name}, current step: {session.step.title}.{suffix}"

    async def _advance(self, session: _Session, *, skip: bool) -> str:
        step = session.step
        complete = self._is_complete(session)
        if not complete and not skip:
            return f"{step.title} is not complete yet. Say skip to move on anyway."
        # Publication only enqueues speech. Clear the delivery queue as well
        # as local notices before changing which step that speech describes.
        await self._invalidate(session.participant_id)
        skipping = skip and not complete
        if skipping:
            session.skipped = True
            session.state.update(copy.deepcopy(step.state_on_skip))
        session.completed_steps.append({"step_id": step.id, "outcome": "skipped" if skipping else "completed",
                                        "verification": (_historical_evidence(session.last_verification)
                                                         if session.last_verification else None)})
        next_step = None if skipping and step.complete_on_skip else step.next_step
        session.revision += 1
        session.control_revision += 1
        session.evidence_hits = 0
        session.notices.clear()
        if next_step is None:
            message = "Guide ended with skipped steps." if session.skipped else session.workflow.complete_message
            self._reset_progress(session, active=False, reason="skipped" if session.skipped else "completed")
            await _cancel(self._runners, session.participant_id)
            return f"{message} Say start guide to begin again."
        await _cancel(self._runners, session.participant_id)
        session.step_id = next_step
        session.last_verification = None
        self._enter_step(session)
        self._start_step(session.participant_id)
        if skipping and step.skip_message:
            return f"{step.skip_message} {session.step.enter_message}"
        return session.step.enter_message

    def _reset_progress(self, session: _Session, *, active: bool = True, reason: str = "reset") -> None:
        if session.run_id is not None:
            session.previous_run = {"run_id": session.run_id, "outcome": reason,
                                    "steps": copy.deepcopy(session.completed_steps)}
        self._end_trace(session, reason)
        session.completed_steps.clear()
        session.last_verification = None
        session.active = active
        session.state = session.workflow.initial_state()
        session.step_id = session.workflow.start_step
        session.revision += 1
        session.control_revision += 1
        session.evidence_hits = 0
        session.notices.clear()
        session.skipped = False
        if active:
            self._begin_trace(session)
        self._enter_step(session)

    def record_event(self, participant_id: str, kind: str, *, run_id: str | None = None, **data) -> str | None:
        session = self._sessions.get(participant_id)
        target = run_id or (session.run_id if session else None)
        if target is not None:
            self._trace.emit(target, kind, **data)
        return target

    def _begin_trace(self, session: _Session) -> None:
        session.run_id = uuid.uuid4().hex
        self._trace.emit(session.run_id, "run_started", participant_id=session.participant_id,
                         guide_sha256=session.guide.sha256, state=dict(session.state),
                         policy=self._policy.model_dump())

    def _end_trace(self, session: _Session, reason: str) -> None:
        self._trace.emit(session.run_id, "run_ended", reason=reason, state=dict(session.state),
                         step=session.step_id, skipped=session.skipped)
        session.run_id = None

    def _enter_step(self, session: _Session) -> None:
        session.observation_history.clear()
        session.last_verification = None
        self._trace.emit(session.run_id, "step_entered", step=session.step_id,
                         revision=session.revision, state=dict(session.state))
        session.runner = None
        session.timer = None
        session.timer_error = None
        if session.active and session.step.trigger.function == "clock__timer":
            try:
                arguments = _TimerRequest.model_validate(self._resolve(session.step.trigger.arguments, session.state))
                session.timer = TimerFlow(arguments.duration_s, session.step.trigger.interval_s, clock=time.monotonic)
            except ValueError as exc:
                # Missing duration leaves the step blocked, never implicitly expired.
                session.timer_error = str(exc)

        if session.active:
            self._prepare_runner(session)

    def _prepare_runner(self, session: _Session) -> None:
        step = session.step
        revision = session.revision

        async def acquire() -> _Observation:
            nonlocal revision
            async with session.lock:
                revision = session.revision
                state = dict(session.state)
            return await self._trigger(session, step, state)

        async def verify(observation: _Observation) -> str | None:
            trace = await self._verify_visual(session, step, revision, observation)
            async with session.lock:
                if session.runner is not runner or not self._current(session, step, revision):
                    return None
                return trace.verdict if trace is not None and trace.error is None else None

        async def apply(update: FlowUpdate) -> bool:
            if isinstance(update.value, VisualProgress):
                async with session.lock:
                    if session.runner is not runner or not session.active:
                        return True
                    if not self._current(session, step, revision):
                        return False
                    session.evidence_hits = update.value.hits
                    if not update.value.observation.available:
                        self._unavailable(session, step, revision, "current observation unavailable")
                    if update.status == "ready" and step.evidence.commit:
                        self._commit(session, step, step.evidence.commit, "")
            else:
                await self._apply_timer(session, step, runner, update.value)
            await self._publish_notices(session)
            return session.runner is not runner or not session.active or self._is_complete(session)

        if step.trigger.function == "current_view":
            flow = VisualVerificationFlow(
                acquire=acquire, verify=verify, interval_s=step.trigger.interval_s,
                required=step.evidence.consecutive, confirmation=step.evidence.confirmation,
                clock=time.monotonic,
            )
        else:
            if session.timer is None:
                return  # An unresolved duration leaves the step explicitly blocked.
            flow = session.timer
        runner = StepRunner(flow, apply=apply, clock=time.monotonic)
        session.runner = runner

    def _start_step(self, participant_id: str) -> None:
        session = self._sessions.get(participant_id)
        if session is None or not session.active or session.runner is None:
            return
        runner = session.runner
        self._runners[participant_id] = runner
        task = runner.start(name=f"guide-step:{participant_id}:{session.step_id}",
                            poll=lambda: self._tick(session))
        task.add_done_callback(lambda completed: _discard(self._runners, participant_id, runner, completed))

    async def _tick(self, session: _Session) -> None:
        """Advance the active runner once, preserving evaluation event attribution."""
        async with session.lock:
            if (not session.active or self._sessions.get(session.participant_id) is not session
                    or self._is_complete(session) or session.runner is None):
                return
            runner, step, revision = session.runner, session.step, session.revision
        try:
            await runner.poll()
        except Exception as exc:
            async with session.lock:
                if session.runner is runner and self._current(session, step, revision):
                    session.evidence_hits = 0
                    self._unavailable(session, step, revision, {"type": type(exc).__name__, "message": str(exc)})
            raise

    def _unavailable(self, session: _Session, step: Step, revision: int, error) -> None:
        session.last_verification = {
            "step_id": step.id, "revision": revision, "verdict": None,
            "observed_monotonic": time.monotonic(), "error": error,
        }
        self._trace.emit(session.run_id, "observation_unavailable", step=step.id, revision=revision, error=error)

    async def _apply_timer(self, session: _Session, step: Step, runner: StepRunner, result) -> None:
        async with session.lock:
            if session.runner is not runner or not self._current(session, step, session.revision):
                return
            revision = session.revision
            value = result.model_dump(mode="json")
            if step.trigger.result_field is not None:
                value = value[step.trigger.result_field]
            observation = _Observation(True, value)
            self._trace.emit(session.run_id, "timer_observation", step=step.id, revision=revision,
                             result=value)
            run_model = True
            evidence = step.evidence
            if evidence is not None:
                text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
                matched = re.fullmatch(evidence.pattern, text.strip()) is not None
                session.evidence_hits = session.evidence_hits + 1 if matched else 0
                if evidence.commit:
                    run_model = False
                    if session.evidence_hits >= evidence.consecutive:
                        self._commit(session, step, evidence.commit, "")
            state = dict(session.state)
        if run_model:
            await self._observation_turn(session, step, revision, state, observation)

    async def _verify_visual(
        self, session: _Session, step: Step, revision: int, observation: _Observation,
    ) -> VisualCheckTrace | None:
        assert step.evidence is not None and observation.frame is not None
        async with session.lock:
            if not self._current(session, step, revision):
                return None
            session.evidence_hits = 0
            session.last_verification = None
            run_id = session.run_id
        trace = VisualCheckTrace()
        error = None
        before = time.monotonic()
        question = _CurrentViewRequest.model_validate(self._resolve(step.trigger.arguments, session.state)).question
        frame = observation.frame.model_dump()
        self._trace.emit(run_id, "observation_started", image=observation.value,
                         step=step.id, revision=revision, frame=frame, question=question,
                         condition=step.evidence.condition,
                         measurement=step.evidence.measurement.model_dump() if step.evidence.measurement else None)
        try:
            await verify_image(
                self._decisions, self._image_query, images=self._current_frame.images,
                image=observation.value, condition=step.evidence.condition, question=question,
                frame=frame, step_id=step.id, revision=revision,
                decision_timeout_s=self._decision_timeout_s, vision_timeout_s=self._vision_timeout_s, trace=trace,
                measurement=step.evidence.measurement, measurement_query=self._measurement_query, policy=self._policy,
            )
        except asyncio.CancelledError:
            error = {"type": "CancelledError", "message": "observation cancelled"}
            raise
        except Exception as exc:  # noqa: BLE001 — Failed checks never authorize progress.
            error = {"type": type(exc).__name__, "message": str(exc)}
            logger.warning("Guide decision unavailable pid={!r} step={}: {}", session.participant_id, step.id, exc)
        finally:
            if trace.prepared_image is not None:
                self._trace.emit(run_id, "decision_image_prepared", image=trace.prepared_image,
                                 step=step.id, revision=revision, frame=frame)
            self._trace.emit(run_id, "observation_finished", step=step.id, revision=revision,
                             frame=frame, latency_s=time.monotonic() - before, error=error, **trace.record())
        trace.error = error
        async with session.lock:
            if not self._current(session, step, revision):
                return trace
            session.last_verification = {
                **trace.record(), "step_id": step.id, "frame": frame, "error": error,
                "condition": step.evidence.condition, "question": question,
                "measurement": step.evidence.measurement.model_dump() if step.evidence.measurement else None,
                "observed_monotonic": time.monotonic(),
            }
            session.observation_history.append(copy.deepcopy(session.last_verification))
            del session.observation_history[:-self._policy.history_limit]
        return trace

    async def _observation_turn(
        self,
        session: _Session,
        step: Step,
        revision: int,
        state: dict[str, Any],
        observation: _Observation,
    ) -> None:
        assert step.agent is not None  # Validation requires a policy for model-backed observations.
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
        self._trace.emit(session.run_id, "state_committed", step=step.id, revision=session.revision,
                         changes=changes, state=dict(session.state), complete=complete)
        session.revision += 1
        if complete:
            session.notices.append(step.complete_message)
        elif message.strip():
            session.notices.append(message.strip())
        return True, complete, "state committed"

    async def _trigger(self, session: _Session, step: Step, state: dict[str, Any]) -> _Observation:
        if step.trigger.function == "current_view":
            try:
                async with asyncio.timeout(self._vision_timeout_s):
                    frame = await self._current_frame.execute(
                        CurrentFrameRequest(participant_id=session.participant_id)
                    )
                # Pin the selected input before inference. Both decision paths
                # must inspect this frame, even when the camera keeps streaming.
                image = self._current_frame.images.resolve(frame.image)
                if isinstance(image, Path):
                    image = await asyncio.to_thread(image.read_bytes)
                elif isinstance(image, str):
                    header, separator, encoded = image.partition(",")
                    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
                        raise ValueError("selected guide frame must be local encoded image data")
                    image = base64.b64decode(encoded, validate=True)
                return _Observation(True, image, _FrameIdentity(
                    participant_id=frame.participant_id, track_id=frame.track_id,
                    timestamp_us=frame.timestamp_us, sequence=frame.sequence,
                ))
            except (FrameUnavailable, TimeoutError) as exc:
                return _Observation(False, str(exc))
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
                    result = await caption_image(
                        self._image_query, images=self._current_frame.images,
                        image=self._current_frame.images.resolve(frame.image), question=request.question,
                        timeout_s=self._vision_timeout_s, owner=participant_id,
                    )
                    return _ViewResult(
                        **result.model_dump(),
                        frame=_FrameIdentity(
                            participant_id=frame.participant_id, track_id=frame.track_id,
                            timestamp_us=frame.timestamp_us, sequence=frame.sequence,
                        ),
                    )
            except FrameUnavailable as exc:
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

    def _timer_tool(self, participant_id: str) -> Tool[EmptyRequest, _TimerStatus]:
        session = self._sessions.get(participant_id)
        timer = session.timer if session is not None else None

        async def read(_request: EmptyRequest) -> _TimerStatus:
            if session is None:
                return _TimerStatus(available=False, unavailable_reason="No active guide timer.")
            async with session.lock:
                if self._sessions.get(participant_id) is not session or timer is not session.timer:
                    return _TimerStatus(available=False, unavailable_reason="The active timer changed.")
                if timer is None or not session.active or not self._connected(participant_id):
                    return _TimerStatus(available=False,
                                        unavailable_reason=session.timer_error or "No active guide timer.")
                return _TimerStatus(available=True, **timer.read().model_dump())

        return Tool(
            "clock__timer",
            "Read the active step's runtime-owned timer. Takes no arguments.",
            EmptyRequest, _TimerStatus, read,
        )

    async def _publish_notices(self, session: _Session) -> None:
        async with session.lock:
            notices = tuple(session.notices)
            session.notices.clear()
            if (not session.active or not self._connected(session.participant_id)
                    or self._sessions.get(session.participant_id) is not session):
                return
            if session.step.finish_on_complete and self._is_complete(session):
                # Finish under the same lock as controls and notice delivery so
                # a racing stop/restart cannot publish an old run's completion.
                await self._invalidate(session.participant_id)
                message = ("Guide ended with skipped steps." if session.skipped
                           else session.workflow.complete_message)
                session.completed_steps.append({"step_id": session.step.id, "outcome": "completed",
                                                "verification": (_historical_evidence(session.last_verification)
                                                         if session.last_verification else None)})
                self._reset_progress(session, active=False, reason="skipped" if session.skipped else "completed")
                await self._announce(session.participant_id, f"{message} Say start guide to begin again.")
                return
            # A transition invalidates already queued speech under this same lock.
            for message in notices:
                await self._announce(session.participant_id, message)

    def _current(self, session: _Session, step: Step, revision: int) -> bool:
        return (
            session.active
            and self._connected(session.participant_id)
            and self._sessions.get(session.participant_id) is session
            and session.step_id == step.id
            and session.revision == revision
            and not self._is_complete(session)
        )

    @staticmethod
    def _is_complete(session: _Session) -> bool:
        # State may be shared with preceding steps. A preexisting completion
        # field cannot authorize advancing past a timer that has not elapsed.
        if not session.active:
            return False
        if session.step.trigger.function == "clock__timer":
            if session.timer is None or not session.timer.read().expired:
                return False
        return session.step.is_complete(session.state)

    @staticmethod
    def _resolve(value: Any, state: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {name: _Guide._resolve(item, state) for name, item in value.items()}
        if isinstance(value, list):
            return [_Guide._resolve(item, state) for item in value]
        if isinstance(value, str) and value.startswith("$state."):
            name = value.removeprefix("$state.")
            if name not in state:
                raise ValueError(f"trigger references missing state: {name}")
            return state[name]
        return value


async def _cancel(runners: dict[str, StepRunner], participant_id: str) -> None:
    runner = runners.get(participant_id)
    if runner is None:
        return
    await runner.stop()
    if runners.get(participant_id) is runner:
        runners.pop(participant_id, None)


def _discard(runners: dict[str, StepRunner], participant_id: str, runner: StepRunner, task: asyncio.Task) -> None:
    if runners.get(participant_id) is runner:
        runners.pop(participant_id, None)
    if task.cancelled():
        return
    with suppress(asyncio.CancelledError):
        if error := task.exception():
            logger.error("Guide step failed pid={!r}: {!r}", participant_id, error)


def _evidence_context(item):
    result = copy.deepcopy(item)
    observed = result.pop("observed_monotonic", None)
    if observed is not None:
        result["age_s"] = max(0.0, time.monotonic() - observed)
    return result


def _historical_evidence(item):
    result = _evidence_context(item)
    if "age_s" in result:
        result["age_at_transition_s"] = result.pop("age_s")
    return result
