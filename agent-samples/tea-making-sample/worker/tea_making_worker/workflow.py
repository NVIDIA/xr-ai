# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native agent that owns tea guidance state and observation tasks."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import nemo_relay
from loguru import logger
from pydantic import BaseModel
from xr_ai_interaction import ParticipantJoined, ParticipantLeft
from xr_ai_models import DecisionService, LLMService
from xr_ai_runtime import (
    Agent,
    AgentRuntime,
    RuntimeClosedError,
    RuntimeContext,
    subscribe,
)
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.vision import ImageQueryTool

from ._capture import _attempt, _event
from ._observations import _decode_values, _resolve
from ._visual_step import _VisualValidationStep
from .events import (
    GUIDANCE_NOTICE_TOPIC,
    GUIDANCE_RECORD_TOPIC,
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    GuidanceNotice,
    GuidanceRecord,
    ParticipantCleanupComplete,
)
from .spec import Step, Workflow
from .workflow_state import (
    WorkflowEvent,
    WorkflowSession,
    WorkflowStore,
    monotonic_now,
)
from .workflow_tools import (
    _NAMED_TOOL_NAMES,
    TimerRequest,
    clock_now_tool,
    clock_timer_tool,
    named_tool_set,
    participant_current_view_tool,
    rag_lookup_tool,
    temperature_threshold_tool,
    temperature_verify_tool,
    workflow_management_tools,
    workflow_start_tool,
    workflow_status_tool,
)

_PROMPTS = Path(__file__).resolve().parent / "prompts"
_VOICE_PROMPT = _PROMPTS / "guidance_voice.txt"
_POLL_INTERVAL_S = 0.25


class _TriggerResult(BaseModel):
    available: bool
    value: Any = None
    detail: str = ""
    frame_key: tuple[str, str, int, int] | None = None


class GuidanceAgent(Agent):
    """Own participant workflow state, tools, and periodic observations."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        llm: LLMService,
        decisions: DecisionService,
        current_frame: CurrentFrameTool,
        image_query: ImageQueryTool,
        rag: RAGTools,
        vlm_timeout_s: float = 15.0,
        record_event: Callable[[WorkflowSession, WorkflowEvent], Awaitable[None]] | None = None,
    ) -> None:
        if vlm_timeout_s <= 0:
            raise ValueError("vlm_timeout_s must be positive")
        super().__init__()
        self._record_event = record_event
        self.workflow = workflow
        self.store = WorkflowStore(workflow)
        self._llm = llm
        self._decisions = decisions
        self._current_frame = current_frame
        self._image_query = image_query
        self._rag = rag
        self._vlm_timeout_s = vlm_timeout_s
        self._voice_prompt = _VOICE_PROMPT.read_text(encoding="utf-8").strip()
        self._runtime: AgentRuntime | None = None
        self._connected: set[str] = set()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._publish_locks: dict[str, asyncio.Lock] = {}
        for step in workflow.steps.values():
            names = set(step.voice.tools)
            if step.observation.evaluation is not None:
                names.add(step.observation.evaluation.function)
            if step.observation.lookup is not None:
                names.add("rag_lookup")
            names.update(projection.function for projection in step.context.values())
            unknown = names - _NAMED_TOOL_NAMES
            if unknown:
                raise ValueError(f"workflow step {step.id!r} references unknown tools: {sorted(unknown)}")

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        """Bind runtime publication before participant work starts."""

        if self._runtime is not None and self._runtime is not runtime:
            raise RuntimeError("guidance agent is already bound to a runtime")
        self._runtime = runtime

    def root_tools(self, participant_id: str) -> ToolSet:
        """Return tea lifecycle tools available while root owns foreground."""

        session = self.store.get(participant_id)
        return ToolSet(
            (
                workflow_start_tool(
                    self.store,
                    session,
                    lambda: self._flush(session),
                ),
                workflow_status_tool(self.store, session),
            )
        )

    def active_tools(self, participant_id: str) -> ToolSet | None:
        """Return management plus current-step tools for active guidance."""

        session = self.store.find(participant_id)
        if session is None or not session.active or session.step_id is None:
            return None
        step = self.workflow.step(session.step_id)
        quick = self._named_tools(session, step.voice.tools)
        tools: dict[str, Tool[Any, Any]] = {
            tool.name: tool
            for tool in workflow_management_tools(
                self.store,
                session,
                lambda: self._flush(session),
            )
        }
        tools.update(dict(quick.items()))
        return ToolSet(tools)

    async def context_snapshot(self, participant_id: str) -> dict[str, Any]:
        """Sample native sources for any question, without invoking perception."""
        while True:
            session = self.store.find(participant_id)
            if session is None:
                return {"active": False, "run_id": None}
            async with session.lock:
                run_id, revision, step_id = session.run_id, session.revision, session.step_id
                state = copy.deepcopy(session.state)
                evidence = {step_id: copy.deepcopy(session.evidence[step_id])} if step_id in session.evidence else {}
                active = session.active
            step = self.workflow.step(step_id) if active and step_id is not None else None
            operations = {}
            if step is not None:
                tools = self._named_tools(session, tuple(value.function for value in step.context.values()))
                for name, projection in step.context.items():
                    tool = tools.get(projection.function)
                    try:
                        arguments = _resolve(projection.arguments, state, {})
                    except KeyError:
                        operations[name] = {"available": False, "reason": "Required state is not established."}
                        continue
                    request = tool.request_model.model_validate(arguments)
                    result = await tool.execute(request)
                    operations[name] = {"available": True, **result.model_dump(mode="json")}
            async with session.lock:
                if self.store.find(participant_id) is not session:
                    continue
                if (session.run_id, session.revision, session.step_id) != (run_id, revision, step_id):
                    continue
                return {
                    "run_id": run_id, "revision": revision, "active": active,
                    "step": {"id": step.id, "title": step.title} if step else None,
                    "state": state, "operations": operations,
                    "sampled_at_us": _timestamp_us(), "evidence": evidence,
                    "confirmation_hits": session.evidence_hits,
                    "step_complete": step.is_complete(state) if step else False,
                    "advancement": self.workflow.advancement,
                }

    def active_context(self, participant_id: str) -> str | None:
        """Return current-step policy and sparse state without conversation history."""

        session = self.store.find(participant_id)
        if session is None or not session.active or session.step_id is None:
            return None
        step = self.workflow.step(session.step_id)
        return json.dumps(
            {
                "workflow": self.workflow.foreground_prompt,
                "voice_policy": self._voice_prompt,
                "step": {"id": step.id, "title": step.title},
                "instructions": step.voice.prompt,
                "state": self.workflow.project(step, session.state),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def status(self, participant_id: str) -> str:
        """Return deterministic guidance status without a model call."""

        return self.store.status(self.store.get(participant_id))

    @subscribe(PARTICIPANT_JOINED_TOPIC)
    async def participant_joined(
        self,
        _event: ParticipantJoined,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        if participant_id in self._connected:
            return
        self._connected.add(participant_id)
        self.store.release(participant_id)
        session = self.store.get(participant_id)
        self.store.record(session, "participant.joined")
        await self._flush(session)
        self._start_monitor(participant_id)

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: ParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        self._connected.discard(participant_id)
        await self._cancel_monitor(participant_id)
        session = self.store.find(participant_id)
        if session is not None:
            async with session.lock:
                self.store.record(session, "participant.left")
            await self._flush(session)
        self.store.release(participant_id)
        self._publish_locks.pop(participant_id, None)
        await ctx.publish(
            PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
            ParticipantCleanupComplete(
                generation=ctx.metadata.message_id,
                producer="guidance",
            ),
        )

    async def stop(self) -> None:
        """Cancel every participant observation task owned by this agent."""

        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        self._connected.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._publish_locks.clear()

    def _named_tools(
        self,
        session: WorkflowSession,
        names: tuple[str, ...],
    ) -> ToolSet:
        current_view = participant_current_view_tool(
            session.participant_id,
            self._current_frame,
            self._image_query,
            timeout_s=self._vlm_timeout_s,
        )
        return named_tool_set(
            names,
            current_view=current_view,
            rag_lookup=rag_lookup_tool(self._rag),
            clock_now=clock_now_tool(),
            clock_timer=clock_timer_tool(),
            temperature_threshold=temperature_threshold_tool(),
            temperature_verify=temperature_verify_tool(session),
        )

    def _visual_step(self, session: WorkflowSession, step: Step) -> _VisualValidationStep:
        names = []
        if step.observation.evaluation is not None:
            names.append(step.observation.evaluation.function)
        if step.observation.lookup is not None:
            names.append("rag_lookup")
        return _VisualValidationStep(
            question=str(step.trigger.arguments.get("question", "")),
            policy=step.observation,
            observer=self._image_query,
            decisions=self._decisions,
            llm=self._llm,
            tools=self._named_tools(session, tuple(names)),
            timeout_s=self._vlm_timeout_s,
        )

    def _start_monitor(self, participant_id: str) -> None:
        existing = self._tasks.get(participant_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._monitor(participant_id),
            name=f"tea-guidance:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(
            lambda completed, pid=participant_id: self._discard_monitor(
                pid,
                completed,
            )
        )

    async def _cancel_monitor(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _discard_monitor(
        self,
        participant_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if task.cancelled():
            return
        with suppress(asyncio.CancelledError):
            if error := task.exception():
                logger.error(
                    "tea guidance monitor stopped pid={!r}: {!r}",
                    participant_id,
                    error,
                )

    async def _monitor(self, participant_id: str) -> None:
        while participant_id in self._connected:
            session = self.store.find(participant_id)
            if session is None:
                return
            try:
                await self._tick(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.opt(exception=True).warning(
                    "tea guidance observation failed pid={!r}",
                    participant_id,
                )
                async with session.lock:
                    self.store.record(
                        session,
                        "agent.observation_error",
                        str(exc),
                    )
                await self._flush(session)
            await asyncio.sleep(_POLL_INTERVAL_S)

    async def _tick(self, session: WorkflowSession) -> None:
        async with session.lock:
            if not session.active or session.step_id is None or self.store.step_complete(session):
                return
            now = monotonic_now()
            if session.next_tick > now:
                return
            step = self.workflow.step(session.step_id)
            revision = session.revision
            run_id = session.run_id
            state = dict(session.state)
            session.next_tick = now + step.trigger.interval_s

        def snapshot():
            return {
                "state": dict(session.state),
                "revision": session.revision,
                "run_id": session.run_id,
                "step_id": session.step_id,
                "evidence_hits": session.evidence_hits,
                "active": session.active,
                "current_session": self.store.find(session.participant_id) is session,
            }

        async with _attempt(session.participant_id, "observer", step_id=step.id, snapshot=snapshot):
            trigger = await self._trigger(session.participant_id, step, state)
            await _event("trigger.finished", **trigger.model_dump(mode="json"))

            next_state: dict[str, Any] | None = None
            async with session.lock:
                if not self._current(session, step, revision):
                    return
                if trigger.frame_key is not None and trigger.frame_key == session.observed_frame:
                    return
                if not trigger.available:
                    session.evidence[step.id] = {
                        "available": False, "detail": trigger.detail,
                        "observed_at_us": trigger.frame_key[3] if trigger.frame_key else _timestamp_us(),
                        "frame_key": trigger.frame_key,
                    }
                    self.store.record(
                        session,
                        "trigger.unavailable",
                        trigger.detail,
                    )
                else:
                    next_state = dict(session.state)
            if trigger.available:
                if next_state is None:
                    raise AssertionError("model observation requires current state")
                processed = await self._observe(
                    session, step, trigger.value, next_state, revision, frame_key=trigger.frame_key
                )
                if processed and trigger.frame_key is not None:
                    async with session.lock:
                        if (
                            self.store.find(session.participant_id) is session
                            and session.run_id == run_id and session.step_id == step.id
                        ):
                            session.observed_frame = trigger.frame_key
            await self._flush(session)

    async def _trigger(
        self,
        participant_id: str,
        step: Step,
        state: dict[str, Any],
    ) -> _TriggerResult:
        arguments = _resolve(step.trigger.arguments, state, {})
        if step.trigger.function == "current_view":
            # Keep visual inference on Omni. Choice-only decisions consume its description.
            async with asyncio.timeout(self._vlm_timeout_s):
                frame = await self._current_frame.execute(CurrentFrameRequest(participant_id=participant_id))
                key = (step.id, frame.track_id, frame.sequence, frame.timestamp_us)
                session = self.store.find(participant_id)
                if session is not None and session.observed_frame == key:
                    return _TriggerResult(available=False, frame_key=key, detail="frame already observed")
                description = await self._visual_step(session, step).observe(frame)
            if description is None:
                return _TriggerResult(available=False, frame_key=key, detail="Visual evidence is unavailable")
            return _TriggerResult(
                available=True,
                value=description,
                frame_key=(step.id, frame.track_id, frame.sequence, frame.timestamp_us),
            )
        if step.trigger.function == "clock__timer":
            result = await clock_timer_tool().execute(TimerRequest.model_validate(arguments))
            value: Any = result.model_dump(mode="json")
            if step.trigger.result_field is not None:
                value = value[step.trigger.result_field]
            return _TriggerResult(available=True, value=value)
        raise ValueError(f"unsupported workflow trigger: {step.trigger.function!r}")

    async def _observe(
        self,
        session: WorkflowSession,
        step: Step,
        observation: Any,
        state: dict[str, Any],
        revision: int,
        *,
        frame_key: tuple[str, str, int, int] | None = None,
    ) -> bool:
        policy = step.observation
        started = time.perf_counter()
        # Retain configured identity facts before retrieval; later unreadable frames
        # cannot erase them, and retrieval failure must not discard visual evidence.
        if policy.retain:
            decoded = _decode_values(policy, observation)
            retained = {name: decoded[name] for name in policy.retain if name not in state and name in decoded}
            if retained:
                async with session.lock:
                    if not self._current(session, step, revision):
                        return False
                    result = self.store.commit(session, retained, "")
                    if not result.accepted:
                        raise ValueError(result.message)
                    revision = session.revision
                    state = dict(session.state)
        outcome = await self._visual_step(session, step).evaluate(
            observation, expected=self.workflow.project(step, state)
        )
        await _event(
            "observation.judged",
            evidence=outcome.evidence,
            values=outcome.values,
            native_result=outcome.result,
            expected_revision=revision,
        )
        commit = None
        async with session.lock:
            if not self._current(session, step, revision):
                return False
            session.evidence[step.id] = {
                "evidence": outcome.evidence, "values": outcome.values,
                "native_result": outcome.result, "observation": observation,
                "observed_at_us": frame_key[3] if frame_key else _timestamp_us(),
                "frame_key": frame_key,
            }
            self.store.observe(session, outcome.evidence if outcome.evidence != "unknown" else None)
            self.store.record(
                session,
                "agent.observation_complete",
                json.dumps(
                    {
                        "kind": policy.kind,
                        "evidence": outcome.evidence,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                ),
            )
            if outcome.evidence == "accepted":
                # Bind values and timestamps only after confirmation, under the revision guard.
                if step.evidence is None or session.evidence_hits >= step.evidence.consecutive:
                    updates = _resolve(
                        {**policy.updates, **outcome.overrides},
                        state,
                        outcome.values,
                        now_us=_timestamp_us(),
                    )
                    commit = self.store.commit(session, updates, "")
            elif policy.lookup is not None:
                # Preserve independently resolved fields/provenance across retries,
                # without applying completion literals or inferring missing values.
                partial = {
                    name: outcome.values[binding.removeprefix("$observation.")]
                    for name, binding in policy.updates.items()
                    if isinstance(binding, str) and binding.startswith("$observation.")
                    and binding.removeprefix("$observation.") in outcome.values
                }
                partial.update(outcome.overrides)
                if partial:
                    commit = self.store.commit(session, partial, "")


        await _event("observation.applied", commit=commit)
        return True

    def _current(
        self,
        session: WorkflowSession,
        step: Step,
        revision: int,
    ) -> bool:
        return (
            self.store.find(session.participant_id) is session
            and session.active
            and session.step_id == step.id
            and session.revision == revision
            and not step.is_complete(session.state)
        )

    async def _flush(self, session: WorkflowSession) -> None:
        runtime = self._runtime
        if runtime is None or not runtime.running:
            return
        lock = self._publish_locks.setdefault(
            session.participant_id,
            asyncio.Lock(),
        )
        async with lock:
            async with session.lock:
                events = self.store.drain_events(session)
                notices = self.store.drain_notices(session)
            for text in notices:
                try:
                    await runtime.publish(
                        GUIDANCE_NOTICE_TOPIC,
                        GuidanceNotice(
                            timestamp_us=_timestamp_us(),
                            text=text,
                        ),
                        participant_id=session.participant_id,
                        source="guidance",
                    )
                except RuntimeClosedError:
                    return
                except Exception:
                    logger.opt(exception=True).error(
                        "tea guidance notice failed pid={!r}",
                        session.participant_id,
                    )
            for event in events:
                if self._record_event is not None:
                    await self._record_event(session, event)
                try:
                    await runtime.publish(
                        GUIDANCE_RECORD_TOPIC,
                        GuidanceRecord(
                            run_id=event.run_id,
                            timestamp_us=_timestamp_us(),
                            event=event.event,
                            step_id=event.step_id,
                            message=event.message,
                            state=event.state,
                        ),
                        participant_id=session.participant_id,
                        source="guidance",
                    )
                except RuntimeClosedError:
                    return
                except Exception:
                    logger.opt(exception=True).error(
                        "tea guidance record failed pid={!r}",
                        session.participant_id,
                    )

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("tea guidance requires a participant")
        return participant_id


def _timestamp_us() -> int:
    return time.time_ns() // 1_000


__all__ = ["GuidanceAgent"]
