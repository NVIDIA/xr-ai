# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native agent that owns tea guidance state and observation tasks."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import nemo_relay
from loguru import logger
from pydantic import BaseModel
from xr_ai_models import ChatMessage, LLMService, ToolDef
from xr_ai_runtime import (
    Agent,
    AgentRuntime,
    RuntimeClosedError,
    RuntimeContext,
    subscribe,
)
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.tool_calling import ToolLoopError, run_tool_loop
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import VoiceParticipantJoined, VoiceParticipantLeft

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
    WorkflowSession,
    WorkflowStore,
    monotonic_now,
)
from .workflow_tools import (
    _NAMED_TOOL_NAMES,
    CommitRequest,
    CurrentViewRequest,
    TimerRequest,
    WorkflowCommitResult,
    clock_now_tool,
    clock_timer_tool,
    named_tool_set,
    participant_current_view_tool,
    rag_lookup_tool,
    temperature_verify_tool,
    workflow_commit_tool,
    workflow_management_tools,
    workflow_start_tool,
    workflow_status_tool,
)

_PROMPTS = Path(__file__).resolve().parent / "prompts"
_OBSERVATION_PROMPT = _PROMPTS / "guidance_observation.txt"
_VOICE_PROMPT = _PROMPTS / "guidance_voice.txt"
_POLL_INTERVAL_S = 0.25


class _TriggerResult(BaseModel):
    available: bool
    value: Any = None
    detail: str = ""


class _StaleObservation(RuntimeError):
    """Stop an observation turn after its workflow revision changes."""


class GuidanceAgent(Agent):
    """Own participant workflow state, tools, and periodic observations."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        llm: LLMService,
        current_frame: CurrentFrameTool,
        image_query: ImageQueryTool,
        rag: RAGTools,
        vlm_timeout_s: float = 15.0,
    ) -> None:
        if vlm_timeout_s <= 0:
            raise ValueError("vlm_timeout_s must be positive")
        super().__init__()
        self.workflow = workflow
        self.store = WorkflowStore(workflow)
        self._llm = llm
        self._current_frame = current_frame
        self._image_query = image_query
        self._rag = rag
        self._vlm_timeout_s = vlm_timeout_s
        self._observation_prompt = _OBSERVATION_PROMPT.read_text(
            encoding="utf-8"
        ).strip()
        self._voice_prompt = _VOICE_PROMPT.read_text(encoding="utf-8").strip()
        self._runtime: AgentRuntime | None = None
        self._connected: set[str] = set()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._publish_locks: dict[str, asyncio.Lock] = {}
        for step in workflow.steps.values():
            unknown = (
                set(step.agent.tools) | set(step.voice.tools)
            ) - _NAMED_TOOL_NAMES
            if unknown:
                raise ValueError(
                    f"workflow step {step.id!r} references unknown tools: "
                    f"{sorted(unknown)}"
                )

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
        _event: VoiceParticipantJoined,
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
        _event: VoiceParticipantLeft,
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
            temperature_verify=temperature_verify_tool(session),
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
            if (
                not session.active
                or session.step_id is None
                or self.store.step_complete(session)
            ):
                return
            now = monotonic_now()
            if session.next_tick > now:
                return
            step = self.workflow.step(session.step_id)
            revision = session.revision
            state = dict(session.state)
            session.next_tick = now + step.trigger.interval_s
        trigger = await self._trigger(
            session.participant_id,
            step,
            state,
        )
        next_state: dict[str, Any] | None = None
        async with session.lock:
            if not self._current(session, step, revision):
                return
            if not trigger.available:
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
            await self._observe(
                session,
                step,
                trigger.value,
                next_state,
                revision,
            )
        await self._flush(session)

    async def _trigger(
        self,
        participant_id: str,
        step: Step,
        state: dict[str, Any],
    ) -> _TriggerResult:
        arguments = _resolve(step.trigger.arguments, state)
        if step.trigger.function == "current_view":
            tool = participant_current_view_tool(
                participant_id,
                self._current_frame,
                self._image_query,
                timeout_s=self._vlm_timeout_s,
            )
            result = await tool.execute(CurrentViewRequest.model_validate(arguments))
            if not result.available:
                return _TriggerResult(
                    available=False,
                    detail=result.text,
                )
            return _TriggerResult(available=True, value=result.text)
        if step.trigger.function == "clock__timer":
            result = await clock_timer_tool().execute(
                TimerRequest.model_validate(arguments)
            )
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
    ) -> None:
        quick = self._named_tools(session, step.agent.tools)
        commit = self._observation_commit_tool(session, step, revision)
        tools = ToolSet(
            {
                commit.name: commit,
                **dict(quick.items()),
            }
        )
        request = json.dumps(
            {
                "observation": observation,
                "already_complete": False,
                "state": self.workflow.project(step, state),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system = "\n".join(
            (
                self._observation_prompt,
                _state_contract(self.workflow, step),
                step.agent.prompt,
            )
        )

        async def call_model(
            messages: tuple[ChatMessage, ...],
            definitions: tuple[ToolDef, ...],
        ):
            response = await self._llm.chat(
                messages,
                tools=definitions,
                max_tokens=512,
                temperature=0.0,
                enable_thinking=False,
            )
            async with session.lock:
                if not self._current(session, step, revision):
                    raise _StaleObservation
            return response

        try:
            result = await run_tool_loop(
                (
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=request),
                ),
                tools,
                call_model,
                max_iterations=3,
                max_tool_calls=3,
            )
        except _StaleObservation:
            return
        except ToolLoopError as exc:
            async with session.lock:
                if self._current(session, step, revision):
                    self._reject_observation_evidence(session, step)
                    self.store.record(
                        session,
                        "agent.observation_skipped",
                        str(exc),
                    )
            return
        called = tuple(record.call.name for record in result.tool_calls)
        commit_record = next(
            (
                record
                for record in result.tool_calls
                if record.call.name == "workflow__commit"
            ),
            None,
        )
        async with session.lock:
            if commit_record is None:
                if self._current(session, step, revision):
                    self._reject_observation_evidence(session, step)
                    self.store.record(
                        session,
                        "agent.observation_skipped",
                        "model did not call workflow__commit",
                    )
                return
            commit_result = json.loads(commit_record.message.content)
            if (
                commit_result.get("accepted") is True
                and session.step_id == step.id
                and session.revision == commit_result.get("revision")
            ):
                self.store.record(
                    session,
                    "agent.observation_complete",
                    ",".join(called),
                )

    def _observation_commit_tool(
        self,
        session: WorkflowSession,
        step: Step,
        revision: int,
    ) -> Tool:
        """Count model completion judgments before applying guarded state."""

        if step.evidence is None:
            return workflow_commit_tool(
                self.store,
                session,
                expected_step_id=step.id,
                expected_revision=revision,
            )

        async def commit(request: CommitRequest) -> WorkflowCommitResult:
            async with session.lock:
                if session.step_id != step.id or session.revision != revision:
                    return WorkflowCommitResult(
                        accepted=False,
                        complete=False,
                        message="observation is stale",
                        revision=session.revision,
                    )
                updates = dict(request.updates)
                judgment = (
                    "accepted"
                    if self.store._completion_proposed(session, updates)
                    else "rejected"
                )
                self.store.observe(session, judgment)
                result = self.store.commit(session, updates, request.message)
            return WorkflowCommitResult(
                accepted=result.accepted,
                complete=result.complete,
                message=result.message,
                revision=result.revision,
            )

        return Tool(
            "workflow__commit",
            (
                "Commit one semantic judgment from the fresh observation. "
                "Use empty updates when the observation does not support completion."
            ),
            CommitRequest,
            WorkflowCommitResult,
            commit,
            return_direct=True,
        )

    def _reject_observation_evidence(
        self,
        session: WorkflowSession,
        step: Step,
    ) -> None:
        if step.evidence is not None:
            self.store.observe(session, "rejected")

    @staticmethod
    def _current(
        session: WorkflowSession,
        step: Step,
        revision: int,
    ) -> bool:
        return (
            session.active
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
                try:
                    await runtime.publish(
                        GUIDANCE_RECORD_TOPIC,
                        GuidanceRecord(
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


def _resolve(value: Any, state: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {name: _resolve(item, state) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, state) for item in value]
    if isinstance(value, str) and value.startswith("$state."):
        name = value.removeprefix("$state.")
        if name not in state:
            raise ValueError(f"trigger references missing state: {name}")
        return state[name]
    return value


def _state_contract(workflow: Workflow, step: Step) -> str:
    writes = "; ".join(
        (
            f"{name}:{workflow.state_fields[name].type} — {workflow.state_fields[name].description.rstrip('.')}"
        )
        for name in step.writes
    )
    completion = ", ".join(
        f"{name}={json.dumps(value)}" for name, value in step.complete_when.items()
    )
    return f"Writable state: {writes}. Completion requires {completion}."


def _timestamp_us() -> int:
    return time.time_ns() // 1_000


__all__ = ["GuidanceAgent"]
