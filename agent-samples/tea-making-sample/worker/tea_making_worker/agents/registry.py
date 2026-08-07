# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build small NAT agents directly from the workflow definition."""

from __future__ import annotations

import json
from typing import Any

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import FunctionRef, LLMRef

from ..runtime.events import emit
from ..runtime.scope import current_invocation
from ..runtime.state import Session
from ..spec import Step, Workflow
from .factory import build_agent
from .invoke import invoke_with_tool_retry
from .prompts import HUMAN, STEP, TEA, VOICE

_COMMIT = FunctionRef("workflow__commit")
_TEA_MANAGEMENT_TOOLS = tuple(
    FunctionRef(f"workflow__{name}") for name in ("advance", "reset", "restart", "status")
)


class AgentRegistry:
    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self._step: dict[str, Function] = {}
        self._tea: dict[str, Function] = {}

    async def build(self, builder: WorkflowBuilder, llm_ref: LLMRef) -> None:
        for step in self.workflow.steps.values():
            self._step[step.id] = await build_agent(
                builder,
                name=f"observe_{step.id}",
                llm_ref=llm_ref,
                prompt=f"{STEP}\n{_state_contract(self.workflow, step)}\n{step.agent.prompt}\n{HUMAN}",
                tools=(_COMMIT, *map(FunctionRef, step.agent.tools)),
                return_direct=(_COMMIT,),
            )
        await self._build_foreground(builder, llm_ref)

    async def build_foreground(self, builder: WorkflowBuilder, llm_ref: LLMRef) -> None:
        """Build only the production foreground agents for model-backed voice evals."""
        await self._build_foreground(builder, llm_ref)

    async def _build_foreground(self, builder: WorkflowBuilder, llm_ref: LLMRef) -> None:
        context = self.workflow.foreground_prompt
        for step in self.workflow.steps.values():
            self._tea[step.id] = await build_agent(
                builder,
                name=f"foreground_tea_{step.id}",
                llm_ref=llm_ref,
                prompt=f"{TEA}\n{VOICE}\n{context}\n{step.voice.prompt}\n{HUMAN}".strip(),
                tools=(*_TEA_MANAGEMENT_TOOLS, *map(FunctionRef, step.voice.tools)),
                return_direct=_TEA_MANAGEMENT_TOOLS,
            )

    async def observe(self, session: Session, observation: Any, trace_id: str) -> str:
        step = self._active_step(session)
        request = _json(
            observation=observation,
            already_complete=step.is_complete(session.state),
            state=self.workflow.project(step, session.state),
        )
        emit(
            "agent.observe.request",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            chars=len(request),
            input=request,
        )
        agent = self._step[step.id]

        def retry(feedback: str) -> None:
            emit(
                "agent.observe.retry",
                participant_id=session.participant_id,
                step=step.id,
                trace_id=trace_id,
                reason="invalid_tool_arguments",
                feedback=feedback,
            )

        result = await invoke_with_tool_retry(
            agent,
            request,
            retry=retry,
            skip_repeated_invalid=True,
        )
        if not result:
            emit(
                "agent.observe.skipped",
                participant_id=session.participant_id,
                step=step.id,
                trace_id=trace_id,
                reason="invalid_tool_arguments",
            )
            return ""
        emit(
            "agent.observe.response",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            output=result,
        )
        return result

    async def route(self, session: Session, request: str, trace_id: str) -> str:
        step = self._active_step(session)
        agent = self._tea.get(step.id)
        foreground = "tea"
        payload = _json(request=request, state=self.workflow.project(step, session.state))
        if agent is None:
            raise RuntimeError("agent registry has not been built")
        emit(
            "agent.foreground.request",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            foreground=foreground,
            chars=len(payload),
            input=payload,
        )
        call = current_invocation()
        call.route_operation = None

        def retry(feedback: str) -> None:
            emit(
                "agent.foreground.retry",
                participant_id=session.participant_id,
                step=step.id,
                trace_id=trace_id,
                foreground=foreground,
                reason="invalid_tool_arguments",
                feedback=feedback,
            )

        result = await invoke_with_tool_retry(agent, payload, retry=retry)
        emit(
            "agent.foreground.response",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            foreground=foreground,
            operation=call.route_operation or "answer",
            active=session.active,
            next_step=session.step_id,
            output=result,
        )
        return result.strip()

    def _active_step(self, session: Session) -> Step:
        if not session.active or session.step_id is None:
            raise ValueError("workflow is idle")
        return self.workflow.step(session.step_id)


def _json(**value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _state_contract(workflow: Workflow, step: Step) -> str:
    writes = "; ".join(
        f"{name}:{workflow.state_fields[name].type} — {workflow.state_fields[name].description.rstrip('.')}"
        for name in step.writes
    )
    completion = ", ".join(f"{name}={json.dumps(value)}" for name, value in step.complete_when.items())
    return f"Writable state: {writes}. Completion requires {completion}."


__all__ = ["AgentRegistry"]
