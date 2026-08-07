# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build small NAT agents directly from the workflow definition."""

from __future__ import annotations

import json
from typing import Any

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import FunctionRef, LLMRef
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from pydantic import ValidationError
from xr_ai_models import LLMService
from xr_ai_nat.llm import ModelsLLMConfig

from ..runtime.events import emit
from ..runtime.scope import current_invocation
from ..runtime.state import Session
from ..spec import Step, Workflow
from .prompts import HUMAN, ROOT, STEP, TEA, VOICE

_COMMIT = FunctionRef("workflow__commit")
_ROOT_TOOLS = tuple(FunctionRef(name) for name in ("workflow__start", "current_view", "rag_lookup"))
_TEA_MANAGEMENT_TOOLS = tuple(
    FunctionRef(f"workflow__{name}") for name in ("advance", "reset", "restart", "status")
)


class AgentRegistry:
    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self._step: dict[str, Function] = {}
        self._root: Function | None = None
        self._tea: dict[str, Function] = {}

    async def build(self, builder: WorkflowBuilder, llm: LLMService) -> None:
        llm_ref = await self._add_llm(builder, llm)
        for step in self.workflow.steps.values():
            self._step[step.id] = await self._agent(
                builder,
                name=f"observe_{step.id}",
                llm_ref=llm_ref,
                prompt=f"{STEP}\n{_state_contract(self.workflow, step)}\n{step.agent.prompt}\n{HUMAN}",
                tools=(_COMMIT, *map(FunctionRef, step.agent.tools)),
                return_direct=(_COMMIT,),
            )
        await self._build_foreground(builder, llm_ref)

    async def build_foreground(self, builder: WorkflowBuilder, llm: LLMService) -> None:
        """Build only the production foreground agents for model-backed voice evals."""
        await self._build_foreground(builder, await self._add_llm(builder, llm))

    async def _build_foreground(self, builder: WorkflowBuilder, llm_ref: LLMRef) -> None:
        context = self.workflow.foreground_prompt
        self._root = await self._agent(
            builder,
            name="foreground_root",
            llm_ref=llm_ref,
            prompt=f"{ROOT}\n{context}\n{HUMAN}".strip(),
            tools=_ROOT_TOOLS,
            return_direct=(FunctionRef("workflow__start"),),
        )
        for step in self.workflow.steps.values():
            self._tea[step.id] = await self._agent(
                builder,
                name=f"foreground_tea_{step.id}",
                llm_ref=llm_ref,
                prompt=f"{TEA}\n{VOICE}\n{context}\n{step.voice.prompt}\n{HUMAN}".strip(),
                tools=(*_TEA_MANAGEMENT_TOOLS, *map(FunctionRef, step.voice.tools)),
                return_direct=_TEA_MANAGEMENT_TOOLS,
            )

    @staticmethod
    async def _add_llm(builder: WorkflowBuilder, llm: LLMService) -> LLMRef:
        llm_ref = LLMRef("guide_llm")
        await builder.add_llm(
            llm_ref,
            ModelsLLMConfig(
                service=llm,
                model_name="guidance-llm",
                max_tokens=512,
                temperature=0,
                enable_thinking=False,
            ),
        )
        return llm_ref

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
        for attempt in range(2):
            try:
                result = await agent.ainvoke(request, to_type=str)
                break
            except Exception as exc:
                if not _is_tool_schema_error(exc):
                    raise
                final = attempt == 1
                emit(
                    "agent.observe.skipped" if final else "agent.observe.retry",
                    participant_id=session.participant_id,
                    step=step.id,
                    trace_id=trace_id,
                    tool=getattr(exc, "tool_name", None),
                    reason="invalid_tool_arguments",
                )
                if final:
                    return ""
                request = f"{request}\nRetry tool arguments: {_schema_feedback(exc)}"
        emit(
            "agent.observe.response",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            output=result,
        )
        return result

    async def route(self, session: Session, request: str, trace_id: str) -> str:
        if session.active:
            step = self._active_step(session)
            agent = self._tea.get(step.id)
            foreground = "tea"
            payload = _json(request=request, state=self.workflow.project(step, session.state))
        else:
            step = None
            agent = self._root
            foreground = "root"
            payload = _json(request=request)
        if agent is None:
            raise RuntimeError("agent registry has not been built")
        emit(
            "agent.foreground.request",
            participant_id=session.participant_id,
            step=step.id if step else None,
            trace_id=trace_id,
            foreground=foreground,
            chars=len(payload),
            input=payload,
        )
        call = current_invocation()
        for attempt in range(2):
            call.route_operation = None
            try:
                result = await agent.ainvoke(payload, to_type=str)
            except Exception as exc:
                if not _is_tool_schema_error(exc) or attempt == 1:
                    raise
                emit(
                    "agent.foreground.retry",
                    participant_id=session.participant_id,
                    step=step.id if step else None,
                    trace_id=trace_id,
                    foreground=foreground,
                    reason="invalid_tool_arguments",
                )
                payload = f"{payload}\nRetry tool arguments: {_schema_feedback(exc)}"
                continue
            emit(
                "agent.foreground.response",
                participant_id=session.participant_id,
                step=step.id if step else None,
                trace_id=trace_id,
                foreground=foreground,
                operation=call.route_operation or "answer",
                active=session.active,
                next_step=session.step_id,
                output=result,
            )
            return result.strip()
        raise AssertionError("unreachable")

    @staticmethod
    async def _agent(
        builder: WorkflowBuilder,
        *,
        name: str,
        llm_ref: LLMRef,
        prompt: str,
        tools: tuple[FunctionRef, ...],
        return_direct: tuple[FunctionRef, ...] = (),
    ) -> Function:
        if not tools:
            raise ValueError(f"NAT tool-calling agent {name!r} needs at least one tool")
        return await builder.add_function(
            name,
            ToolCallAgentWorkflowConfig(
                llm_name=llm_ref,
                tool_names=list(tools),
                return_direct=list(return_direct) or None,
                system_prompt=prompt,
                max_iterations=4,
                max_history=6,
                handle_tool_errors=False,
                verbose=True,
                log_response_max_chars=4_000,
                description=f"Guidance agent {name}",
            ),
        )

    def _active_step(self, session: Session) -> Step:
        if not session.active or session.step_id is None:
            raise ValueError("workflow is idle")
        return self.workflow.step(session.step_id)


def _is_tool_schema_error(exc: Exception) -> bool:
    return isinstance(getattr(exc, "source", None), ValidationError)


def _schema_feedback(exc: Exception) -> str:
    source: ValidationError = getattr(exc, "source")
    return "; ".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
        for error in source.errors(include_url=False, include_input=False)
    )


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
