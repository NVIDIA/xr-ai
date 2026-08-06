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
from ..runtime.state import Session
from ..spec import Step, Workflow
from .prompts import GENERAL, HUMAN, ROUTER, STEP, VOICE

_COMMIT = FunctionRef("workflow__commit")
_ROUTES = tuple(
    FunctionRef(f"workflow__{name}")
    for name in ("start", "advance", "reset", "status", "ask_step", "ask_general")
)


class AgentRegistry:
    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self._step: dict[str, Function] = {}
        self._voice: dict[str, Function] = {}
        self._general: Function | None = None
        self._router: Function | None = None

    async def build(self, builder: WorkflowBuilder, llm: LLMService) -> None:
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
        for step in self.workflow.steps.values():
            self._step[step.id] = await self._agent(
                builder,
                name=f"observe_{step.id}",
                llm_ref=llm_ref,
                prompt=f"{STEP}\n{_state_contract(self.workflow, step)}\n{step.agent.prompt}\n{HUMAN}",
                tools=(_COMMIT, *map(FunctionRef, step.agent.tools)),
                return_direct=(_COMMIT,),
            )
            self._voice[step.id] = await self._agent(
                builder,
                name=f"voice_{step.id}",
                llm_ref=llm_ref,
                prompt=f"{VOICE}\n{step.voice.prompt}\n{HUMAN}",
                tools=tuple(map(FunctionRef, step.voice.tools)),
            )
        self._general = await self._agent(
            builder,
            name="guide_general",
            llm_ref=llm_ref,
            prompt=f"{GENERAL}\n{HUMAN}",
            tools=(FunctionRef("current_view"), FunctionRef("rag_lookup")),
        )
        self._router = await self._agent(
            builder,
            name="guide_router",
            llm_ref=llm_ref,
            prompt=f"{ROUTER}\n{self.workflow.router_prompt}".strip(),
            tools=_ROUTES,
            return_direct=_ROUTES,
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

    async def answer(self, session: Session, question: str, trace_id: str) -> str:
        if not session.active or session.step_id is None:
            return f"{self.workflow.name} guidance is idle. Ask me to start when you are ready."
        step = self._active_step(session)
        request = _json(
            question=question,
            step=step.title,
            state=self.workflow.project(step, session.state),
        )
        emit(
            "agent.voice.request",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            chars=len(request),
            input=request,
        )
        result = await self._voice[step.id].ainvoke(request, to_type=str)
        emit(
            "agent.voice.response",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            output=result,
        )
        return result.strip()

    async def answer_general(self, session: Session, question: str, trace_id: str) -> str:
        if self._general is None:
            raise RuntimeError("agent registry has not been built")
        request = _json(question=question)
        emit(
            "agent.general.request",
            participant_id=session.participant_id,
            step=session.step_id,
            trace_id=trace_id,
            chars=len(request),
            input=request,
        )
        result = await self._general.ainvoke(request, to_type=str)
        emit(
            "agent.general.response",
            participant_id=session.participant_id,
            step=session.step_id,
            trace_id=trace_id,
            output=result,
        )
        return result.strip()

    async def route(self, session: Session, request: str, trace_id: str) -> str:
        if self._router is None:
            raise RuntimeError("agent registry has not been built")
        step = self.workflow.step(session.step_id) if session.active and session.step_id else None
        payload = _json(
            request=request,
            active=session.active,
            step=step.title if step else None,
        )
        emit(
            "agent.router.request",
            participant_id=session.participant_id,
            trace_id=trace_id,
            chars=len(payload),
            input=payload,
        )
        result = await self._router.ainvoke(payload, to_type=str)
        emit(
            "agent.router.response",
            participant_id=session.participant_id,
            trace_id=trace_id,
            output=result,
        )
        return result.strip()

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
