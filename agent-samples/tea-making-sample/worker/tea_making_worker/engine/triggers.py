# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invoke any YAML-selected NAT function as a step trigger."""

from __future__ import annotations

import time
from typing import Any

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from pydantic import BaseModel

from ..runtime.events import emit
from ..runtime.state import Session
from ..spec import Step, Workflow


class TriggerRegistry:
    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self._functions: dict[str, Function] = {}

    async def build(self, builder: WorkflowBuilder) -> None:
        for name in {step.trigger.function for step in self.workflow.steps.values()}:
            self._functions[name] = await builder.get_function(name)

    async def invoke(self, session: Session, step: Step, trace_id: str) -> Any:
        arguments = _resolve(step.trigger.arguments, session)
        started = time.monotonic()
        emit(
            "trigger.request",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            function=step.trigger.function,
            arguments=arguments,
        )
        result = _plain(await self._functions[step.trigger.function].ainvoke(arguments))
        if step.trigger.result_field:
            result = result[step.trigger.result_field]
        emit(
            "trigger.response",
            participant_id=session.participant_id,
            step=step.id,
            trace_id=trace_id,
            function=step.trigger.function,
            latency_ms=round((time.monotonic() - started) * 1_000),
            result=result,
        )
        return result


def _resolve(value: Any, session: Session) -> Any:
    if value == "$participant_id":
        return session.participant_id
    if isinstance(value, str) and value.startswith("$state."):
        return session.state[value.removeprefix("$state.")]
    if isinstance(value, dict):
        return {key: _resolve(item, session) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, session) for item in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


__all__ = ["TriggerRegistry"]
