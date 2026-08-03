# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-local native NAT groups for task state and controls."""

from typing import Any

from loguru import logger
from nat.plugin_api import Builder, FunctionGroup, FunctionGroupBaseConfig, register_function_group
from pydantic import ConfigDict, Field

from .models import TaskStatusRequest, TaskStatusResult
from .task_store import TaskStore


class TaskStateFunctionsConfig(FunctionGroupBaseConfig, name="visual_task_state"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any = Field(exclude=True, repr=False)


class TaskControlFunctionsConfig(FunctionGroupBaseConfig, name="visual_task_control"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any = Field(exclude=True, repr=False)


@register_function_group(config_type=TaskStateFunctionsConfig)
async def task_state_functions(config: TaskStateFunctionsConfig, _builder: Builder):
    store: TaskStore = config.store

    async def status(request: TaskStatusRequest) -> TaskStatusResult:
        progress = store.progress(request.participant_id)
        return TaskStatusResult(
            progress=progress,
            current_step=store.current_step(progress),
            next_step=store.next_step(progress),
        )

    group = FunctionGroup(config=config)
    group.add_function(
        "get_task_status",
        status,
        description="Return trusted current task progress and the active step.",
    )
    yield group


@register_function_group(config_type=TaskControlFunctionsConfig)
async def task_control_functions(config: TaskControlFunctionsConfig, _builder: Builder):
    store: TaskStore = config.store

    def result(participant_id: str) -> TaskStatusResult:
        progress = store.progress(participant_id)
        return TaskStatusResult(
            progress=progress,
            current_step=store.current_step(progress),
            next_step=store.next_step(progress),
        )

    async def start(request: TaskStatusRequest) -> TaskStatusResult:
        store.start(request.participant_id)
        status = result(request.participant_id)
        logger.info("task started pid={!r} revision={}", request.participant_id, status.progress.revision)
        return status

    async def reset(request: TaskStatusRequest) -> TaskStatusResult:
        store.reset(request.participant_id)
        status = result(request.participant_id)
        logger.info("task reset pid={!r} revision={}", request.participant_id, status.progress.revision)
        return status

    async def advance(request: TaskStatusRequest) -> TaskStatusResult:
        store.advance(request.participant_id)
        status = result(request.participant_id)
        logger.info(
            "task advanced pid={!r} state={} step={} revision={}",
            request.participant_id,
            status.progress.state,
            status.current_step.id if status.current_step else "complete",
            status.progress.revision,
        )
        return status

    group = FunctionGroup(config=config)
    group.add_function("start_task", start, description="Start the current participant's task at step one.")
    group.add_function("reset_task", reset, description="Reset the current participant's task to not started.")
    group.add_function(
        "advance_task",
        advance,
        description="Advance the running task by exactly one explicit step.",
    )
    yield group


__all__ = [
    "TaskControlFunctionsConfig",
    "TaskStateFunctionsConfig",
]
