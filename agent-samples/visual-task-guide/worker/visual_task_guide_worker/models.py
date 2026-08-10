# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed task-guide function contracts."""

from pydantic import BaseModel, ConfigDict, Field

from .task_store import TaskProgress, TaskStep


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskStatusRequest(StrictModel):
    participant_id: str


class TaskStatusResult(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    progress: TaskProgress
    current_step: TaskStep | None
    next_step: TaskStep | None


class GuideAgentRequest(StrictModel):
    participant_id: str
    user_text: str
    latest_observation: str | None = None


class TaskGuideRequest(StrictModel):
    participant_id: str
    text: str = Field(min_length=1, max_length=1_000)


class TaskGuideReply(StrictModel):
    response: str


__all__ = [
    "GuideAgentRequest",
    "TaskGuideReply",
    "TaskGuideRequest",
    "TaskStatusRequest",
    "TaskStatusResult",
]
