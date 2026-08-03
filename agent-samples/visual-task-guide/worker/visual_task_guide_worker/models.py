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
    progress: TaskProgress
    current_step: TaskStep | None
    next_step: TaskStep | None


class KnowledgeSearchRequest(StrictModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=4, ge=1, le=8)


class KnowledgeResult(StrictModel):
    citation: str
    text: str


class KnowledgeSearchResult(StrictModel):
    results: list[KnowledgeResult]


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
    "KnowledgeResult",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResult",
    "GuideAgentRequest",
    "TaskGuideReply",
    "TaskGuideRequest",
    "TaskStatusRequest",
    "TaskStatusResult",
]
