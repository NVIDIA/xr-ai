# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed requests and results exchanged by the scene workflow."""

from pydantic import BaseModel, ConfigDict, Field


class SceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transcript: str
    participant_id: str = ""
    timestamp_us: int = 0
    trace_id: str = ""


class SceneReply(BaseModel):
    response: str


class SubagentTask(BaseModel):
    """Self-contained task passed from the supervisor to one focused agent.

    Model-visible: this docstring is exposed in the subagent tool schema
    and shapes how the supervisor phrases delegations. Participant identity
    and the utterance timestamp are bound per turn by the supervisor, never
    copied by the model.
    """

    instruction: str = Field(description="Focused task including facts returned by earlier subagents.")


class SubagentResult(BaseModel):
    """Focused result returned to the supervisor for further planning."""

    result: str
    handled: bool = True
    """Whether this subagent accepted the delegated task as its responsibility."""

    suggested_owner: str | None = None
    """Optional domain or agent hint when the task was outside this agent's scope."""


__all__ = ["SceneReply", "SceneRequest", "SubagentResult", "SubagentTask"]
