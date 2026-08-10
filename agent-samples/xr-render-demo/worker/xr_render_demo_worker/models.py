# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed requests and results exchanged by the scene workflow."""

from pydantic import BaseModel, ConfigDict, Field


class SceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transcript: str
    participant_id: str = ""
    timestamp_us: int = 0


class SceneReply(BaseModel):
    response: str


class SubagentTask(BaseModel):
    """Self-contained task passed from the supervisor to one focused agent.

    Model-visible: NAT lifts this docstring into every subagent's tool
    schema, so it shapes how the supervisor phrases delegations.
    """

    instruction: str = Field(description="Focused task including facts returned by earlier subagents.")
    participant_id: str = Field(description="Active participant ID copied from the user request.")
    reference_time_us: int = Field(default=0, description="Timestamp of the active user utterance.")


class SubagentResult(BaseModel):
    """Focused result returned to the supervisor for further planning."""

    result: str


__all__ = ["SceneReply", "SceneRequest", "SubagentResult", "SubagentTask"]
