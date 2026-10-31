# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-local, YAML-owned routing and action descriptions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..flows import ObservationPolicy


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    criterion: str = Field(min_length=1)
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    generate: bool = False
    observation: ObservationPolicy | None = None
    handoff: bool = False
    authorize: bool = False

    @model_validator(mode="after")
    def validate_action(self) -> _Action:
        if not self.handoff and not self.tool and not self.generate:
            raise ValueError("an action needs a tool, generation, or handoff")
        if self.handoff and (self.tool or self.generate or self.observation):
            raise ValueError("handoff cannot execute or generate")
        if self.observation is not None and (
            self.observation.kind != "extract"
            or self.observation.updates
            or self.observation.lookup
            or not self.generate
            or self.tool != "current_view"
        ):
            raise ValueError("foreground extraction must be a read-only current-view fact")
        return self


class _ActionQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    instructions: str = Field(min_length=1)
    actions: dict[str, _Action] = Field(min_length=1)


class _ToolDestinationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    description: str = Field(min_length=1)
    question: _ActionQuestion



class _AuthorizationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    instructions: str = Field(min_length=1)
    criteria: dict[str, str]

    @model_validator(mode="after")
    def validate_criteria(self) -> _AuthorizationQuestion:
        if set(self.criteria) != {"command", "not_command", "unknown"}:
            raise ValueError("authorization criteria must be command, not_command, unknown")
        return self


class ConversationPolicy(BaseModel):
    """YAML-owned focused admission, conversation wording and application routes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workflow_description: str = Field(min_length=1)
    active_question: _ActionQuestion
    idle_question: _ActionQuestion
    authorization: _AuthorizationQuestion
    destinations: dict[str, _ToolDestinationPolicy]
    conversation_description: str = Field(min_length=1)
    active_prompt: str = Field(min_length=1)
    conversation_prompt: str = Field(min_length=1)
    stale_message: str = Field(min_length=1)
    unavailable_message: str = Field(min_length=1)
    argument_prompt: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_names(self) -> ConversationPolicy:
        if set(self.destinations) & {"workflow", "conversation", "clarify"}:
            raise ValueError("application destination names cannot replace router-owned roles")
        return self
