# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the external workflow's conversation and routing policy.

The YAML supplies model instructions, bounded choices and destination descriptions.
These models enforce the supported choice sets; they do not execute actions or
translate arbitrary model output into state changes.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] = Field(min_length=1)


class _DestinationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    description: str = Field(min_length=1)
    question: _Question | None = None


class ConversationPolicy(BaseModel):
    """YAML-owned SOP control criteria, destination descriptions and answer wording."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workflow_description: str = Field(min_length=1)
    active_question: _Question
    idle_question: _Question
    authorization: _Question
    destinations: dict[str, _DestinationPolicy]
    conversation_description: str = Field(min_length=1)
    active_prompt: str = Field(min_length=1)
    conversation_prompt: str = Field(min_length=1)
    stale_message: str = Field(min_length=1)
    argument_prompt: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_choices(self):
        expected = (
            (self.active_question, {"advance", "skip", "restart", "handoff", "reset", "answer"}),
            (self.idle_question, {"start", "status", "other"}),
            (self.authorization, {"command", "not_command", "unknown"}),
        )
        for question, choices in expected:
            if set(question.criteria) != choices or any(not text.strip() for text in question.criteria.values()):
                raise ValueError(f"question requires nonblank criteria for {sorted(choices)}")
        if set(self.destinations) - {"watch", "transcript", "journal", "observation", "reference", "history"}:
            raise ValueError("unknown sample application destination")
        for name in self.destinations.keys() & {"watch", "transcript", "journal"}:
            question = self.destinations[name].question
            if question is None or set(question.criteria) != {"start", "stop", "status", "other"}:
                raise ValueError(f"{name} requires start, stop, status and other criteria")
        return self
