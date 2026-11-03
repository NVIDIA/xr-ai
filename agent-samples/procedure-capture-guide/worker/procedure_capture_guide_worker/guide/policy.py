# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load explicit sample instructions and inference policies for guide execution.

Guide YAML owns task-specific requirements; this policy owns application
conversation, control interpretation and verifier instructions. Custom policy
files must provide the complete contract rather than inherit hidden prompts.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from xr_ai_models import DecisionQuestion


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instructions: str = Field(min_length=1)
    criteria: dict[str, str]

    def decision(self):
        return DecisionQuestion(self.instructions, dict(self.criteria))


class InferenceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_tokens: int = Field(gt=0)
    enable_thinking: bool
    thinking_budget: int | None = Field(default=None, ge=0)


class GuidePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    conversation_prompt: str = Field(min_length=1)
    vision_prompt: str = Field(min_length=1)
    measurement_prompt: str = Field(min_length=1)
    control_question: Question
    authorization_question: Question
    caption_question: Question
    image_question: Question
    guide_description: str = Field(min_length=1)
    conversation_description: str = Field(min_length=1)
    history_limit: int = Field(gt=0, le=100)
    image_confidence_threshold: float = Field(ge=0, le=1)
    foreground_options: dict
    observation_options: dict

    @field_validator("foreground_options", "observation_options")
    @classmethod
    def inference_options(cls, value):
        return InferenceOptions.model_validate(value).model_dump(exclude_none=True)

    @model_validator(mode="after")
    def choices(self):
        expected = {
            "control_question": {"advance", "skip", "restart", "reset", "status", "unsupported", "answer"},
            "authorization_question": {"command", "not_command", "reported_instruction", "explanation", "unknown"},
            "caption_question": {"accepted", "rejected", "unknown"},
            "image_question": {"accepted", "rejected", "unknown"},
        }
        for name, choices in expected.items():
            question = getattr(self, name)
            if set(question.criteria) != choices or any(not value.strip() for value in question.criteria.values()):
                raise ValueError(f"{name} must define nonempty criteria for {sorted(choices)}")
        allowed = {"max_tokens", "enable_thinking", "thinking_budget"}
        for options in (self.foreground_options, self.observation_options):
            if options.keys() - allowed:
                raise ValueError("model options may configure only output and reasoning limits")
        return self


def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[3] / "yaml/guide-policy.yaml"


def load_policy(path=None):
    source = Path(path) if path is not None else default_policy_path()
    return GuidePolicy.model_validate(yaml.safe_load(source.read_text()))
