# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tea application's Python inputs and validated results, not an execution language.

The reusable vision/RAG retrieval flow accepts these ordinary Pydantic models. Only
this application knows that its values are tea names, temperatures and durations.
The runner's procedure below connects the resulting values explicitly in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from xr_ai_models import DecisionQuestion


class TeaIdentity(BaseModel):
    """Tea name/type read from the image; an unavailable label is not an identity."""

    model_config = ConfigDict(extra="forbid")
    tea_name: str = Field(min_length=1, max_length=300)

    @field_validator("tea_name")
    @classmethod
    def require_readable_identity(cls, value: str) -> str:
        """Reject unavailable markers rather than passing them to retrieval."""
        value = value.strip()
        if value.lower() in {"", "unknown", "unreadable", "none", "null", "not visible", "n/a"}:
            raise ValueError("tea identity is not readable")
        return value


class TemperatureSetting(BaseModel):
    """An explicit-unit temperature from a passage or labeled recommendation."""

    model_config = ConfigDict(extra="forbid")
    value: float = Field(gt=0, allow_inf_nan=False)
    unit: Literal["celsius", "fahrenheit"]

    @property
    def celsius(self) -> float:
        """Return the native Celsius target used to construct the next question."""
        return self.value if self.unit == "celsius" else (self.value - 32) * 5 / 9


class DurationSetting(BaseModel):
    """An explicit-unit brewing duration from a passage or recommendation."""

    model_config = ConfigDict(extra="forbid")
    value: float = Field(gt=0, allow_inf_nan=False)
    unit: Literal["seconds", "minutes"]

    @property
    def seconds(self) -> float:
        """Return the duration passed to the native timer."""
        return self.value * 60 if self.unit == "minutes" else self.value


class BrewingSettings(BaseModel):
    """Validated partial settings; the runner supplies defaults only for missing fields."""

    model_config = ConfigDict(extra="forbid")
    temperature: TemperatureSetting | None = None
    steep_duration: DurationSetting | None = None


@dataclass(frozen=True, slots=True)
class VisualInstructions:
    """Application wording passed directly to a generic visual flow."""

    question: str
    completion: DecisionQuestion


@dataclass(frozen=True, slots=True)
class SOPConfig:
    """Python application policy for the explicit tea procedure.

    YAML loading is an optional application adapter constructing this object.
    No flow accepts this configuration or interprets its fields as step commands.
    """

    identification_question: str
    extraction_instructions: str
    fallback_instructions: str
    retrieval_query_suffix: str
    water: VisualInstructions
    temperature: VisualInstructions
    tea_bag: VisualInstructions
    control_question: DecisionQuestion
    authorization_question: DecisionQuestion
    workflow_description: str
    conversation_description: str
    conversation_instructions: str
    messages: dict[str, str]
    default_tea_name: str
    default_temperature_c: float
    default_duration_seconds: float
    advancement: Literal["automatic", "manual"]
    poll_interval_seconds: float
    inference_timeout_seconds: float
    required_positives: int
    confirmation_window_size: int
    history_limit: int
    observation_text_limit: int
    history_max_age_seconds: float | None
    max_retrieved_passages: int
    retrieved_text_limit: int
    timer_update_interval_seconds: float
