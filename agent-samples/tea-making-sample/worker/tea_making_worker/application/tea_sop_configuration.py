# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain Python arguments for the tea procedure; application YAML is optional."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from xr_ai_models import DecisionQuestion


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
