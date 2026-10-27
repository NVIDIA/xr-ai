# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample settings and paths resolved against the worker YAML."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: Path
    models: Path
    poll_interval_s: float = Field(default=0.15, gt=0, allow_inf_nan=False)
    hold_s: float = Field(default=0.6, gt=0, allow_inf_nan=False)
    max_gap_s: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    movement_tolerance: float = Field(default=0.05, gt=0, lt=1)
    release_s: float = Field(default=0.4, gt=0, allow_inf_nan=False)
    corner_tolerance: float = Field(default=0.35, gt=0, lt=1)
    angle_cosine: float = Field(default=0.45, gt=0, lt=1)
    aligned_photo: bool = False
    query: str = Field(min_length=1)


def load_settings(path: Path) -> Settings:
    settings = Settings.model_validate(yaml.safe_load(path.read_text()))
    settings.model_path = (path.parent / settings.model_path).resolve()
    settings.models = (path.parent / settings.models).resolve()
    if not settings.aligned_photo:
        raise ValueError("Verify matching photo/video camera framing and set aligned_photo: true")
    return settings
