# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample settings and paths resolved against the worker YAML."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CircleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    pinch_ratio: float = Field(default=0.25, gt=0)
    release_ratio: float = Field(default=0.45, gt=0)
    min_points: int = Field(default=8, ge=4)
    max_points: int = Field(default=256, ge=4)
    min_diameter: float = Field(default=0.1, gt=0, le=1)
    min_axis_ratio: float = Field(default=0.65, gt=0, le=1)
    radial_tolerance: float = Field(default=0.35, gt=0, lt=1)
    closure_tolerance: float = Field(default=1.0, gt=0, lt=2)
    min_sweep_deg: float = Field(default=300, ge=300, le=360)
    clear_timeout_s: float = Field(default=2.0, gt=0)
    clear_overlap: float = Field(default=0.2, ge=0, lt=1)
    max_backtrack: float = Field(default=0.15, ge=0, lt=1)
    max_segment_radius: float = Field(default=1.0, gt=0, lt=2)
    max_jump_palm: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def _validate_limits(self):
        if self.release_ratio <= self.pinch_ratio:
            raise ValueError("circle release_ratio must exceed pinch_ratio")
        if self.max_points < self.min_points:
            raise ValueError("circle max_points must be at least min_points")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: Path
    models: Path
    poll_interval_s: float = Field(default=0.08, gt=0, allow_inf_nan=False)
    hold_s: float = Field(default=0.6, gt=0, allow_inf_nan=False)
    max_gap_s: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    movement_tolerance: float = Field(default=0.05, gt=0, lt=1)
    release_s: float = Field(default=0.4, gt=0, allow_inf_nan=False)
    corner_tolerance: float = Field(default=0.35, gt=0, lt=1)
    angle_cosine: float = Field(default=0.45, gt=0, lt=1)
    gesture_audio: bool = True
    gesture_diagnostics: bool = False
    finger_clearance_ratio: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    circle: CircleSettings = Field(default_factory=CircleSettings)
    aligned_photo: bool = False
    query: str = Field(min_length=1)


def load_settings(path: Path) -> Settings:
    settings = Settings.model_validate(yaml.safe_load(path.read_text()))
    settings.model_path = (path.parent / settings.model_path).resolve()
    settings.models = (path.parent / settings.models).resolve()
    return settings
