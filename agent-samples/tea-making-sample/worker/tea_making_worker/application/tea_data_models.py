# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tea-specific visual input and brewing results supplied to reusable flows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
