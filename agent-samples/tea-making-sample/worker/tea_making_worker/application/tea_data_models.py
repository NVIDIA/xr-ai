# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tea-specific visual input and brewing results supplied to reusable flows."""

from __future__ import annotations

import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class TeaIdentity(BaseModel):
    """Tea name/type read from the image; an unavailable label is not an identity."""

    model_config = ConfigDict(extra="forbid")
    tea_name: str | None = Field(default=None, min_length=1, max_length=300)
    tea_type: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        description="Tea style explicitly printed in the visible text, if readable.",
    )

    @field_validator("tea_name", "tea_type", mode="before")
    @classmethod
    def normalize_readable_text(cls, value: Any) -> Any:
        """Treat unreadable optional fields as absent; either readable field suffices."""
        if not isinstance(value, str):
            return value
        value = value.strip()
        return None if value.lower() in {"", "unknown", "unreadable", "none", "null", "not visible", "n/a"} else value

    @model_validator(mode="after")
    def require_readable_identity(self) -> Self:
        """Permit a visible tea type without requiring a brand or product name."""
        if self.tea_name is None and self.tea_type is None:
            raise ValueError("tea name and type are not readable")
        return self


def _validate_quantity_source(value: float, unit: str, info: ValidationInfo) -> None:
    """Keep scalar recommendations inside explicitly cited, unit-bearing ranges.

    This application-level check recognizes numeric ranges in its brewing sources;
    it does not pick a value. A selected point is generated, never a literal
    retrieved scalar. The reusable flow knows neither tea nor these units.
    """
    context = info.context or {}
    quote = context.get("source_quote", "")
    matches = re.finditer(
        r"(-?\d+(?:\.\d+)?)\s*[–-]\s*(-?\d+(?:\.\d+)?)\s*"
        r"(°\s*[CF]|celsius|fahrenheit|min(?:ute)?s?|sec(?:ond)?s?)\b",
        quote,
        re.IGNORECASE,
    )
    temperature = unit in {"celsius", "fahrenheit"}
    ranges = []

    def base(number: float, source_unit: str) -> float:
        if source_unit == "fahrenheit":
            return (number - 32) * 5 / 9
        return number * 60 if source_unit == "minutes" else number

    for match in matches:
        source_unit = match[3].lower().replace(" ", "")
        source_unit = {"°c": "celsius", "°f": "fahrenheit"}.get(source_unit, source_unit)
        source_unit = "minutes" if source_unit.startswith("min") else source_unit
        source_unit = "seconds" if source_unit.startswith("sec") else source_unit
        if (source_unit in {"celsius", "fahrenheit"}) == temperature:
            ranges.append((base(float(match[1]), source_unit), base(float(match[2]), source_unit)))
    if not ranges:
        return
    if context.get("source_kind") == "retrieved":
        raise ValueError("A cited range requires a generated recommendation, not a literal retrieved value.")
    point = base(value, unit)
    if any(not lower <= point <= upper for lower, upper in ranges):
        raise ValueError("The recommendation is outside the cited source range.")


class TemperatureSetting(BaseModel):
    """An explicit-unit temperature from a passage or labeled recommendation."""

    model_config = ConfigDict(extra="forbid")
    value: float = Field(gt=0, allow_inf_nan=False)
    unit: Literal["celsius", "fahrenheit"]

    @model_validator(mode="after")
    def validate_source_evidence(self, info: ValidationInfo) -> Self:
        """Apply the application's source-range rules without choosing a value."""
        _validate_quantity_source(self.value, self.unit, info)
        return self

    @property
    def celsius(self) -> float:
        """Return the native Celsius target used to construct the next question."""
        return self.value if self.unit == "celsius" else (self.value - 32) * 5 / 9


class DurationSetting(BaseModel):
    """An explicit-unit brewing duration from a passage or recommendation."""

    model_config = ConfigDict(extra="forbid")
    value: float = Field(gt=0, allow_inf_nan=False)
    unit: Literal["seconds", "minutes"]

    @model_validator(mode="after")
    def validate_source_evidence(self, info: ValidationInfo) -> Self:
        """Apply the application's source-range rules without choosing a value."""
        _validate_quantity_source(self.value, self.unit, info)
        return self

    @property
    def seconds(self) -> float:
        """Return the duration passed to the native timer."""
        return self.value * 60 if self.unit == "minutes" else self.value


class BrewingSettings(BaseModel):
    """Validated partial settings; the runner supplies defaults only for missing fields."""

    model_config = ConfigDict(extra="forbid")
    temperature: TemperatureSetting | None = None
    steep_duration: DurationSetting | None = None
