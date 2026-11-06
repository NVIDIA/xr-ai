# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tea-specific visual input and brewing results supplied to reusable flows."""

from __future__ import annotations

import re
from functools import partial
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, ValidationInfo, computed_field, field_validator, model_validator


class TeaIdentity(BaseModel):
    """Tea name/type read from the image; an unavailable label is not an identity."""

    model_config = ConfigDict(extra="forbid")
    tea_name: str | None = Field(default=None, min_length=1, max_length=300)
    tea_type: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        description="Descriptive tea style explicitly printed in legible label text, not its brand or manufacturer; null when unavailable.",
    )

    @field_validator("tea_name", "tea_type", mode="before")
    @classmethod
    def normalize_readable_text(cls, value: Any) -> Any:
        """Treat unreadable optional fields as absent; either readable field suffices."""
        if not isinstance(value, str):
            return value
        value = value.strip()
        return None if value.lower() in {"", "unknown", "unreadable", "none", "null", "not visible", "n/a"} else value

    @field_validator("tea_type")
    @classmethod
    def require_descriptive_type(cls, value: str | None) -> str | None:
        """A device number is not descriptive tea-type text."""
        return value if value is None or any(character.isalpha() for character in value) else None

    @model_validator(mode="after")
    def require_readable_identity(self) -> Self:
        """Permit a visible tea type without requiring a brand or product name."""
        if self.tea_name is None and self.tea_type is None:
            raise ValueError("tea name and type are not readable")
        return self


class _TemperatureObservation(BaseModel):
    """Partial display data; unit absence and setpoints do not erase a number."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    value: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    unit_text: str | None = Field(default=None, max_length=40, description="Exact legible unit text; null when no unit glyph/word can be read.")
    role: Literal["current", "setpoint", "unknown"] = "unknown"
    role_evidence: str | None = Field(default=None, max_length=120, description="Visible label/mode indicator distinguishing current measurement from setpoint; null if not established.")

    @field_validator("role_evidence")
    @classmethod
    def retain_descriptive_mode_evidence(cls, value: str | None) -> str | None:
        """A copied numeric readout alone cannot establish measurement mode."""
        if value is not None and not any(character.isalpha() for character in value):
            return None
        return value

    @computed_field
    @property
    def unit(self) -> Literal["celsius", "fahrenheit"] | None:
        """Normalize only literal unit text, without inferring from the value."""
        text = "".join(self.unit_text.split()).casefold() if self.unit_text is not None else ""
        return {"°c": "celsius", "c": "celsius", "celsius": "celsius",
                "°f": "fahrenheit", "f": "fahrenheit", "fahrenheit": "fahrenheit"}.get(text)

    @property
    def has_other_unit(self) -> bool:
        """An explicit non-temperature unit cannot inherit a kettle's C/F setting."""
        return self.unit is None and self.unit_text is not None and self.unit_text.strip() not in {"", "°", "º"}


def _temperature_assessment_issue(reading: _TemperatureObservation, declared_unit: str | None) -> str | None:
    """Check available evidence only; completion remains the existing Clef judgment."""
    if reading.value is None:
        return "numeric_reading_unavailable"
    if reading.role == "setpoint":
        return "explicit_setpoint_not_measurement"
    if reading.has_other_unit:
        return "non_temperature_or_unrecognized_unit"
    if reading.unit is not None and declared_unit is not None and reading.unit != declared_unit:
        return "conflicting_units"
    if (reading.unit or declared_unit) is None:
        return "instrument_unit_unconfirmed"
    return None


class _RetrievedRangeRequiresRecommendation(ValueError):
    """A valid cited range is evidence, but its selected point is not literal data."""


def _validate_quantity_source(value: float, unit: str, info: ValidationInfo) -> None:
    """Keep scalar recommendations inside explicitly cited, unit-bearing ranges.

    This application-level check recognizes numeric ranges in its brewing sources;
    it does not pick a value. A selected point is generated, never a literal
    retrieved scalar. An explicit range in the recommendation's unit is
    authoritative; a rounded secondary-unit rendering must not tighten it.
    The reusable flow knows neither tea nor these units.
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
            ranges.append((source_unit, float(match[1]), float(match[2])))
    if not ranges:
        return
    if context.get("source_kind") == "retrieved":
        raise _RetrievedRangeRequiresRecommendation("A cited range requires a generated recommendation, not a literal retrieved value.")
    same_unit_ranges = [(lower, upper) for source_unit, lower, upper in ranges if source_unit == unit]
    if same_unit_ranges:
        point = value
        applicable_ranges = same_unit_ranges
    else:
        point = base(value, unit)
        applicable_ranges = [
            (base(lower, source_unit), base(upper, source_unit)) for source_unit, lower, upper in ranges
        ]
    if any(not lower <= point <= upper for lower, upper in applicable_ranges):
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


def _require_type_for_generated_setting(value: Any, info: ValidationInfo) -> Any:
    """An opaque product name cannot authorize a conventional type-based setting."""
    context = info.context or {}
    if context.get("source_kind") == "generated" and not context.get("visual_data", {}).get("tea_type"):
        return None
    return value


def _prepare_quantity(value: Any, info: ValidationInfo, *, quantity_model: type[BaseModel]) -> Any:
    """Preserve valid range evidence as unresolved, without repairing it forever."""
    value = _require_type_for_generated_setting(value, info)
    if value is None:
        return None
    try:
        return quantity_model.model_validate(value, context=info.context)
    except ValidationError as error:
        if (info.context or {}).get("source_kind") == "retrieved" and all(
            isinstance(item.get("ctx", {}).get("error"), _RetrievedRangeRequiresRecommendation)
            for item in error.errors()
        ):
            # The generic flow already retained the strictly validated quote.
            # A recommendation can use it once its application prerequisite is
            # satisfied; no invented literal setting is published meanwhile.
            return None
        raise


class BrewingSettings(BaseModel):
    """Validated partial settings; the runner supplies defaults only for missing fields."""

    model_config = ConfigDict(extra="forbid")
    temperature: Annotated[TemperatureSetting | None, BeforeValidator(partial(_prepare_quantity, quantity_model=TemperatureSetting))] = None
    steep_duration: Annotated[DurationSetting | None, BeforeValidator(partial(_prepare_quantity, quantity_model=DurationSetting))] = None
