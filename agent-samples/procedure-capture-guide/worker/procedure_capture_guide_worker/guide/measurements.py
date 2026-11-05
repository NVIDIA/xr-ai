# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract visible measurements and compare them without model-written arithmetic.

A guide declares explicit unit-specific ranges. Cosmos supplies readings through
schema-constrained decoding; native code validates and compares those readings.
This module never selects frames or changes procedure progress.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from xr_ai_models import ModelsConfig, VLMService, make_vlm


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    value: float
    unit: str
    role: Literal["current", "target", "unlabeled"]


class Readings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    readings: list[Reading]


class MeasurementRange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    unit: str = Field(min_length=1)
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum or self.unit != self.unit.strip():
            raise ValueError("measurement ranges require ordered bounds and an exact unit")
        return self


class MeasurementPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ranges: list[MeasurementRange] = Field(min_length=1)
    prefer_current: bool = True

    @model_validator(mode="after")
    def unique_units(self):
        units = [band.unit for band in self.ranges]
        if len(set(units)) != len(units):
            raise ValueError("measurement ranges must use distinct units")
        return self


@dataclass(frozen=True)
class MeasurementResult:
    verdict: Literal["accepted", "rejected", "unknown"]
    readings: list[dict]


MEASUREMENT_PROMPT = """Read only the measurement displays requested by the user question.
Copy visible numbers and units exactly; do not compare to a target, convert units,
or infer values from expected settings. CURRENT/ACTUAL and measured thermometer
readings have role current; TARGET/SET readings have role target; otherwise use
unlabeled. Include every relevant display so a conflicting current value is not
hidden by a target. Omit unreadable values or values without an explicit unit.
Return only the required JSON readings object. Empty readings means unavailable.
"""


def make_measurement_vlm(models: ModelsConfig) -> VLMService:
    """Use the configured Cosmos endpoint with a per-client decoding contract."""
    spec = models.vlm("vlm")
    extras = dict(spec.adapter.default_extras)
    extras["response_format"] = {
        "type": "json_schema", "json_schema": {
            "name": "procedure_measurements", "strict": True,
            "schema": Readings.model_json_schema(),
        },
    }
    entries = dict(models.entries)
    entries["vlm"] = replace(spec, adapter=replace(spec.adapter, default_extras=extras))
    return make_vlm(ModelsConfig(entries=entries), "vlm")


def compare_measurements(text: str, policy: MeasurementPolicy) -> MeasurementResult:
    """Preserve unknown separately from a readable, contradictory measurement."""
    # Strict parsing rejects prose, invented fields, booleans, and nonfinite numbers.
    readings = Readings.model_validate(json.loads(text)).readings
    if not readings:
        return MeasurementResult("unknown", [])
    selected = readings
    current = [reading for reading in readings if reading.role == "current"]
    if policy.prefer_current and current:
        selected = current
    def matches(reading):
        return any(reading.unit == band.unit and band.minimum <= reading.value <= band.maximum
                   for band in policy.ranges)
    # Every selected current reading must agree. Otherwise one qualifying display
    # suffices; units are case-sensitive (Nm and nm are different quantities).
    accepted = all(map(matches, selected)) if policy.prefer_current and current else any(map(matches, selected))
    return MeasurementResult("accepted" if accepted else "rejected",
                             [reading.model_dump() for reading in readings])
