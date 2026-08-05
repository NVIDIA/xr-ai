# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify observed temperatures through one deterministic NAT function."""

from typing import Literal

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ..runtime.events import emit
from ..runtime.scope import current_invocation


class TemperatureVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    reading: float = Field(description="Exact observed numeric temperature.")
    unit: Literal["celsius", "fahrenheit"] = Field(description="Unit shown with the observed reading.")
    target_c: float = Field(description="Target temperature from state, in Celsius.")


class TemperatureVerifyResult(BaseModel):
    ready: bool


class TemperatureVerifyConfig(FunctionBaseConfig, name="tea_guidance_temperature_verify"):
    pass


@register_function(config_type=TemperatureVerifyConfig)
async def temperature_verify(_config: TemperatureVerifyConfig, _builder: Builder):
    async def verify(request: TemperatureVerifyRequest) -> TemperatureVerifyResult:
        reading_c = request.reading if request.unit == "celsius" else (request.reading - 32) * 5 / 9
        result = TemperatureVerifyResult(ready=reading_c >= request.target_c)
        call = current_invocation()
        emit(
            "temperature.verify",
            participant_id=call.session.participant_id,
            step=call.session.step_id,
            trace_id=call.trace_id,
            reading=request.reading,
            unit=request.unit,
            target_c=request.target_c,
            reading_c=reading_c,
            ready=result.ready,
        )
        return result

    yield FunctionInfo.from_fn(
        verify,
        description="Convert an observed Celsius or Fahrenheit reading and report whether it meets the Celsius target.",
    )


async def add_temperature_functions(builder: Builder) -> None:
    await builder.add_function("temperature__verify", TemperatureVerifyConfig())


__all__ = [
    "TemperatureVerifyConfig",
    "TemperatureVerifyRequest",
    "TemperatureVerifyResult",
    "add_temperature_functions",
]
