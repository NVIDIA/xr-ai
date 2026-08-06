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


class TemperatureVerifyResult(BaseModel):
    reading_c: float
    target_c: float
    ready: bool


class TemperatureVerifyConfig(FunctionBaseConfig, name="tea_guidance_temperature_verify"):
    pass


@register_function(config_type=TemperatureVerifyConfig)
async def temperature_verify(_config: TemperatureVerifyConfig, _builder: Builder):
    async def verify(request: TemperatureVerifyRequest) -> TemperatureVerifyResult:
        call = current_invocation()
        target_c = float(call.session.state["target_temperature_c"])
        reading_c = request.reading if request.unit == "celsius" else (request.reading - 32) * 5 / 9
        result = TemperatureVerifyResult(
            reading_c=reading_c,
            target_c=target_c,
            ready=reading_c >= target_c,
        )
        emit(
            "temperature.verify",
            participant_id=call.session.participant_id,
            step=call.session.step_id,
            trace_id=call.trace_id,
            reading=request.reading,
            unit=request.unit,
            target_c=target_c,
            reading_c=reading_c,
            ready=result.ready,
        )
        return result

    yield FunctionInfo.from_fn(
        verify,
        description="Compare an exact observed Celsius or Fahrenheit reading with the active tea target.",
    )


async def add_temperature_functions(builder: Builder) -> None:
    await builder.add_function("temperature__verify", TemperatureVerifyConfig())


__all__ = [
    "TemperatureVerifyConfig",
    "TemperatureVerifyRequest",
    "TemperatureVerifyResult",
    "add_temperature_functions",
]
