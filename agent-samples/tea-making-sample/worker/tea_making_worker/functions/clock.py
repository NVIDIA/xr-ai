# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Individually selectable native clock functions."""

import math
import time
from typing import Any, Literal

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NowRequest(_Request):
    pass


class NowResult(BaseModel):
    epoch_us: int


class TimerRequest(_Request):
    started_at_us: int = Field(gt=0)
    duration_s: int = Field(gt=0)


class TimerResult(BaseModel):
    elapsed_s: int
    remaining_s: int
    expired: bool


class ClockFunctionConfig(FunctionBaseConfig, name="tea_guidance_clock_function"):
    operation: Literal["now", "timer"]


@register_function(config_type=ClockFunctionConfig)
async def clock_function(config: ClockFunctionConfig, _builder: Builder):
    async def now(request: NowRequest) -> NowResult:
        return NowResult(epoch_us=time.time_ns() // 1_000)

    async def timer(request: TimerRequest) -> TimerResult:
        elapsed_us = max(0, time.time_ns() // 1_000 - request.started_at_us)
        duration_us = request.duration_s * 1_000_000
        return TimerResult(
            elapsed_s=elapsed_us // 1_000_000,
            remaining_s=max(0, math.ceil((duration_us - elapsed_us) / 1_000_000)),
            expired=elapsed_us >= duration_us,
        )

    handlers: dict[str, tuple[Any, str]] = {
        "now": (now, "Return current Unix time in microseconds."),
        "timer": (timer, "Return fresh elapsed, remaining, and expiry values for a timer."),
    }
    handler, description = handlers[config.operation]
    yield FunctionInfo.from_fn(handler, description=description)


async def add_clock_functions(builder: Builder) -> None:
    for operation in ("now", "timer"):
        await builder.add_function(
            f"clock__{operation}",
            ClockFunctionConfig(operation=operation),
        )


__all__ = ["ClockFunctionConfig", "NowRequest", "NowResult", "TimerRequest", "TimerResult", "add_clock_functions"]
