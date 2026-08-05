# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose only current-frame vision as an individually selectable NAT tool."""

from typing import Any

from nat.builder.function import Function
from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_hub import FrameUnavailable
from xr_ai_nat.functions.vision import LiveVisionRequest, LiveVisionResult

from ..runtime.events import emit
from ..runtime.scope import current_invocation


class CurrentViewConfig(FunctionBaseConfig, name="tea_guidance_current_view"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Any = Field(exclude=True, repr=False)


class CurrentViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)


@register_function(config_type=CurrentViewConfig)
async def current_view(config: CurrentViewConfig, _builder: Builder):
    source: Function = config.source

    async def look(request: CurrentViewRequest) -> LiveVisionResult:
        call = current_invocation()
        try:
            return await source.ainvoke(
                LiveVisionRequest(
                    participant_id=call.session.participant_id,
                    question=request.question,
                ),
                to_type=LiveVisionResult,
            )
        except FrameUnavailable as exc:
            emit(
                "vision.unavailable",
                participant_id=call.session.participant_id,
                step=call.session.step_id,
                trace_id=call.trace_id,
                reason=str(exc),
            )
            return LiveVisionResult(answer=f"VISUAL_UNAVAILABLE: {exc}")

    yield FunctionInfo.from_fn(
        look,
        description="Inspect the participant's current camera frame for a specific visible fact.",
    )


__all__ = ["CurrentViewConfig", "CurrentViewRequest"]
