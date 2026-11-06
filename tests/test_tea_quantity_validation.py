# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-unit authority at the application's native quantity boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from xr_ai_models import ChatResponse
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.rag import RetrievedChunk, RetrieveResult
from xr_ai_tools.vision import ImageQueryResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample/worker"))

from tea_making_worker.application.tea_data_models import (  # noqa: E402
    BrewingSettings,
    DurationSetting,
    TeaIdentity,
    TemperatureSetting,
)
from tea_making_worker.components import run_vision_rag_retrieval_flow  # noqa: E402

_BLACK_TEA_RANGE = "Black tea | 96–100 °C / 205–212 °F | 3–5 minutes"


@pytest.mark.parametrize("quote", (_BLACK_TEA_RANGE, "Black tea | 205–212 °F / 96–100 °C | 3–5 minutes"))
@pytest.mark.parametrize("value, unit", ((96, "celsius"), (100, "celsius"), (205, "fahrenheit"), (212, "fahrenheit")))
def test_explicit_same_unit_endpoints_are_valid_despite_rounded_secondary_units(quote, value, unit):
    result = TemperatureSetting.model_validate(
        {"value": value, "unit": unit}, context={"source_kind": "generated", "source_quote": quote}
    )
    assert result.value == value and result.unit == unit


@pytest.mark.parametrize(
    "value, unit", ((95.99, "celsius"), (100.01, "celsius"), (204.99, "fahrenheit"), (212.01, "fahrenheit"))
)
def test_same_unit_bounds_remain_strict(value, unit):
    with pytest.raises(ValidationError, match="outside the cited source range"):
        TemperatureSetting.model_validate(
            {"value": value, "unit": unit},
            context={"source_kind": "generated", "source_quote": _BLACK_TEA_RANGE},
        )


def test_cross_unit_conversion_applies_only_when_no_matching_unit_range_exists():
    context = {"source_kind": "generated", "source_quote": "Black tea | 205–212 °F"}
    with pytest.raises(ValidationError, match="outside the cited source range"):
        TemperatureSetting.model_validate({"value": 96, "unit": "celsius"}, context=context)
    result = TemperatureSetting.model_validate({"value": 97, "unit": "celsius"}, context=context)
    assert result.value == 97


def test_range_selected_value_is_still_generated_and_conflicting_same_unit_ranges_are_not_ignored():
    with pytest.raises(ValidationError, match="generated recommendation"):
        TemperatureSetting.model_validate(
            {"value": 96, "unit": "celsius"},
            context={"source_kind": "retrieved", "source_quote": _BLACK_TEA_RANGE},
        )
    with pytest.raises(ValidationError, match="outside the cited source range"):
        TemperatureSetting.model_validate(
            {"value": 96, "unit": "celsius"},
            context={"source_kind": "generated", "source_quote": "96–100 °C; 80–90 °C"},
        )


def test_duration_conversion_and_source_dimension_separation_are_preserved():
    context = {"source_kind": "generated", "source_quote": _BLACK_TEA_RANGE}
    assert DurationSetting.model_validate({"value": 240, "unit": "seconds"}, context=context).seconds == 240
    with pytest.raises(ValidationError, match="outside the cited source range"):
        DurationSetting.model_validate({"value": 360, "unit": "seconds"}, context=context)


@pytest.mark.asyncio
async def test_real_flow_resolves_the_field_range_endpoint_without_repairing_valid_generation():
    """The actual source envelope and generated96C complete in two language calls."""
    source = {"applicability": "applicable", "passage_id": "P1", "quote": _BLACK_TEA_RANGE}
    responses = (
        {
            "values": {"temperature": None, "steep_duration": None},
            "sources": {"temperature": source, "steep_duration": source},
        },
        {
            "values": {
                "temperature": {"value": 96, "unit": "celsius"},
                "steep_duration": {"value": 4, "unit": "minutes"},
            }
        },
    )
    language = SimpleNamespace(
        chat=AsyncMock(side_effect=[ChatResponse(json.dumps(value), None, None, "stop", {}) for value in responses])
    )
    frame = ImageFrame(
        image=ImageRegistry().put(b"synthetic image boundary"),
        participant_id="p",
        track_id="camera",
        sequence=1,
        timestamp_us=1000,
        width=2,
        height=2,
    )
    result = await run_vision_rag_retrieval_flow(
        frame_source=AsyncMock(return_value=frame),
        observer=SimpleNamespace(execute=AsyncMock(return_value=ImageQueryResult(text='{"tea_type":"black tea"}'))),
        llm=language,
        retrieval=SimpleNamespace(
            execute=AsyncMock(
                return_value=RetrieveResult(results=[RetrievedChunk(text=_BLACK_TEA_RANGE, source="tea.md", score=1)])
            )
        ),
        visual_model=TeaIdentity,
        result_model=BrewingSettings,
        visual_question="Read the tea type or return null.",
        extraction_instructions="Extract only source-supported fields.",
        fallback_instructions="Recommend missing fields within their documented ranges.",
        retrieval_query=lambda _identity: "black tea settings",
        poll_interval_seconds=0.001,
        timeout_seconds=1,
        history_limit=4,
        observation_text_limit=1000,
        max_retrieved_passages=2,
        retrieved_text_limit=2000,
    )
    assert result.status == "completed"
    assert result.resolved_data.temperature.value == 96
    assert result.resolved_data.steep_duration.seconds == 240
    assert result.sources["temperature"].kind == "generated"
    assert result.sources["steep_duration"].kind == "generated"
    assert language.chat.await_count == 2
