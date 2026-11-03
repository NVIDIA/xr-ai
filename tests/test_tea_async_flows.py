# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for the independent Python SOP flows."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionQuestion, DecisionResponse
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.rag import RetrievedChunk, RetrieveResult
from xr_ai_tools.vision import ImageQueryResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample/worker"))

from tea_making_worker.application.tea_data_models import BrewingSettings, TeaIdentity  # noqa: E402
from tea_making_worker.components import (  # noqa: E402
    run_timer_flow,
    run_vision_rag_retrieval_flow,
    run_visual_verification_flow,
)


def _chat(content):
    return ChatResponse(content, None, None, "stop", {})


def _frame(sequence):
    return ImageFrame(
        image=ImageRegistry().put(b"image"),
        participant_id="participant",
        track_id="camera",
        sequence=sequence,
        timestamp_us=sequence * 1000,
        width=2,
        height=2,
    )


def _answer(choice, key="completion"):
    return DecisionResponse(
        {
            key: DecisionAnswer(
                choice, 1.0, {label: float(label == choice) for label in ("positive", "negative", "unknown")}
            )
        },
        "fixture",
        {},
        {},
    )


def _visual(choices, *, frames=None, **overrides):
    choices = list(choices)
    arguments = {
        "frame_source": AsyncMock(side_effect=frames or [_frame(index + 1) for index in range(len(choices))]),
        "observer": SimpleNamespace(
            execute=AsyncMock(
                side_effect=[ImageQueryResult(text=f"original observation {index}") for index in range(len(choices))]
            )
        ),
        "decisions": SimpleNamespace(decide=AsyncMock(side_effect=[_answer(choice) for choice in choices])),
        "question": "Read only visible evidence.",
        "completion": DecisionQuestion(
            "Judge evidence.",
            {
                "positive": "The condition is established.",
                "negative": "The condition is disproved.",
                "unknown": "The image is unreadable or evidence is unclear.",
            },
        ),
        "required_positives": 2,
        "window_size": 4,
        "poll_interval_seconds": 0.001,
        "history_limit": 8,
        "observation_text_limit": 1000,
        "timeout_seconds": 1,
    }
    arguments.update(overrides)
    return arguments


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choices, expected_window",
    [
        (["positive", "positive"], ("positive", "positive")),
        (["positive", "unknown", "negative", "positive"], ("positive", "negative", "positive")),
        (
            ["positive", "negative", "negative", "negative", "positive", "positive"],
            ("negative", "negative", "positive", "positive"),
        ),
    ],
)
async def test_visual_window_ignores_unknown_and_completes_at_threshold(choices, expected_window):
    states = []
    arguments = _visual(choices, on_state=AsyncMock(side_effect=states.append))
    result = await run_visual_verification_flow(**arguments)
    assert result.status == "completed"
    assert result.confirmation_window == expected_window
    assert result.positives_in_window == 2
    assert len(states) == len(choices)
    assert [state.status for state in states[:-1]] == ["running"] * (len(choices) - 1)
    assert arguments["observer"].execute.await_count == len(choices)
    if "unknown" in choices:
        assert any(observation.judgment == "unknown" for observation in result.observations)


@pytest.mark.asyncio
async def test_duplicate_images_cannot_supply_another_confirmation():
    arguments = _visual(["positive", "positive"], frames=[_frame(1), _frame(1), _frame(0), _frame(2)])
    result = await run_visual_verification_flow(**arguments)
    assert len(result.observations) == 2
    assert arguments["frame_source"].await_count == 4
    assert arguments["observer"].execute.await_count == 2


@pytest.mark.asyncio
async def test_visual_callback_keeps_bounded_original_context_without_changing_old_snapshots():
    states = []
    result = await run_visual_verification_flow(
        **_visual(
            ["negative", "unknown", "positive", "positive"],
            history_limit=2,
            observation_text_limit=12,
            on_state=AsyncMock(side_effect=states.append),
        )
    )
    assert len(result.observations) == 2
    assert result.observations_omitted == 2
    assert all(
        observation.description == "original obs" and observation.truncated for observation in result.observations
    )
    assert len(states[0].observations) == 1
    assert states[0].observations[0].judgment == "negative"
    assert result.question == "Read only visible evidence."
    assert result.completion_criteria["unknown"]


@pytest.mark.asyncio
async def test_visual_cancellation_cannot_publish_or_succeed_after_backend_swallows_it():
    inference_started = asyncio.Event()
    backend_released = asyncio.Event()
    states = AsyncMock()

    async def blocked_inference(_request):
        inference_started.set()
        try:
            await backend_released.wait()
        except asyncio.CancelledError:
            await backend_released.wait()
        return ImageQueryResult(text="positive late answer")

    arguments = _visual(["positive"], required_positives=1, on_state=states)
    arguments["observer"].execute = blocked_inference
    task = asyncio.create_task(run_visual_verification_flow(**arguments))
    try:
        await asyncio.wait_for(inference_started.wait(), 1)
        task.cancel()
        assert not task.done()
        backend_released.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert states.await_count == 0
        assert arguments["decisions"].decide.await_count == 0
    finally:
        backend_released.set()
        task.cancel()


@pytest.mark.asyncio
async def test_native_timer_owns_start_deadline_updates_and_completion(monkeypatch):
    import tea_making_worker.components.timer_flow as timer_module

    monotonic_seconds = [100.0]
    wall_seconds = [1000.0]
    real_sleep = asyncio.sleep

    async def advance(seconds):
        monotonic_seconds[0] += seconds
        wall_seconds[0] += seconds
        await real_sleep(0)

    monkeypatch.setattr(timer_module.time, "monotonic", lambda: monotonic_seconds[0])
    monkeypatch.setattr(timer_module.time, "time_ns", lambda: int(wall_seconds[0] * 1_000_000_000))
    monkeypatch.setattr(timer_module.asyncio, "sleep", advance)
    states = []
    execution = run_timer_flow(
        duration_seconds=3, update_interval_seconds=2, on_state=AsyncMock(side_effect=states.append)
    )
    wall_seconds[0] = 2000
    result = await execution
    assert [state.remaining_seconds for state in states] == [3, 1, 0]
    assert result.started_at_us == 2_000_000_000
    assert result.deadline_us == 2_003_000_000
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_timer_cancellation_stops_further_updates():
    first_update = asyncio.Event()
    updates = []

    async def observe(state):
        updates.append(state)
        first_update.set()

    task = asyncio.create_task(
        run_timer_flow(
            duration_seconds=100,
            update_interval_seconds=100,
            on_state=observe,
        )
    )
    await asyncio.wait_for(first_update.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(updates) == 1
    assert updates[0].status == "running"


def _vision_rag_retrieval(**overrides):
    passage = RetrievedChunk(
        source="white-tea.md",
        text="Northwind white tea: use 80 degrees Celsius.",
        score=0.9,
    )
    arguments = {
        "frame_source": AsyncMock(side_effect=[_frame(1), _frame(2)]),
        "observer": SimpleNamespace(
            execute=AsyncMock(
                side_effect=[
                    ImageQueryResult(text='{"tea_name":null}'),
                    ImageQueryResult(text='{"tea_name":"Northwind White Tea","tea_type":"white tea"}'),
                ]
            )
        ),
        "llm": SimpleNamespace(
            chat=AsyncMock(
                side_effect=[
                    _chat(
                        json.dumps(
                            {
                                "values": {"temperature": {"value": 80, "unit": "celsius"}},
                                "sources": {
                                    "temperature": {
                                        "applicability": "applicable",
                                        "passage_id": "P1",
                                        "quote": "80 degrees Celsius",
                                    }
                                },
                            }
                        )
                    ),
                    _chat(
                        json.dumps(
                            {
                                "values": {"steep_duration": {"value": 3, "unit": "minutes"}},
                            }
                        )
                    ),
                ]
            )
        ),
        "retrieval": SimpleNamespace(execute=AsyncMock(return_value=RetrieveResult(results=[passage]))),
        "visual_model": TeaIdentity,
        "result_model": BrewingSettings,
        "visual_question": "Read the visible tea name; return null if unreadable.",
        "extraction_instructions": "Extract only supported settings.",
        "fallback_instructions": "Recommend only missing settings for the identified type.",
        "retrieval_query": lambda identity: identity.tea_name + " brewing settings",
        "poll_interval_seconds": 0.001,
        "timeout_seconds": 1,
        "history_limit": 8,
        "observation_text_limit": 1000,
        "max_retrieved_passages": 2,
        "retrieved_text_limit": 2000,
    }
    arguments.update(overrides)
    return arguments


@pytest.mark.asyncio
async def test_vision_rag_retrieval_retries_then_generates_only_missing_fields():
    states = []
    arguments = _vision_rag_retrieval(on_state=AsyncMock(side_effect=states.append))
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.visual_data.tea_name == "Northwind White Tea"
    assert result.resolved_data.temperature.celsius == 80
    assert result.resolved_data.steep_duration.seconds == 180
    assert result.sources["temperature"].kind == "retrieved"
    assert result.sources["temperature"].quote == "80 degrees Celsius"
    assert result.sources["steep_duration"].kind == "generated"
    fallback_request = json.loads(arguments["llm"].chat.await_args_list[1].args[0][1].content)
    assert set(fallback_request["requested_fields"]) == {"steep_duration"}
    fallback_schema = fallback_request["response_schema"]
    assert fallback_schema["required"] == ["values"]
    assert set(fallback_schema["properties"]["values"]["properties"]) == {"steep_duration"}
    assert "DurationSetting" in fallback_schema["$defs"]
    assert arguments["retrieval"].execute.await_count == 1
    assert states[1].visual_data is None
    assert any(
        state.resolved_data.temperature is not None and state.resolved_data.steep_duration is None for state in states
    )
    assert len(result.observations) == 2
    assert result.observations[0].usable is False
    assert result.passages[0]["source"] == "white-tea.md"


@pytest.mark.asyncio
async def test_unsupported_source_quote_cannot_be_labeled_retrieved():
    arguments = _vision_rag_retrieval()
    arguments["llm"].chat.side_effect = [
        _chat(
            json.dumps(
                {
                    "values": {"temperature": {"value": 90, "unit": "celsius"}},
                    "sources": {
                        "temperature": {
                            "applicability": "applicable",
                            "passage_id": "P1",
                            "quote": "this quote does not occur",
                        }
                    },
                }
            )
        ),
        _chat(
            json.dumps(
                {
                    "values": {
                        "temperature": {"value": 80, "unit": "celsius"},
                        "steep_duration": {"value": 3, "unit": "minutes"},
                    }
                }
            )
        ),
    ]
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert all(source.kind == "generated" for source in result.sources.values())
    assert result.resolved_data.temperature.celsius == 80


@pytest.mark.asyncio
async def test_history_age_expires_context_without_expiring_confirmation_evidence(monkeypatch):
    import tea_making_worker.components.visual_verification_flow as visual_module

    now_ns = [1_000_000_000]
    monkeypatch.setattr(visual_module.time, "time_ns", lambda: now_ns[0])
    attempts = [0]
    states = []

    async def acquire():
        attempts[0] += 1
        if attempts[0] == 2:
            now_ns[0] += 2_000_000_000
            return None
        return _frame(attempts[0])

    result = await run_visual_verification_flow(
        **_visual(
            ["positive", "positive"],
            frame_source=acquire,
            history_max_age_seconds=1,
            on_state=AsyncMock(side_effect=states.append),
        )
    )
    unavailable = next(state for state in states if state.waiting_reason)
    assert unavailable.observations == ()
    assert unavailable.positives_in_window == 1
    assert unavailable.observations_omitted == 1
    assert len(result.observations) == 1
    assert result.confirmation_window == ("positive", "positive")


@pytest.mark.asyncio
async def test_unavailable_visual_answer_stays_useful_context_without_becoming_negative():
    arguments = _visual(["positive"])
    arguments["observer"].execute.side_effect = [
        ImageQueryResult(text="The requested display is occluded.", available=False),
        ImageQueryResult(text="The current reading is clear."),
    ]
    arguments["frame_source"].side_effect = [_frame(1), _frame(2)]
    arguments["required_positives"] = 1
    result = await run_visual_verification_flow(**arguments)
    assert result.observations[0].description == "The requested display is occluded."
    assert result.observations[0].judgment == "unknown"
    assert result.confirmation_window == ("positive",)


@pytest.mark.asyncio
async def test_vision_rag_retrieval_accepts_non_identity_visual_data_and_result_models():
    class GaugeReading(BaseModel):
        pressure_bar: float
        warning_light: bool

    class InspectionAdvice(BaseModel):
        action: str | None = None
        priority: int | None = None

    arguments = _vision_rag_retrieval(
        visual_model=GaugeReading,
        result_model=InspectionAdvice,
        visual_question="Read gauge pressure and warning-light status.",
        retrieval_query=lambda reading: f"pressure={reading.pressure_bar} warning={reading.warning_light}",
        extraction_instructions="Extract only documented inspection guidance.",
        fallback_instructions="Recommend only missing advice fields.",
        observer=SimpleNamespace(
            execute=AsyncMock(
                return_value=ImageQueryResult(
                    text='{"pressure_bar":2.4,"warning_light":true}',
                )
            )
        ),
        retrieval=SimpleNamespace(
            execute=AsyncMock(
                return_value=RetrieveResult(
                    results=[
                        RetrievedChunk(source="inspection.md", text="Inspect the inlet.", score=1),
                    ]
                )
            )
        ),
        llm=SimpleNamespace(
            chat=AsyncMock(
                side_effect=[
                    _chat(
                        '{"values":{"action":"Inspect the inlet."},'
                        '"sources":{"action":{"applicability":"applicable",'
                        '"passage_id":"P1","quote":"Inspect the inlet."}}}'
                    ),
                    _chat('{"values":{"priority":2}}'),
                ]
            )
        ),
    )
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.visual_data == GaugeReading(pressure_bar=2.4, warning_light=True)
    assert result.resolved_data == InspectionAdvice(action="Inspect the inlet.", priority=2)
    assert result.sources["action"].kind == "retrieved"
    assert result.sources["priority"].kind == "generated"
    assert arguments["retrieval"].execute.await_args.args[0].query == "pressure=2.4 warning=True"
    extraction = json.loads(arguments["llm"].chat.await_args_list[0].args[0][1].content)
    fallback = json.loads(arguments["llm"].chat.await_args_list[1].args[0][1].content)
    assert extraction["visual_data"] == {"pressure_bar": 2.4, "warning_light": True}
    envelope = extraction["response_schema"]
    assert extraction["requested_fields"] == ["action", "priority"]
    assert envelope["properties"]["values"]["required"] == ["action", "priority"]
    assert all("default" not in schema for schema in envelope["properties"]["values"]["properties"].values())
    assert set(envelope["required"]) == {"values", "sources"}
    assert set(envelope["properties"]["values"]["properties"]) == {"action", "priority"}
    assert envelope["properties"]["values"]["additionalProperties"] is False
    evidence_schema = envelope["properties"]["sources"]["properties"]["action"]
    assert set(evidence_schema["required"]) == {"passage_id", "quote", "applicability"}
    assert set(evidence_schema["properties"]["applicability"]["enum"]) == {"applicable", "unrelated", "unknown"}
    assert set(fallback["requested_fields"]) == {"priority"}


def _resolved_point_response():
    """A fully sourced scalar response used to verify protocol repair and caching."""
    return _chat(
        json.dumps(
            {
                "values": {
                    "temperature": {"value": 80, "unit": "celsius"},
                    "steep_duration": {"value": 3, "unit": "minutes"},
                },
                "sources": {
                    name: {"applicability": "applicable", "passage_id": "P1", "quote": quote}
                    for name, quote in (("temperature", "80 degrees Celsius"), ("steep_duration", "3 minutes"))
                },
            }
        )
    )


def _single_visual_input():
    """A usable type-only input with two literal settings in its source."""
    return _vision_rag_retrieval(
        frame_source=AsyncMock(return_value=_frame(1)),
        observer=SimpleNamespace(execute=AsyncMock(return_value=ImageQueryResult(text='{"tea_type":"white tea"}'))),
        retrieval=SimpleNamespace(
            execute=AsyncMock(
                return_value=RetrieveResult(
                    results=[
                        RetrievedChunk(source="tea.md", text="Use 80 degrees Celsius for 3 minutes.", score=1),
                    ]
                )
            )
        ),
        retrieval_query=lambda visual: visual.tea_name or visual.tea_type,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", [2, "2", "P2", ["P1"]])
async def test_source_ids_are_closed_and_invalid_ids_repair_without_reacquisition(invalid_id):
    states = []
    arguments = _single_visual_input()
    arguments["on_state"] = AsyncMock(side_effect=states.append)
    arguments["llm"].chat.side_effect = [
        _chat(
            json.dumps(
                {
                    "values": {"temperature": {"value": 80, "unit": "celsius"}},
                    "sources": {
                        "temperature": {
                            "applicability": "applicable",
                            "passage_id": invalid_id,
                            "quote": "80 degrees Celsius",
                        }
                    },
                }
            )
        ),
        _chat('{"values":{"temperature":null,"steep_duration":null}}'),
        _resolved_point_response(),
    ]
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.resolved_data.temperature.celsius == 80
    assert all(source.kind == "retrieved" for source in result.sources.values())
    arguments["frame_source"].assert_awaited_once()
    arguments["observer"].execute.assert_awaited_once()
    arguments["retrieval"].execute.assert_awaited_once()
    assert arguments["llm"].chat.await_count == 3
    payload = json.loads(arguments["llm"].chat.await_args_list[0].args[0][1].content)
    assert payload["passages"][0]["passage_id"] == "P1"
    assert payload["response_schema"]["properties"]["sources"]["properties"]["temperature"]["properties"]["passage_id"][
        "enum"
    ] == ["P1"]
    assert any("passage_id" in state.issue for state in states)


@pytest.mark.asyncio
async def test_malformed_resolution_retries_cached_input_and_propagates_feedback():
    arguments = _single_visual_input()
    arguments["llm"].chat.side_effect = [_chat("{"), _resolved_point_response()]
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.resolved_data.steep_duration.seconds == 180
    arguments["frame_source"].assert_awaited_once()
    arguments["retrieval"].execute.assert_awaited_once()
    payload = json.loads(arguments["llm"].chat.await_args_list[1].args[0][1].content)
    assert "valid JSON" in payload["validation_feedback"]["response"]


@pytest.mark.asyncio
async def test_generated_value_respects_retained_range_and_invalid_value_is_repaired():
    arguments = _single_visual_input()
    arguments["retrieval"].execute.return_value = RetrieveResult(
        results=[
            RetrievedChunk(source="tea.md", text="Use 80 degrees Celsius; steep for 2–3 minutes.", score=1),
        ]
    )
    arguments["llm"].chat.side_effect = [
        _chat(
            json.dumps(
                {
                    "values": {"temperature": {"value": 80, "unit": "celsius"}, "steep_duration": None},
                    "sources": {
                        "temperature": {
                            "applicability": "applicable",
                            "passage_id": "P1",
                            "quote": "80 degrees Celsius",
                        },
                        "steep_duration": {"applicability": "applicable", "passage_id": "P1", "quote": "2–3 minutes"},
                    },
                }
            )
        ),
        _chat('{"values":{"steep_duration":{"value":4,"unit":"minutes"}}}'),
        _chat('{"values":{"steep_duration":{"value":3,"unit":"minutes"}}}'),
    ]
    states = []
    arguments["on_state"] = AsyncMock(side_effect=states.append)
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.resolved_data.steep_duration.seconds == 180
    assert result.sources["temperature"].kind == "retrieved"
    assert result.sources["steep_duration"].kind == "generated"
    assert any("outside the cited source range" in state.issue for state in states)
    assert arguments["llm"].chat.await_count == 3
    for call in arguments["llm"].chat.await_args_list[1:]:
        payload = json.loads(call.args[0][1].content)
        assert set(payload["requested_fields"]) == {"steep_duration"}
        assert payload["field_evidence"]["steep_duration"]["quote"] == "2–3 minutes"
        assert payload["passages"]
    arguments["frame_source"].assert_awaited_once()
    arguments["retrieval"].execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_valid_null_outcome_waits_for_changed_visual_data_without_resolution_churn(monkeypatch):
    import tea_making_worker.components.vision_rag_retrieval_flow as flow_module

    clock = [1_000_000_000]
    waits = []

    async def wait(interval):
        waits.append(interval)
        clock[0] += round(interval * 1_000_000_000)

    monkeypatch.setattr(flow_module, "time", SimpleNamespace(time_ns=lambda: clock[0]))
    monkeypatch.setattr(
        flow_module,
        "asyncio",
        SimpleNamespace(
            current_task=asyncio.current_task,
            timeout=asyncio.timeout,
            CancelledError=asyncio.CancelledError,
            sleep=wait,
        ),
    )
    arguments = _single_visual_input()
    arguments["frame_source"].side_effect = [_frame(number) for number in range(1, 5)]
    arguments["observer"].execute.side_effect = [
        ImageQueryResult(text='{"tea_type":"white tea"}'),
        ImageQueryResult(text="null"),
        ImageQueryResult(text='{"tea_type":"white tea"}'),
        ImageQueryResult(text='{"tea_type":"black tea"}'),
    ]
    arguments["retrieval"].execute.return_value = RetrieveResult(results=[])
    arguments["llm"].chat.side_effect = [
        _chat('{"values":{"temperature":{"value":80,"unit":"celsius"},"steep_duration":null}}'),
        _chat('{"values":{"temperature":{"value":98,"unit":"celsius"},"steep_duration":{"value":4,"unit":"minutes"}}}'),
    ]
    states = []
    arguments["on_state"] = AsyncMock(side_effect=states.append)
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.visual_data.tea_type == "black tea"
    assert result.resolved_data.temperature.celsius == 98
    assert result.resolved_data.steep_duration.seconds == 240
    assert arguments["retrieval"].execute.await_count == 2
    assert arguments["llm"].chat.await_count == 2
    assert waits == [arguments["poll_interval_seconds"]] * 3
    assert result.updated_at_us - result.started_at_us == 3000
    assert any(
        state.resolved_data.temperature is not None and state.resolved_data.steep_duration is None for state in states
    )
    assert any(
        state.observations
        and not state.observations[-1].usable
        and state.visual_data is not None
        and state.visual_data.tea_type == "white tea"
        and state.resolved_data.temperature.celsius == 80
        for state in states
    )
    assert all(left.updated_at_us <= right.updated_at_us for left, right in zip(states, states[1:]))


def test_tea_type_is_usable_without_name_and_empty_visual_data_is_rejected():
    from pydantic import ValidationError

    assert TeaIdentity(tea_type="green tea").tea_name is None
    assert TeaIdentity(tea_name="", tea_type="green tea").tea_name is None
    with pytest.raises(ValidationError):
        TeaIdentity(tea_name="unknown", tea_type=None)


@pytest.mark.parametrize(
    "quantity, value, unit, quote",
    [
        ("duration", 180, "seconds", "2–3 minutes"),
        ("temperature", 80, "celsius", "167–180 °F"),
    ],
)
def test_app_quantities_require_generated_provenance_for_cited_ranges(quantity, value, unit, quote):
    from pydantic import TypeAdapter, ValidationError
    from tea_making_worker.application.tea_data_models import DurationSetting, TemperatureSetting

    adapter = TypeAdapter(DurationSetting if quantity == "duration" else TemperatureSetting)
    payload = {"value": value, "unit": unit}
    with pytest.raises(ValidationError, match="generated recommendation"):
        adapter.validate_python(payload, context={"source_kind": "retrieved", "source_quote": quote})
    validated = adapter.validate_python(payload, context={"source_kind": "generated", "source_quote": quote})
    assert validated.value == value


@pytest.mark.asyncio
async def test_resolution_callback_failure_is_not_mistaken_for_model_parse_failure():
    arguments = _single_visual_input()

    async def receive(state):
        if state.phase == "extracting":
            raise ValueError("callback unavailable")

    arguments["on_state"] = receive
    with pytest.raises(ValueError, match="callback unavailable"):
        await run_vision_rag_retrieval_flow(**arguments)
    arguments["llm"].chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_range_endpoint_is_generated_rather_than_mislabeled_as_retrieved():
    arguments = _single_visual_input()
    arguments["retrieval"].execute.return_value = RetrieveResult(
        results=[
            RetrievedChunk(source="tea.md", text="80 degrees Celsius; 2–3 minutes.", score=1),
        ]
    )
    arguments["llm"].chat.side_effect = [
        _chat(
            json.dumps(
                {
                    "values": {
                        "temperature": {"value": 80, "unit": "celsius"},
                        "steep_duration": {"value": 2, "unit": "minutes"},
                    },
                    "sources": {
                        "temperature": {
                            "applicability": "applicable",
                            "passage_id": "P1",
                            "quote": "80 degrees Celsius",
                        },
                        "steep_duration": {"applicability": "applicable", "passage_id": "P1", "quote": "2–3 minutes"},
                    },
                }
            )
        ),
        _chat('{"values":{"steep_duration":{"value":2.5,"unit":"minutes"}}}'),
    ]
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.resolved_data.steep_duration.seconds == 150
    assert result.sources["steep_duration"].kind == "generated"
    assert result.sources["temperature"].kind == "retrieved"
    assert arguments["llm"].chat.await_count == 2


@pytest.mark.asyncio
async def test_cancelling_a_cached_null_outcome_stops_further_callbacks_and_model_calls():
    arguments = _single_visual_input()
    arguments["frame_source"].side_effect = [_frame(1), _frame(2)]
    arguments["retrieval"].execute.return_value = RetrieveResult(results=[])
    arguments["llm"].chat.side_effect = [_chat('{"values":{"temperature":null,"steep_duration":null}}')]
    received = []

    async def receive(state):
        received.append(state)
        if len(state.observations) == 2:
            raise asyncio.CancelledError

    arguments["on_state"] = receive
    with pytest.raises(asyncio.CancelledError):
        await run_vision_rag_retrieval_flow(**arguments)
    assert arguments["llm"].chat.await_count == 1
    arguments["retrieval"].execute.assert_awaited_once()
    assert received[-1].observations[-1].usable


@pytest.mark.asyncio
@pytest.mark.parametrize("applicability", ["unrelated", "unknown"])
async def test_inapplicable_evidence_neither_supplies_values_nor_constrains_generation(applicability):
    arguments = _single_visual_input()
    text = "Another item requires 96–100°C and 3–5 minutes."
    arguments["retrieval"].execute.return_value = RetrieveResult(
        results=[RetrievedChunk(source="other.md", text=text, score=1)]
    )
    arguments["llm"].chat.side_effect = [
        _chat(
            json.dumps(
                {
                    "values": {"temperature": {"value": 100, "unit": "celsius"}},
                    "sources": {
                        "temperature": {
                            "applicability": applicability,
                            "passage_id": "P1",
                            "quote": text,
                        }
                    },
                }
            )
        ),
        _chat('{"values":{"temperature":{"value":80,"unit":"celsius"},"steep_duration":{"value":2,"unit":"minutes"}}}'),
    ]
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.resolved_data.temperature.celsius == 80
    assert all(source.kind == "generated" for source in result.sources.values())
    generation = json.loads(arguments["llm"].chat.await_args_list[1].args[0][1].content)
    assert generation["field_evidence"] == {}
    assert arguments["llm"].chat.await_count == 2
    arguments["retrieval"].execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("applicability", [None, []])
async def test_invalid_evidence_applicability_repairs_cached_input(applicability):
    arguments = _single_visual_input()
    arguments["llm"].chat.side_effect = [
        _chat(
            json.dumps(
                {
                    "values": {"temperature": {"value": 80, "unit": "celsius"}},
                    "sources": {
                        "temperature": {
                            "applicability": applicability,
                            "passage_id": "P1",
                            "quote": "80 degrees Celsius",
                        }
                    },
                }
            )
        ),
        _resolved_point_response(),
    ]
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.resolved_data.temperature.celsius == 80
    assert arguments["llm"].chat.await_count == 2
    arguments["observer"].execute.assert_awaited_once()
    arguments["retrieval"].execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("description", ["null", '{"not_tea":"laptop"}'])
async def test_no_fallback_without_usable_visual_identity(description):
    arguments = _single_visual_input()
    arguments["observer"].execute.return_value = ImageQueryResult(text=description)

    async def receive(state):
        if state.observations:
            assert state.visual_data is None and not state.observations[-1].usable
            raise asyncio.CancelledError

    arguments["on_state"] = receive
    with pytest.raises(asyncio.CancelledError):
        await run_vision_rag_retrieval_flow(**arguments)
    arguments["llm"].chat.assert_not_awaited()
    arguments["retrieval"].execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_retrieved_settings_do_not_require_an_observed_tea_type():
    arguments = _single_visual_input()
    arguments["observer"].execute.return_value = ImageQueryResult(text='{"tea_name":"Moon Orchard"}')
    arguments["llm"].chat.side_effect = [_resolved_point_response()]
    result = await run_vision_rag_retrieval_flow(**arguments)
    assert result.visual_data.tea_type is None
    assert result.resolved_data.temperature.celsius == 80
    assert result.resolved_data.steep_duration.seconds == 180
    assert all(source.kind == "retrieved" for source in result.sources.values())
    arguments["llm"].chat.assert_awaited_once()


def test_model_facing_schema_defaults_are_removed_without_deleting_named_fields():
    from tea_making_worker.components.vision_rag_retrieval_flow import _without_schema_defaults

    original = {
        "properties": {"default": {"type": ["string", "null"], "default": None}},
        "$defs": {"Choice": {"properties": {"value": {"type": "integer", "default": 9}}}},
    }
    rendered = _without_schema_defaults(original)
    assert rendered["properties"]["default"] == {"type": ["string", "null"]}
    assert rendered["$defs"]["Choice"]["properties"]["value"] == {"type": "integer"}
    assert original["properties"]["default"]["default"] is None
    assert original["$defs"]["Choice"]["properties"]["value"]["default"] == 9
