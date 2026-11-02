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
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionQuestion, DecisionResponse
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.rag import RetrievedChunk, RetrieveResult
from xr_ai_tools.vision import ImageQueryResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample/worker"))

from tea_making_worker import BrewingSettings, TeaIdentity  # noqa: E402
from tea_making_worker.flows import (  # noqa: E402
    run_identification_flow,
    run_timer_flow,
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
    import tea_making_worker.flows._timer as timer_module

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


def _identity(**overrides):
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
                    ImageQueryResult(text='{"tea_name":"Northwind White Tea"}'),
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
                                "sources": {"temperature": {"passage": 0, "quote": "80 degrees Celsius"}},
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
        "identity_model": TeaIdentity,
        "settings_model": BrewingSettings,
        "identification_question": "Read the visible tea name; return null if unreadable.",
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
async def test_identification_retries_then_retrieves_and_generates_only_missing_settings():
    states = []
    arguments = _identity(on_state=AsyncMock(side_effect=states.append))
    result = await run_identification_flow(**arguments)
    assert result.identity.tea_name == "Northwind White Tea"
    assert result.settings.temperature.celsius == 80
    assert result.settings.steep_duration.seconds == 180
    assert result.sources["temperature"].kind == "retrieved"
    assert result.sources["temperature"].quote == "80 degrees Celsius"
    assert result.sources["steep_duration"].kind == "generated"
    fallback_request = json.loads(arguments["llm"].chat.await_args_list[1].args[0][1].content)
    assert set(fallback_request["requested_fields"]) == {"steep_duration"}
    assert arguments["retrieval"].execute.await_count == 1
    assert states[1].identity is None
    assert any(state.settings.temperature is not None and state.settings.steep_duration is None for state in states)
    assert len(result.observations) == 2
    assert result.observations[0].usable is False
    assert result.passages[0]["source"] == "white-tea.md"


@pytest.mark.asyncio
async def test_unsupported_source_quote_cannot_be_labeled_retrieved():
    arguments = _identity()
    arguments["llm"].chat.side_effect = [
        _chat(
            json.dumps(
                {
                    "values": {"temperature": {"value": 90, "unit": "celsius"}},
                    "sources": {"temperature": {"passage": 0, "quote": "this quote does not occur"}},
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
    result = await run_identification_flow(**arguments)
    assert all(source.kind == "generated" for source in result.sources.values())
    assert result.settings.temperature.celsius == 80


@pytest.mark.asyncio
async def test_history_age_expires_context_without_expiring_confirmation_evidence(monkeypatch):
    import tea_making_worker.flows._visual as visual_module

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
