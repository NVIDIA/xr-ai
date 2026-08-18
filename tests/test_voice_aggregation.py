# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for participant-scoped voice output aggregation."""

from __future__ import annotations

import asyncio

import pytest
from xr_ai_models import ChatMessage, ChatResponse
from xr_ai_runtime import Agent, AgentRuntime, MessageMetadata, RuntimeContext, subscribe
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    VOICE_OUTPUT_TOPIC,
    VoiceAggregationAgent,
    VoiceOutput,
)


class _LLM:
    def __init__(self, content: str = "Combined update.", *, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[tuple[tuple[ChatMessage, ...], dict]] = []

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.calls.append((tuple(messages), kwargs))
        if self.error is not None:
            raise self.error
        return ChatResponse(self.content, None, None, "stop", {})


class _Recorder(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[tuple[VoiceOutput, MessageMetadata]] = []
        self.changed = asyncio.Condition()

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def record(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        async with self.changed:
            self.outputs.append((output, ctx.metadata))
            self.changed.notify_all()

    async def wait_for(self, count: int) -> None:
        async with self.changed:
            await asyncio.wait_for(
                self.changed.wait_for(lambda: len(self.outputs) >= count),
                1.0,
            )


async def _start(
    llm: _LLM,
    **kwargs,
) -> tuple[AgentRuntime, VoiceAggregationAgent, _Recorder]:
    runtime = AgentRuntime()
    aggregator = runtime.register(
        "voice-aggregation",
        VoiceAggregationAgent(
            llm=llm,  # type: ignore[arg-type]
            coalesce_window_s=0.01,
            participant_idle_timeout_s=1.0,
            **kwargs,
        ),
    )
    recorder = runtime.register("recorder", _Recorder())
    await runtime.start()
    return runtime, aggregator, recorder


async def _stop(runtime: AgentRuntime, aggregator: VoiceAggregationAgent) -> None:
    await aggregator.stop()
    await runtime.stop()


async def test_single_contribution_bypasses_llm() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(llm)
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="The timer is done.", timestamp_us=11),
            participant_id="alice",
            source="timer",
        )
        await recorder.wait_for(1)
    finally:
        await _stop(runtime, aggregator)

    output, metadata = recorder.outputs[0]
    assert output == VoiceOutput(text="The timer is done.", timestamp_us=11)
    assert metadata.participant_id == "alice"
    assert metadata.source == "voice-aggregation"
    assert llm.calls == []


async def test_simultaneous_contributions_are_rewritten_once() -> None:
    llm = _LLM("Temperature rose to 24 degrees, and the timer is done.")
    runtime, aggregator, recorder = await _start(llm)
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Temperature rose to 24 degrees.", timestamp_us=20),
            participant_id="alice",
            source="instrument-monitor",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="The timer is done.", interrupt=True, timestamp_us=10),
            participant_id="alice",
            source="timer",
        )
        await recorder.wait_for(1)
    finally:
        await _stop(runtime, aggregator)

    assert recorder.outputs[0][0] == VoiceOutput(
        text="Temperature rose to 24 degrees, and the timer is done.",
        interrupt=True,
        timestamp_us=10,
    )
    assert len(llm.calls) == 1
    messages, kwargs = llm.calls[0]
    assert "instrument-monitor" in str(messages[-1].content)
    assert "timer" in str(messages[-1].content)
    assert kwargs == {
        "max_tokens": 192,
        "temperature": 0.0,
        "enable_thinking": False,
    }


async def test_participants_aggregate_independently() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(llm)
    try:
        await asyncio.gather(
            runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(text="Alice update."),
                participant_id="alice",
                source="monitor",
            ),
            runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(text="Bob update."),
                participant_id="bob",
                source="monitor",
            ),
        )
        await recorder.wait_for(2)
    finally:
        await _stop(runtime, aggregator)

    assert {
        (metadata.participant_id, output.text)
        for output, metadata in recorder.outputs
    } == {("alice", "Alice update."), ("bob", "Bob update.")}
    assert llm.calls == []


async def test_lone_stream_passes_through_while_pending_updates_coalesce() -> None:
    llm = _LLM("Temperature changed, and the timer completed.")
    runtime, aggregator, recorder = await _start(llm)
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="I can see ", response_id="view", final=False),
            participant_id="alice",
            source="foreground",
        )
        await recorder.wait_for(1)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Temperature changed."),
            participant_id="alice",
            source="instrument-monitor",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="The timer completed."),
            participant_id="alice",
            source="timer",
        )
        assert len(recorder.outputs) == 1
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="a beaker.", response_id="view"),
            participant_id="alice",
            source="foreground",
        )
        await recorder.wait_for(3)
    finally:
        await _stop(runtime, aggregator)

    first, second, third = [output for output, _metadata in recorder.outputs]
    assert first.text == "I can see "
    assert first.final is False
    assert first.response_id
    assert first.response_id != "view"
    assert second == VoiceOutput(text="a beaker.", response_id=first.response_id)
    assert third == VoiceOutput(text="Temperature changed, and the timer completed.")
    assert len(llm.calls) == 1


async def test_urgent_contribution_interrupts_active_stream() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(llm)
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="A long description", response_id="view", final=False),
            participant_id="alice",
            source="foreground",
        )
        await recorder.wait_for(1)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Move away from the heater.", interrupt=True),
            participant_id="alice",
            source="safety-monitor",
        )
        await recorder.wait_for(2)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(response_id="view"),
            participant_id="alice",
            source="foreground",
        )
        await asyncio.sleep(0)
    finally:
        await _stop(runtime, aggregator)

    assert [output.text for output, _metadata in recorder.outputs] == [
        "A long description",
        "Move away from the heater.",
    ]
    assert recorder.outputs[-1][0].interrupt is True
    assert llm.calls == []


async def test_rewrite_failure_preserves_all_updates() -> None:
    llm = _LLM(error=RuntimeError("model unavailable"))
    runtime, aggregator, recorder = await _start(llm)
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="First update."),
            participant_id="alice",
            source="one",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Second update."),
            participant_id="alice",
            source="two",
        )
        await recorder.wait_for(1)
    finally:
        await _stop(runtime, aggregator)

    assert recorder.outputs[0][0].text == "First update. Second update."


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": " "}, "prompt"),
        ({"coalesce_window_s": -1}, "window"),
        ({"queue_capacity": 0}, "capacity"),
        ({"max_batch_size": 1}, "batch size"),
        ({"max_tokens": 0}, "token limit"),
        ({"stream_idle_timeout_s": 0}, "stream idle timeout"),
        ({"participant_idle_timeout_s": 0}, "participant idle timeout"),
    ],
)
def test_configuration_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VoiceAggregationAgent(llm=_LLM(), **kwargs)  # type: ignore[arg-type]
