# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for participant-scoped voice output aggregation."""

from __future__ import annotations

import asyncio

import pytest
from loguru import logger
from xr_ai_models import ChatMessage, ChatResponse
from xr_ai_runtime import Agent, AgentRuntime, MessageMetadata, RuntimeContext, subscribe
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    VOICE_OUTPUT_TOPIC,
    VoiceAggregationAgent,
    VoiceOutput,
)


class _LLM:
    def __init__(
        self,
        content: str = "Combined update.",
        *,
        error: Exception | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.gate = gate
        self.started = asyncio.Event()
        self.cancelled = False
        self.calls: list[tuple[tuple[ChatMessage, ...], dict]] = []

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.calls.append((tuple(messages), kwargs))
        self.started.set()
        if self.error is not None:
            raise self.error
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return ChatResponse(self.content, None, None, "stop", {})


class _Recorder(Agent):
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        super().__init__()
        self.outputs: list[tuple[VoiceOutput, MessageMetadata]] = []
        self.changed = asyncio.Condition()
        self.gate = gate

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def record(self, output: VoiceOutput, ctx: RuntimeContext) -> None:
        async with self.changed:
            self.outputs.append((output, ctx.metadata))
            self.changed.notify_all()
        if self.gate is not None:
            await self.gate.wait()

    async def wait_for(self, count: int) -> None:
        async with self.changed:
            await asyncio.wait_for(
                self.changed.wait_for(lambda: len(self.outputs) >= count),
                1.0,
            )


async def _start(
    llm: _LLM,
    *,
    output_gate: asyncio.Event | None = None,
    **kwargs,
) -> tuple[AgentRuntime, VoiceAggregationAgent, _Recorder]:
    kwargs.setdefault("speech_rate_wpm", 60_000.0)
    kwargs.setdefault("minimum_playback_s", 0.0)
    kwargs.setdefault("coalesce_window_s", 0.01)
    runtime = AgentRuntime()
    aggregator = runtime.register(
        "voice-aggregation",
        VoiceAggregationAgent(
            llm=llm,  # type: ignore[arg-type]
            participant_idle_timeout_s=1.0,
            **kwargs,
        ),
    )
    recorder = runtime.register("recorder", _Recorder(output_gate))
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
    assert output.text == "The timer is done."
    assert output.timestamp_us == 11
    assert output.final is True
    assert output.response_id
    assert metadata.participant_id == "alice"
    assert metadata.source == "voice-aggregation"
    assert llm.calls == []


async def test_simultaneous_contributions_are_rewritten_once() -> None:
    llm = _LLM("Temperature rose to 24 degrees, and the timer is done.")
    runtime, aggregator, recorder = await _start(llm, minimum_playback_s=0.2)
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Temperature rose to 24 degrees.", timestamp_us=20),
            participant_id="alice",
            source="instrument-monitor",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="The timer is done.", timestamp_us=10),
            participant_id="alice",
            source="timer",
        )
        await recorder.wait_for(1)
        assert aggregator._states["alice"].task is not None
        assert not aggregator._states["alice"].task.done()
    finally:
        await _stop(runtime, aggregator)

    output = recorder.outputs[0][0]
    assert output.text == "Temperature rose to 24 degrees, and the timer is done."
    assert output.interrupt is False
    assert output.timestamp_us == 10
    assert output.final is True
    assert output.response_id
    assert len(llm.calls) == 1
    messages, kwargs = llm.calls[0]
    assert "instrument-monitor" in str(messages[-1].content)
    assert "timer" in str(messages[-1].content)
    assert kwargs == {
        "max_tokens": 192,
        "temperature": 0.0,
        "enable_thinking": False,
        "timeout": 5.0,
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

    assert {(metadata.participant_id, output.text) for output, metadata in recorder.outputs} == {
        ("alice", "Alice update."),
        ("bob", "Bob update."),
    }
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
    assert second == VoiceOutput(
        text="a beaker.",
        response_id=first.response_id,
        final=True,
    )
    assert third.text == "Temperature changed, and the timer completed."
    assert third.final is True
    assert third.response_id
    assert len(llm.calls) == 1


async def test_stream_buffered_during_playback_keeps_all_chunks() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        minimum_playback_s=0.08,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Initial announcement."),
            participant_id="alice",
            source="foreground",
        )
        await recorder.wait_for(1)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Buffered ", response_id="view", final=False),
            participant_id="alice",
            source="foreground",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="stream.", response_id="view"),
            participant_id="alice",
            source="foreground",
        )
        await recorder.wait_for(3)
    finally:
        await _stop(runtime, aggregator)

    outputs = [output for output, _metadata in recorder.outputs]
    assert outputs[0].final is True
    assert outputs[1].text == "Buffered "
    assert outputs[2] == VoiceOutput(
        text="stream.",
        response_id=outputs[1].response_id,
        final=True,
    )


async def test_interleaved_streams_keep_each_stream_ordered() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(llm)
    try:
        for source, response_id, text, final in (
            ("one", "first", "First ", False),
            ("two", "second", "Second ", False),
            ("one", "first", "stream.", True),
            ("two", "second", "stream.", True),
        ):
            await runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(text=text, response_id=response_id, final=final),
                participant_id="alice",
                source=source,
            )
        await recorder.wait_for(4)
    finally:
        await _stop(runtime, aggregator)

    outputs = [output for output, _metadata in recorder.outputs]
    first_id = outputs[0].response_id
    second_id = outputs[2].response_id
    assert [output.text for output in outputs] == [
        "First ",
        "stream.",
        "Second ",
        "stream.",
    ]
    assert [output.response_id for output in outputs] == [
        first_id,
        first_id,
        second_id,
        second_id,
    ]
    assert [output.final for output in outputs] == [False, True, False, True]


async def test_updates_during_estimated_playback_coalesce_after_current_output() -> None:
    llm = _LLM("Temperature changed, and the timer completed.")
    runtime, aggregator, recorder = await _start(
        llm,
        speech_rate_wpm=6_000.0,
        minimum_playback_s=0.1,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="This response remains active while it is spoken."),
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
        await recorder.wait_for(2)
    finally:
        await _stop(runtime, aggregator)

    first, combined = [output for output, _metadata in recorder.outputs]
    assert first.text == "This response remains active while it is spoken."
    assert first.final is True
    assert combined.text == "Temperature changed, and the timer completed."
    assert combined.final is True
    assert len(llm.calls) == 1


async def test_pending_capacity_drops_oldest_across_all_buffered_work() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        queue_capacity=2,
        minimum_playback_s=0.2,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Current response."),
            participant_id="alice",
            source="foreground",
        )
        await recorder.wait_for(1)
        for index in range(5):
            await runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(text=f"Update {index}."),
                participant_id="alice",
                source="monitor",
            )
        state = aggregator._states["alice"]
        assert len(state.pending) == 2
        assert [item.output.text for item in state.pending] == [
            "Update 3.",
            "Update 4.",
        ]
    finally:
        await _stop(runtime, aggregator)


async def test_pending_capacity_preserves_urgent_work() -> None:
    gate = asyncio.Event()
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        output_gate=gate,
        queue_capacity=2,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Blocked output."),
            participant_id="alice",
            source="foreground",
        )
        await recorder.wait_for(1)
        for text in ("Urgent one.", "Urgent two."):
            await runtime.publish(
                VOICE_CONTRIBUTION_TOPIC,
                VoiceOutput(text=text, interrupt=True),
                participant_id="alice",
                source="safety",
            )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Routine stream.", response_id="routine", final=False),
            participant_id="alice",
            source="monitor",
        )

        state = aggregator._states["alice"]
        assert [item.output.text for item in state.pending] == [
            "Urgent one.",
            "Urgent two.",
        ]
        assert ("monitor", "routine") in state.discarded_streams

        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Urgent three.", interrupt=True),
            participant_id="alice",
            source="safety",
        )
        assert [item.output.text for item in state.pending] == [
            "Urgent two.",
            "Urgent three.",
        ]
    finally:
        gate.set()
        await _stop(runtime, aggregator)


async def test_discarded_stream_does_not_resume_after_other_interrupts() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        queue_capacity=1,
        stream_idle_timeout_s=0.2,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="A-start", response_id="A", final=False),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(1)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(
                text="B-start",
                response_id="B",
                final=False,
                interrupt=True,
            ),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(2)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(
                text="C-start",
                response_id="C",
                final=False,
                interrupt=True,
            ),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(3)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="C-end", response_id="C"),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(4)

        state = aggregator._states["alice"]
        a_key = ("stream", "A")
        old_expiry = state.discarded_streams[a_key]
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="A-stale-tail", response_id="A", final=False),
            participant_id="alice",
            source="stream",
        )
        await asyncio.sleep(0.02)
        assert len(recorder.outputs) == 4
        assert state.discarded_streams[a_key] > old_expiry

        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(response_id="A"),
            participant_id="alice",
            source="stream",
        )
        for _ in range(100):
            if a_key not in state.discarded_streams:
                break
            await asyncio.sleep(0.001)
        assert a_key not in state.discarded_streams
    finally:
        await _stop(runtime, aggregator)

    assert [output.text for output, _metadata in recorder.outputs] == [
        "A-start",
        "B-start",
        "C-start",
        "C-end",
    ]
    assert recorder.outputs[-1][0].final is True


async def test_discarded_stream_can_restart_after_idle_expiry() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        queue_capacity=1,
        stream_idle_timeout_s=0.03,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="A-old", response_id="A", final=False),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(1)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(
                text="B-start",
                response_id="B",
                final=False,
                interrupt=True,
            ),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(2)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="B-end", response_id="B"),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(3)
        await asyncio.sleep(0.04)

        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="A-new", response_id="A", final=False),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(4)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="A-end", response_id="A"),
            participant_id="alice",
            source="stream",
        )
        await recorder.wait_for(5)
    finally:
        await _stop(runtime, aggregator)

    assert [output.text for output, _metadata in recorder.outputs] == [
        "A-old",
        "B-start",
        "B-end",
        "A-new",
        "A-end",
    ]
    assert recorder.outputs[2][0].final is True
    assert recorder.outputs[4][0].final is True


async def test_expired_discarded_streams_are_pruned_by_finite_work() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        stream_idle_timeout_s=0.02,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Current status."),
            participant_id="alice",
            source="monitor",
        )
        await recorder.wait_for(1)
        state = aggregator._states["alice"]
        state.discarded_streams[("stream", "stale")] = asyncio.get_running_loop().time() + 0.01
        await asyncio.sleep(0.02)

        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Updated status."),
            participant_id="alice",
            source="monitor",
        )
        await recorder.wait_for(2)

        assert state.discarded_streams == {}
    finally:
        await _stop(runtime, aggregator)


async def test_release_cancels_only_departed_participant_state() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        speech_rate_wpm=1.0,
        minimum_playback_s=0.0,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Long response for Alice."),
            participant_id="alice",
            source="foreground",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Long response for Bob."),
            participant_id="bob",
            source="foreground",
        )
        await recorder.wait_for(2)

        await aggregator.release("alice")

        assert "alice" not in aggregator._states
        assert "bob" in aggregator._states
    finally:
        await _stop(runtime, aggregator)


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


async def test_urgent_batch_speaks_alert_first_and_retains_routine_order() -> None:
    llm = _LLM("Routine one, then routine two.")
    runtime, aggregator, recorder = await _start(llm)
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Routine one."),
            participant_id="alice",
            source="monitor",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Routine two."),
            participant_id="alice",
            source="timer",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Move away.", interrupt=True),
            participant_id="alice",
            source="safety",
        )
        await recorder.wait_for(2)
    finally:
        await _stop(runtime, aggregator)

    assert [output.text for output, _metadata in recorder.outputs] == [
        "Move away.",
        "Routine one, then routine two.",
    ]
    assert recorder.outputs[0][0].interrupt is True
    assert all(output.final for output, _metadata in recorder.outputs)
    assert len(llm.calls) == 1
    assert "Routine one." in str(llm.calls[0][0][-1].content)
    assert "Routine two." in str(llm.calls[0][0][-1].content)
    assert str(llm.calls[0][0][-1].content).index("Routine one.") < str(llm.calls[0][0][-1].content).index(
        "Routine two."
    )


async def test_stream_boundary_does_not_hide_queued_urgent_output() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        coalesce_window_s=0.05,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Routine."),
            participant_id="alice",
            source="monitor",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Stream start", response_id="foreground", final=False),
            participant_id="alice",
            source="foreground",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Move now.", interrupt=True),
            participant_id="alice",
            source="safety",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Stream end", response_id="foreground"),
            participant_id="alice",
            source="foreground",
        )
        await recorder.wait_for(4)
    finally:
        await _stop(runtime, aggregator)

    assert [output.text for output, _metadata in recorder.outputs] == [
        "Move now.",
        "Routine.",
        "Stream start",
        "Stream end",
    ]
    assert recorder.outputs[0][0].interrupt is True
    assert recorder.outputs[-1][0].final is True
    assert llm.calls == []


async def test_urgent_first_contribution_skips_coalescing_delay() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(
        llm,
        coalesce_window_s=0.5,
    )
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Move away.", interrupt=True),
            participant_id="alice",
            source="safety",
        )
        await asyncio.wait_for(recorder.wait_for(1), timeout=0.2)
    finally:
        await _stop(runtime, aggregator)

    assert recorder.outputs[0][0].text == "Move away."
    assert recorder.outputs[0][0].interrupt is True


async def test_urgent_contribution_cancels_in_flight_rewrite() -> None:
    gate = asyncio.Event()
    llm = _LLM(gate=gate)
    runtime, aggregator, recorder = await _start(llm)
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="First routine update."),
            participant_id="alice",
            source="one",
        )
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Second routine update."),
            participant_id="alice",
            source="two",
        )
        await asyncio.wait_for(llm.started.wait(), 1.0)
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Move away now.", interrupt=True),
            participant_id="alice",
            source="safety",
        )
        await recorder.wait_for(1)
        gate.set()
        await recorder.wait_for(2)
    finally:
        await _stop(runtime, aggregator)

    assert llm.cancelled is True
    assert recorder.outputs[0][0].text == "Move away now."
    assert recorder.outputs[0][0].interrupt is True
    assert recorder.outputs[1][0].text == "Combined update."
    assert len(llm.calls) == 2
    assert "First routine update." in str(llm.calls[1][0][-1].content)
    assert "Second routine update." in str(llm.calls[1][0][-1].content)


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


async def test_rewrite_timeout_is_enforced_around_service_call() -> None:
    gate = asyncio.Event()
    llm = _LLM(gate=gate)
    runtime, aggregator, recorder = await _start(
        llm,
        rewrite_timeout_s=0.02,
    )
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

    assert llm.cancelled is True
    assert recorder.outputs[0][0].text == "First update. Second update."
    assert llm.calls[0][1]["timeout"] == 0.02


@pytest.mark.parametrize("release", [False, True])
async def test_shutdown_logs_prepublication_in_flight_work(release: bool) -> None:
    gate = asyncio.Event()
    llm = _LLM(gate=gate)
    runtime, aggregator, _recorder = await _start(llm)
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
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
        await asyncio.wait_for(llm.started.wait(), 1.0)

        if release:
            await aggregator.release("alice")
        else:
            await aggregator.stop()
    finally:
        logger.remove(handler_id)
        await aggregator.stop()
        await runtime.stop()

    reason = "participant release" if release else "shutdown"
    assert any(
        "discarded accepted voice contributions" in message and "count=2" in message and f"reason={reason}" in message
        for message in messages
    )
    assert llm.cancelled is True


async def test_contribution_after_stop_is_dropped_without_publisher_error() -> None:
    llm = _LLM()
    runtime, aggregator, recorder = await _start(llm)
    await aggregator.stop()
    try:
        await runtime.publish(
            VOICE_CONTRIBUTION_TOPIC,
            VoiceOutput(text="Late update."),
            participant_id="alice",
            source="monitor",
        )
        await asyncio.sleep(0)
    finally:
        await runtime.stop()

    assert recorder.outputs == []
    assert aggregator._states == {}


def test_playback_duration_uses_configured_ceiling() -> None:
    aggregator = VoiceAggregationAgent(
        llm=_LLM(),  # type: ignore[arg-type]
        speech_rate_wpm=1.0,
        minimum_playback_s=0.0,
        maximum_playback_s=2.5,
    )

    assert aggregator._playback_duration("A very long response") == 2.5


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
        ({"speech_rate_wpm": 0}, "speech rate"),
        ({"minimum_playback_s": -1}, "minimum playback"),
        ({"maximum_playback_s": 0}, "maximum playback"),
        (
            {"minimum_playback_s": 2, "maximum_playback_s": 1},
            "minimum playback",
        ),
        ({"rewrite_timeout_s": 0}, "rewrite timeout"),
    ],
)
def test_configuration_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VoiceAggregationAgent(llm=_LLM(), **kwargs)  # type: ignore[arg-type]
