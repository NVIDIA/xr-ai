# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatResponse, ChatStreamEvent, DecisionAnswer, DecisionResponse, ToolCall
from xr_ai_tools.current_frame import CurrentFrameRequest, ImageFrame
from xr_ai_tools.image import ImageReference
from xr_ai_tools.vision import ImageQueryChunk, ImageQueryRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-sdk"))

import xr_ai_conversation.front_end as front_end_module  # noqa: E402
from xr_ai_conversation import (  # noqa: E402
    Admission,
    Conversation,
    ConversationContextUpdate,
    ConversationExchange,
    ConversationFrontEnd,
    ConversationTurn,
    DestinationResult,
    RoutingAid,
)
from xr_ai_conversation._context import _ConversationContext  # noqa: E402
from xr_ai_conversation.conversation import _messages  # noqa: E402
from xr_ai_conversation.front_end import _is_valid_acknowledgment  # noqa: E402
from xr_ai_runtime import AgentRuntime, Topic  # noqa: E402


class _Prediction:
    def __init__(self, choice="uncertain", probabilities=None, *, error=None, blocked=False) -> None:
        self.choice = choice
        self.probabilities = probabilities or {"slow": 0.0, "within_delay": 0.0, "uncertain": 1.0}
        self.error = error
        self.blocked = blocked
        self.calls = []
        self.called = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def decide(self, state, questions, **_kwargs):
        self.calls.append((state, questions))
        self.called.set()
        try:
            if self.blocked:
                await asyncio.Event().wait()
            if self.error is not None:
                raise self.error
            return DecisionResponse(
                answers={
                    "first_answer_timing": DecisionAnswer(
                        self.choice,
                        self.probabilities.get(self.choice, 0.0),
                        self.probabilities,
                    )
                },
                model="fake",
                usage={},
                raw={},
            )
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _AcknowledgmentLLM:
    def __init__(self, content="Let me think through that.", *, blocked=False, error=None) -> None:
        self.content = content
        self.blocked = blocked
        self.error = error
        self.calls = []
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        self.started.set()
        try:
            if self.blocked:
                await self.release.wait()
            if self.error is not None:
                raise self.error
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return ChatResponse(self.content, None, None, "stop", {})


class _AckDestination:
    def __init__(self, *, preamble="", blocked=False, answer="Here is the answer.") -> None:
        self.preamble = preamble
        self.blocked = blocked
        self.answer_text = answer
        self.answer_started = asyncio.Event()
        self.answer_emit_started = asyncio.Event()
        self.release_answer = asyncio.Event()
        self.calls = []

    def routing_aid(self, _participant_id):
        return RoutingAid("destination", "Answers requests about the selected topic.")

    async def admit(self, _turn, *, focused):
        return Admission("accept", preamble=self.preamble)

    async def answer(self, turn, *, emit):
        self.calls.append(turn.query)
        self.answer_started.set()
        if self.blocked:
            await self.release_answer.wait()
        if self.answer_text:
            self.answer_emit_started.set()
            await emit("destination", self.answer_text)
        return DestinationResult("answer", self.answer_text)

    async def release(self, _participant_id):
        pass


async def _ack_output(front_end, query, output, *, participant="alice"):
    async def emit(source, text):
        output.append((source, text))

    return await front_end.handle(
        ConversationTurn(participant, f"turn-{participant}-{query}", query),
        emit=emit,
    )


class _Decisions:
    async def decide(self, *_args, **_kwargs):
        raise AssertionError("one registered destination does not need route inference")


class _LLM:
    def __init__(self) -> None:
        self.messages = []
        self.calls = []

    def stream_events(self, messages, **kwargs):
        self.messages.append(messages)
        self.calls.append(kwargs)

        async def events():
            yield ChatStreamEvent(kind="reasoning", reasoning="internal planning must not be spoken")
            yield ChatStreamEvent(kind="text", text="Hello there.")
            yield ChatStreamEvent(kind="finish", finish_reason="stop")

        return events()

    async def chat(self, messages, **_kwargs):
        return ChatResponse(
            content="Compacted source facts.",
            reasoning=None,
            tool_calls=None,
            finish_reason="stop",
            raw={},
        )


class _CurrentViewLLM:
    def __init__(self, arguments='{"query":"What color is the sign?"}'):
        self.arguments = arguments
        self.messages = []
        self.tools = []
        self.closed = asyncio.Event()

    def stream_events(self, messages, *, tools=(), **_kwargs):
        self.messages.append(messages)
        self.tools.append(tools)

        async def events():
            try:
                yield ChatStreamEvent(
                    kind="tool_call",
                    tool_call=ToolCall("call-1", "current_view", self.arguments),
                )
                yield ChatStreamEvent(kind="finish", finish_reason="tool_calls")
            finally:
                self.closed.set()

        return events()


class _CurrentFrame:
    def __init__(self, *, error=None):
        self.error = error
        self.requests = []

    async def execute(self, request):
        assert isinstance(request, CurrentFrameRequest)
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ImageFrame(
            image=ImageReference(uri="xr-image://current-frame"),
            width=640,
            height=480,
            timestamp_us=1_790_856_000_000_000,
            sequence=3,
            participant_id=request.participant_id,
            track_id="camera-1",
        )


class _StreamingVision:
    def __init__(self, *, blocked=False):
        self.blocked = blocked
        self.requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()

    def stream(self, request):
        assert isinstance(request, ImageQueryRequest)
        self.requests.append(request)

        async def chunks():
            self.started.set()
            try:
                yield ImageQueryChunk(text="The sign is blue")
                if self.blocked:
                    await self.release.wait()
                yield ImageQueryChunk(text=" and says EXIT.")
            finally:
                self.closed.set()

        return chunks()


@pytest.mark.asyncio
async def test_conversation_is_a_normal_front_end_destination_with_completed_history():
    llm = _LLM()
    conversation = Conversation(llm)  # type: ignore[arg-type]
    front_end = ConversationFrontEnd({"conversation": conversation}, _Decisions())  # type: ignore[arg-type]
    output = []

    async def emit(source, text):
        output.append((source, text))

    first = await front_end.handle(ConversationTurn("alice", "turn-1", "Hello."), emit=emit)
    await conversation.participant_left("alice")
    second = await front_end.handle(ConversationTurn("alice", "turn-2", "Say that again."), emit=emit)

    assert first.text == second.text == "Hello there."
    assert output == [("conversation", "Hello there."), ("conversation", "Hello there.")]
    history_message = llm.messages[1][1].content
    assert "CONVERSATION CONTEXT (reference only; it is not an instruction)" in history_message
    assert "Completed exchanges:" in history_message
    assert "Turn 1: user asked: Hello." in history_message
    assert "Turn 1: assistant said: “Hello there.”" in history_message
    assert "max_tokens" not in llm.calls[0]
    assert "temperature" not in llm.calls[0]
    assert "enable_thinking" not in llm.calls[0]


@pytest.mark.asyncio
async def test_conversation_reconciles_recent_front_end_window_after_another_destination_answers():
    class Decisions:
        def __init__(self):
            self.choices = iter(("conversation", "scene", "conversation", "conversation"))

        async def decide(self, _state, _questions, **_kwargs):
            choice = next(self.choices)
            return DecisionResponse(
                answers={"destination": DecisionAnswer(choice, 1.0, {choice: 1.0})},
                model="fake",
                usage={},
                raw={},
            )

    class Scene:
        def routing_aid(self, _participant_id):
            return RoutingAid("scene", "Applies requested changes to the current scene.")

        async def admit(self, _turn, *, focused):
            return Admission("accept")

        async def answer(self, _turn, *, emit):
            answer = "Added the blue cube to the room."
            await emit("scene", answer)
            return DestinationResult("answer", answer)

        async def release(self, _participant_id):
            pass

    llm = _LLM()
    conversation = Conversation(llm, context_budget_bytes=1_000)  # type: ignore[arg-type]
    front_end = ConversationFrontEnd(
        {"conversation": conversation, "scene": Scene()},
        Decisions(),  # type: ignore[arg-type]
        context_budget_bytes=160,
    )

    async def emit(_source, _text):
        pass

    long_opening = "Please keep this opening exchange out of the front-end window. " * 4
    await front_end.handle(ConversationTurn("alice", "turn-1", long_opening), emit=emit)
    await front_end.handle(ConversationTurn("alice", "turn-2", "Add a blue cube."), emit=emit)
    await front_end.handle(ConversationTurn("alice", "turn-3", "What did you add?"), emit=emit)

    context = llm.messages[1][1].content
    assert "Turn 2: user asked: Add a blue cube." in context
    assert "Turn 2: assistant said: “Added the blue cube to the room.”" in context

    await front_end.handle(ConversationTurn("alice", "turn-4", "What color was it?"), emit=emit)
    repeated_context = llm.messages[2][1].content
    assert repeated_context.count("Added the blue cube to the room.") == 1
    assert "Turn 2: user asked: Add a blue cube." in repeated_context


@pytest.mark.asyncio
async def test_optional_current_view_tool_uses_typed_frame_and_streaming_vision_requests():
    llm = _CurrentViewLLM()
    frames = _CurrentFrame()
    vision = _StreamingVision()
    conversation = Conversation(llm, frames=frames, vision=vision)  # type: ignore[arg-type]
    front_end = ConversationFrontEnd({"conversation": conversation}, _Decisions())  # type: ignore[arg-type]
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await front_end.handle(ConversationTurn("alice", "view-1", "What color is the sign?"), emit=emit)

    assert result.text == "The sign is blue and says EXIT."
    assert output == [
        ("conversation", "The sign is blue"),
        ("conversation", " and says EXIT."),
    ]
    assert llm.tools[0][0].name == "current_view"
    assert frames.requests == [CurrentFrameRequest(participant_id="alice")]
    assert vision.requests == [
        ImageQueryRequest(image=ImageReference(uri="xr-image://current-frame"), query="What color is the sign?")
    ]
    assert llm.closed.is_set()
    assert vision.closed.is_set()


@pytest.mark.asyncio
async def test_current_view_rejects_invalid_tool_arguments_and_reports_frame_unavailability():
    invalid_llm = _CurrentViewLLM(arguments="not-json")
    invalid_frames = _CurrentFrame()
    invalid_conversation = Conversation(
        invalid_llm,
        frames=invalid_frames,
        vision=_StreamingVision(),
    )  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="invalid current_view arguments"):
        _ = [
            chunk
            async for chunk in invalid_conversation.stream("Look here.", "alice")
        ]
    assert invalid_frames.requests == []
    assert invalid_llm.closed.is_set()

    unavailable_llm = _CurrentViewLLM()
    unavailable_frames = _CurrentFrame(error=FrameUnavailable("No camera frame is available."))
    unavailable_conversation = Conversation(
        unavailable_llm,
        frames=unavailable_frames,
        vision=_StreamingVision(),
    )  # type: ignore[arg-type]
    output = [
        chunk
        async for chunk in unavailable_conversation.stream("Look here.", "alice")
    ]

    assert output == ["No camera frame is available."]
    assert len(unavailable_frames.requests) == 1
    assert unavailable_llm.closed.is_set()


@pytest.mark.asyncio
async def test_current_view_rejects_tool_calls_when_visual_tools_are_not_configured():
    llm = _CurrentViewLLM()
    conversation = Conversation(llm)

    with pytest.raises(RuntimeError, match="without a configured tool"):
        _ = [chunk async for chunk in conversation.stream("Look here.", "alice")]
    assert llm.closed.is_set()


@pytest.mark.asyncio
async def test_current_view_cancellation_closes_model_and_visual_streams():
    llm = _CurrentViewLLM()
    vision = _StreamingVision(blocked=True)
    conversation = Conversation(llm, frames=_CurrentFrame(), vision=vision)  # type: ignore[arg-type]
    front_end = ConversationFrontEnd({"conversation": conversation}, _Decisions())  # type: ignore[arg-type]

    async def emit(_source, _text):
        pass

    task = asyncio.create_task(
        front_end.handle(ConversationTurn("alice", "view-cancel", "Look here."), emit=emit)
    )
    await asyncio.wait_for(vision.started.wait(), timeout=1)
    await front_end.interrupt("alice")
    with pytest.raises(asyncio.CancelledError):
        await task

    assert llm.closed.is_set()
    assert vision.closed.is_set()


def test_message_history_keeps_full_exchanges_and_latest_request_separate():
    history = tuple(ConversationExchange(f"Question {index}", f"Answer {index}") for index in range(6))
    messages = _messages("Current question", history, instructions="", vision_available=True)

    assert [message.role for message in messages] == ["system", "user", "user"]
    assert messages[-1].content == "Current question"
    context = messages[-2].content
    assert "Turn 1: user asked: Question 0" in context
    assert "Turn 6: user asked: Question 5" in context
    assert "Turn 1: assistant said: “Answer 0”" in context
    assert "Turn 6: assistant said: “Answer 5”" in context
    assert "Exchange completion time: unknown" in context


def test_message_history_renders_coarse_known_recency_and_source_ranges():
    messages = _messages(
        "Current question",
        (ConversationExchange("Earlier question", "Earlier answer"),),
        instructions="",
        vision_available=True,
        history_timestamps=(1_790_856_000_000_000,),
        background=(
            (
                "tea.status",
                "Timer is running.",
                "from around 2026-10-01 12:00 UTC through around 2026-10-01 13:00 UTC",
            ),
        ),
    )
    context = messages[-2].content
    assert "Exchange completion time: 2026-10-01 12:00 UTC" in context
    assert (
        "Source tea.status (publication time from around 2026-10-01 12:00 UTC through around 2026-10-01 13:00 UTC)"
        in context
    )


def test_message_history_drops_whole_old_context_to_byte_budget_without_truncating_query():
    history = tuple(
        ConversationExchange(f"Question {index}: " + "q" * 160, f"Answer {index}: " + "a" * 160) for index in range(6)
    )
    query = "Current question " + "z" * 300
    messages = _messages(
        query,
        history,
        instructions="keep the application instruction",
        vision_available=True,
        background=(("tea", "summary " + "t" * 100),),
        context_budget_bytes=650,
    )
    context = messages[-2].content

    assert messages[-1].content == query
    assert "keep the application instruction" in messages[0].content
    assert len(context.encode("utf-8")) <= 650
    assert "Question 5" in context
    assert "Question 0" not in context
    assert "summary" not in context or "Source tea" in context


@pytest.mark.asyncio
async def test_acknowledgment_is_disabled_by_default_and_fast_answer_cancels_prediction():
    destination = _AckDestination()
    decisions = _Prediction()
    llm = _AcknowledgmentLLM()
    front_end = ConversationFrontEnd({"destination": destination}, decisions, default_destination="destination")  # type: ignore[arg-type]
    output = []

    await _ack_output(front_end, "Quick request.", output)

    assert output == [("destination", "Here is the answer.")]
    assert not decisions.calls
    assert not llm.calls

    enabled = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=20.0,
    )
    await _ack_output(enabled, "Another quick request.", output)
    assert output[-1] == ("destination", "Here is the answer.")
    assert not decisions.calls
    assert not llm.calls


@pytest.mark.asyncio
async def test_confident_slow_prediction_acknowledges_then_records_only_real_answer():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction(
        "slow",
        {"slow": 0.8, "within_delay": 0.1, "uncertain": 0.1},
    )
    llm = _AcknowledgmentLLM()
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    emitted = asyncio.Event()

    async def emit(source, text):
        output.append((source, text))
        emitted.set()

    task = asyncio.create_task(front_end.handle(ConversationTurn("alice", "turn-1", "Explain the request."), emit=emit))
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(decisions.called.wait(), timeout=1)
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    await asyncio.wait_for(emitted.wait(), timeout=1)
    destination.release_answer.set()
    result = await asyncio.wait_for(task, timeout=1)

    assert result.disposition == "answer"
    assert output == [
        ("destination", "Let me think through that. "),
        ("destination", "Here is the answer."),
    ]
    assert llm.calls[0][1] == {"max_tokens": 64, "enable_thinking": False}
    acknowledgment_messages = llm.calls[0][0]
    assert acknowledgment_messages[0].role == "system"
    assert "Let me think through [short neutral restatement" in acknowledgment_messages[0].content
    assert "Do not include findings, advice, recommendations" in acknowledgment_messages[0].content
    assert "Current user request: Explain the request." in acknowledgment_messages[1].content
    assert acknowledgment_messages[1].content.endswith("Explain the request.")
    assert "prior_first_answer_examples" in decisions.calls[0][0]
    assert front_end._history["alice"][-1].assistant == "Here is the answer."
    assert front_end._history["alice"][-1].assistant != output[0][1]
    assert front_end._first_answer_samples["alice"]["destination"][0][0] == "Explain the request."


@pytest.mark.asyncio
async def test_uncertain_prediction_uses_delay_fallback_and_admission_preamble_suppresses_ack():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction()
    llm = _AcknowledgmentLLM()
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    emitted = asyncio.Event()

    async def emit(source, text):
        output.append((source, text))
        emitted.set()

    task = asyncio.create_task(
        front_end.handle(ConversationTurn("alice", "turn-1", "An unfamiliar request."), emit=emit)
    )
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(emitted.wait(), timeout=1)
    assert output == [("destination", "Let me think through that. ")]
    destination.release_answer.set()
    await asyncio.wait_for(task, timeout=1)

    preamble_destination = _AckDestination(preamble="I will check.")
    preamble_front_end = ConversationFrontEnd(
        {"destination": preamble_destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    previous_calls = len(decisions.calls)
    previous_llm_calls = len(llm.calls)
    await _ack_output(preamble_front_end, "Request with preamble.", output)
    assert len(decisions.calls) == previous_calls
    assert len(llm.calls) == previous_llm_calls
    assert output[-2:] == [
        ("destination", "I will check."),
        ("destination", "Here is the answer."),
    ]


@pytest.mark.asyncio
async def test_real_answer_cancels_pending_ack_generation_without_waiting():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM(blocked=True)
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    task = asyncio.create_task(_ack_output(front_end, "Wait on model output.", output))
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    destination.release_answer.set()
    await asyncio.wait_for(task, timeout=1)

    assert output == [("destination", "Here is the answer.")]
    assert llm.cancelled.is_set()


@pytest.mark.asyncio
async def test_answer_does_not_reuse_interrupt_slot_after_ack_was_delivered():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM()
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    first = True
    acknowledgment_emit_started = asyncio.Event()
    acknowledgment_emit_release = asyncio.Event()

    async def emit(source, text):
        nonlocal first
        interrupt = first
        first = False
        output.append((source, text, interrupt))
        if text.startswith("Let me think through"):
            acknowledgment_emit_started.set()
            await acknowledgment_emit_release.wait()

    task = asyncio.create_task(
        front_end.handle(ConversationTurn("alice", "turn-1", "Wait on model output."), emit=emit)
    )
    await asyncio.wait_for(acknowledgment_emit_started.wait(), timeout=1)
    destination.release_answer.set()
    await asyncio.wait_for(destination.answer_emit_started.wait(), timeout=1)
    acknowledgment_emit_release.set()
    await asyncio.wait_for(task, timeout=1)

    assert output == [
        ("destination", "Let me think through that. ", True),
        ("destination", "Here is the answer.", False),
    ]
    assert llm.calls


@pytest.mark.asyncio
async def test_answer_timing_excludes_committed_acknowledgment_output(monkeypatch):
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM()
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    now = 0.0
    monkeypatch.setattr(front_end_module, "_monotonic", lambda: now)
    acknowledgment_emit_started = asyncio.Event()
    acknowledgment_emit_release = asyncio.Event()

    async def emit(_source, text):
        if text.startswith("Let me think through"):
            acknowledgment_emit_started.set()
            await acknowledgment_emit_release.wait()

    task = asyncio.create_task(
        front_end.handle(ConversationTurn("alice", "turn-1", "Wait on model output."), emit=emit)
    )
    await asyncio.wait_for(acknowledgment_emit_started.wait(), timeout=1)
    now = 1.0
    destination.release_answer.set()
    await asyncio.wait_for(destination.answer_emit_started.wait(), timeout=1)
    await asyncio.sleep(0)
    now = 50.0
    acknowledgment_emit_release.set()
    await asyncio.wait_for(task, timeout=1)

    assert front_end._first_answer_samples["alice"]["destination"] == [("Wait on model output.", 1.0)]


@pytest.mark.parametrize(
    ("text", "valid"),
    [
        ("Let me think through creating a clear walkway.", True),
        ("One practical way is to create a clear central pathway.", False),
        ("I can help you rearrange the room to create a clear walkway.", False),
        ("Let me think through creating a clear walkway. Move the chairs.", False),
    ],
)
def test_acknowledgment_gate_rejects_answer_content(text, valid):
    assert _is_valid_acknowledgment(text) is valid


@pytest.mark.parametrize(
    "generated",
    [
        "Let me think through tell me the next one and then summarize all of them in one paragraph.",
        "Let me think through removing the ceiling light fixture and placing it on the floor safely.",
        "Let me think through moving the desk behind you to a different spot.",
        "Let me think through how safe it is to climb on the desk behind you to remove the lights.",
    ],
)
@pytest.mark.asyncio
async def test_invalid_acknowledgment_uses_neutral_fallback_without_recording_it(generated):
    destination = _AckDestination(blocked=True)
    decisions = _Prediction()
    llm = _AcknowledgmentLLM(content=generated)
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=0.001,
    )
    output = []
    fallback_emitted = asyncio.Event()

    async def emit(source, text):
        output.append((source, text))
        if text == "Let me think that through. ":
            fallback_emitted.set()

    task = asyncio.create_task(
        front_end.handle(ConversationTurn("alice", "turn-1", "Continue with the next request."), emit=emit)
    )
    await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
    await asyncio.wait_for(fallback_emitted.wait(), timeout=1)
    destination.release_answer.set()
    result = await asyncio.wait_for(task, timeout=1)

    assert result.disposition == "answer"
    assert output == [
        ("destination", "Let me think that through. "),
        ("destination", "Here is the answer."),
    ]
    assert front_end._history["alice"][-1].assistant == "Here is the answer."


@pytest.mark.parametrize("disposition", ["focus", "release"])
@pytest.mark.asyncio
async def test_completed_focus_and_release_answers_train_first_answer_timing(disposition):
    class Destination(_AckDestination):
        async def answer(self, _turn, *, emit):
            await emit("destination", self.answer_text)
            return DestinationResult(disposition, self.answer_text)

    destination = Destination()
    front_end = ConversationFrontEnd(
        {"destination": destination},
        _Decisions(),
        default_destination="destination",
    )

    async def emit(_source, _text):
        pass

    result = await front_end.handle(
        ConversationTurn("alice", "turn-1", "Wait on model output."),
        emit=emit,
    )

    assert result.disposition == disposition
    assert front_end._first_answer_samples["alice"]["destination"][0][0] == "Wait on model output."


@pytest.mark.asyncio
async def test_prediction_failure_or_hang_does_not_delay_real_answer():
    for decisions in (_Prediction(error=RuntimeError("prediction failed")), _Prediction(blocked=True)):
        destination = _AckDestination(blocked=True)
        llm = _AcknowledgmentLLM(error=RuntimeError("acknowledgment failed"))
        front_end = ConversationFrontEnd(
            {"destination": destination},
            decisions,  # type: ignore[arg-type]
            default_destination="destination",
            acknowledgment_llm=llm,  # type: ignore[arg-type]
            acknowledgment_delay_secs=0.001,
        )
        output = []
        task = asyncio.create_task(_ack_output(front_end, "Quick answer while prediction runs.", output))
        await asyncio.wait_for(destination.answer_started.wait(), timeout=1)
        await asyncio.wait_for(decisions.called.wait(), timeout=1)
        await asyncio.wait_for(llm.started.wait(), timeout=1)
        destination.release_answer.set()
        await asyncio.wait_for(task, timeout=1)
        assert output == [("destination", "Here is the answer.")]
        if decisions.blocked:
            assert decisions.cancelled.is_set()
        assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_turn_cancellation_cancels_ack_and_departure_drops_timing_examples():
    destination = _AckDestination(blocked=True)
    decisions = _Prediction("slow", {"slow": 0.9, "within_delay": 0.05, "uncertain": 0.05})
    llm = _AcknowledgmentLLM(blocked=True)
    front_end = ConversationFrontEnd(
        {"destination": destination},
        decisions,  # type: ignore[arg-type]
        default_destination="destination",
        acknowledgment_llm=llm,  # type: ignore[arg-type]
        acknowledgment_delay_secs=10.0,
    )
    output = []
    task = asyncio.create_task(_ack_output(front_end, "Cancel this turn.", output))
    await asyncio.wait_for(llm.started.wait(), timeout=1)
    await front_end.interrupt("alice")
    with pytest.raises(asyncio.CancelledError):
        await task

    assert llm.cancelled.is_set()
    assert output == []
    front_end._first_answer_samples["alice"] = {"destination": [("old request", 0.1)]}
    await front_end.participant_left("alice")
    assert "alice" not in front_end._first_answer_samples


@pytest.mark.asyncio
async def test_private_dialogue_compaction_keeps_full_transcript_and_commits_old_prefix():
    context = _ConversationContext()
    context.reconcile_transcript(
        "alice",
        (
            ("The label says copper.", "I will remember copper."),
            ("Keep the facilities check open.", "I will keep it open."),
        ),
    )
    context.reconcile_transcript(
        "alice",
        (
            ("The label says copper.", "I will remember copper."),
            ("Keep the facilities check open.", "I will keep it open."),
        ),
    )
    assert len(context.transcript("alice")) == 2

    async def compact(previous, entries):
        assert previous == ""
        assert [(entry.user, entry.assistant) for entry in entries] == [
            ("The label says copper.", "I will remember copper.")
        ]
        return "The user identified a copper label."

    task = context.schedule_dialogue_compaction(
        "alice",
        trigger_budget_bytes=1,
        payload_budget_bytes=1000,
        summary_budget_bytes=1000,
        compact=compact,
    )
    assert task is not None
    assert await task is True
    summary, recent = context.dialogue_snapshot("alice")
    assert summary == "The user identified a copper label."
    assert [(entry.user, entry.assistant) for entry in recent] == [
        ("Keep the facilities check open.", "I will keep it open.")
    ]
    assert len(context.transcript("alice")) == 2


def test_private_transcript_reconciles_overlapping_windows_and_keeps_timestamp_provenance():
    context = _ConversationContext()
    context.reconcile_transcript(
        "alice",
        (
            ("First request", "First reply", 10),
            ("Scene request", "Scene reply", 20),
        ),
    )

    # The front end may send a recent suffix with one new completed exchange.
    context.reconcile_transcript(
        "alice",
        (
            ("Scene request", "Scene reply", 20),
            ("Latest request", "Latest reply", None),
        ),
    )
    # Replaying the same bounded window must not duplicate either exchange.
    context.reconcile_transcript(
        "alice",
        (
            ("Scene request", "Scene reply", 20),
            ("Latest request", "Latest reply", None),
        ),
    )

    transcript = context.transcript("alice")
    assert [(entry.user, entry.assistant) for entry in transcript] == [
        ("First request", "First reply"),
        ("Scene request", "Scene reply"),
        ("Latest request", "Latest reply"),
    ]
    assert [entry.timestamp_us for entry in transcript] == [10, 20, None]


@pytest.mark.asyncio
async def test_conversation_uses_dialogue_summary_and_keeps_newer_exchanges_verbatim():
    class _DialogueLLM(_LLM):
        def __init__(self):
            super().__init__()
            self.compaction_messages = None

        async def chat(self, messages, **_kwargs):
            self.compaction_messages = messages
            assert "completed conversation history" in messages[0].content
            content = messages[1].content
            if "User: Check the facilities later." in content:
                assert "The user identified a copper label." in content
                summary = "The user identified a copper label and left the facilities check open."
            else:
                assert "User: The label is copper." in content
                assert "Assistant: The label was observed." in content
                summary = "The user identified a copper label."
            return ChatResponse(
                content=summary,
                reasoning=None,
                tool_calls=None,
                finish_reason="stop",
                raw={},
            )

    llm = _DialogueLLM()
    conversation = Conversation(llm, context_budget_bytes=700, compaction_input_budget_bytes=4096)  # type: ignore[arg-type]
    old_history = (
        ConversationExchange("The label is copper." + " x" * 160, "The label was observed." + " y" * 160),
        ConversationExchange("Check the facilities later." + " z" * 160, "That request remains open." + " w" * 160),
    )
    first_output = [chunk async for chunk in conversation.stream("What is the label?", "alice", history=old_history)]
    assert first_output == ["Hello there."]
    memory = conversation._context._participants["alice"]
    for _ in range(3):
        task = memory.dialogue_job
        if task is None:
            break
        assert await task is True
        await asyncio.sleep(0)
    assert conversation._context.dialogue_snapshot("alice")[0] == (
        "The user identified a copper label and left the facilities check open."
    )

    current = ConversationExchange("What is the label?", "Hello there.")
    second_output = [
        chunk
        async for chunk in conversation.stream(
            "What should I remember?",
            "alice",
            history=old_history + (current,),
        )
    ]
    assert second_output == ["Hello there."]
    prompt_context = llm.messages[-1][-2].content
    assert "The user identified a copper label and left the facilities check open." in prompt_context
    assert "Turn 1: user asked: What is the label?" in prompt_context
    assert "Turn 1: assistant said: “Hello there.”" in prompt_context
    assert llm.messages[-1][-1].content == "What should I remember?"
    assert "The label is copper." not in prompt_context
    assert len(conversation._context.transcript("alice")) == 4
    await conversation.close()


def test_dialogue_compaction_does_not_split_an_oversized_exchange():
    context = _ConversationContext()
    context.append("alice", "é" * 8, "answer")

    async def unused_compactor(_previous, _entries):
        raise AssertionError("oversized exchange must not be sent partially")

    assert (
        context.schedule_dialogue_compaction(
            "alice",
            trigger_budget_bytes=1,
            payload_budget_bytes=10,
            summary_budget_bytes=100,
            compact=unused_compactor,
        )
        is None
    )
    assert context.transcript("alice")[0].utf8_size > len("é" * 8) + len("answer")


@pytest.mark.asyncio
async def test_dialogue_compaction_keeps_appended_tail_and_cancels_on_departure():
    context = _ConversationContext()
    context.append("alice", "Old question", "Old answer")
    context.append("alice", "Middle question", "Middle answer")
    first_entered = asyncio.Event()
    finish_first = asyncio.Event()
    second_entered = asyncio.Event()
    calls = 0

    async def blocked_compaction(previous, entries):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            await finish_first.wait()
            assert previous == ""
            assert [entry.user for entry in entries] == ["Old question"]
            return "Old question was answered."
        second_entered.set()
        assert "Old question was answered." in previous
        assert [entry.user for entry in entries] == ["Middle question"]
        await asyncio.Event().wait()
        return "never completes"

    task = context.schedule_dialogue_compaction(
        "alice",
        trigger_budget_bytes=1,
        payload_budget_bytes=180,
        summary_budget_bytes=100,
        compact=blocked_compaction,
    )
    assert task is not None
    await first_entered.wait()
    context.append("alice", "New question", "New answer")
    finish_first.set()
    assert await task is True
    await second_entered.wait()
    summary, recent = context.dialogue_snapshot("alice")
    assert summary == "Old question was answered."
    assert [(entry.user, entry.assistant) for entry in recent] == [
        ("Middle question", "Middle answer"),
        ("New question", "New answer"),
    ]
    task = context._participants["alice"].dialogue_job
    assert task is not None
    await context.clear("alice")
    assert task.cancelled()
    assert context.transcript("alice") == ()


@pytest.mark.asyncio
async def test_runtime_named_context_topics_use_configured_modes_per_participant():
    llm = _LLM()
    tea = Topic("tea.status", ConversationContextUpdate)
    xr = Topic("xr.observations", ConversationContextUpdate)
    conversation = Conversation(
        llm,
        summary_context_topics=(tea.name,),
        incremental_context_topics=(xr.name,),
    )  # type: ignore[arg-type]
    runtime = AgentRuntime()
    runtime.register("conversation-context", conversation.context_subscriber())

    async with runtime:
        await runtime.publish(tea, {"text": "Tea timer is at 4 minutes."}, participant_id="alice")
        await runtime.publish(tea, {"text": "Timer is now at 6 minutes."}, participant_id="alice")
        await runtime.publish(xr, {"text": "The exhibit has three panels."}, participant_id="alice")
        await runtime.publish(xr, {"text": "One panel is interactive."}, participant_id="alice")
        await runtime.publish(
            tea,
            {"text": "A different timer is at 2 minutes."},
            participant_id="bob",
        )

    alice = conversation._context.sections("alice")
    bob = conversation._context.sections("bob")
    assert alice == (
        ("tea.status", "Timer is now at 6 minutes."),
        ("xr.observations", "Compacted source facts."),
    )
    assert bob == (("tea.status", "A different timer is at 2 minutes."),)
    assert "Timer was extended once" not in repr(alice)


def test_context_update_is_text_only():
    assert ConversationContextUpdate(text="A source-owned fact.").model_dump() == {"text": "A source-owned fact."}
    with pytest.raises(ValueError):
        ConversationContextUpdate(text="A source-owned fact.", mode="summary")


def test_context_topic_names_must_be_nonempty_and_unique():
    llm = _LLM()

    with pytest.raises(ValueError, match="sequence of topic names"):
        Conversation(llm, summary_context_topics="tea.status")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        Conversation(llm, summary_context_topics=("tea.status", " "))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        Conversation(llm, summary_context_topics=("tea.status", None))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        Conversation(llm, summary_context_topics=("tea.status", "tea.status"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="across summary and incremental"):
        Conversation(llm, summary_context_topics=("tea.status",), incremental_context_topics=("tea.status",))
    with pytest.raises(ValueError, match="sequence of topic names"):
        Conversation(llm, incremental_context_topics="tea.status")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence of topic names"):
        Conversation(llm, summary_context_topics=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one context topic"):
        Conversation(llm).context_subscriber()  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_context_recency_uses_publication_time_and_keeps_it_after_compaction():
    context = _ConversationContext()
    await context.update(
        "alice",
        "tea.status",
        "incremental",
        "Timer started.",
        context_budget_bytes=1000,
        timestamp_us=1_790_856_000_000_000,
    )
    summary_call = []

    async def compact(source, previous, entries):
        summary_call.append((source, previous, entries[0].timestamp_us))
        return "Timer was started."

    task = context.schedule_context_compaction("alice", "tea.status", payload_budget_bytes=1000, compact=compact)
    assert task is not None and await task is True
    assert summary_call == [("tea.status", "", 1_790_856_000_000_000)]
    section = context.sections_with_recency("alice")[0]
    assert section[1] == "Timer was started."
    assert "2026-10-01 12:00 UTC" in section[2]

    await context.update(
        "alice",
        "tea.status",
        "incremental",
        "Timer extended.",
        context_budget_bytes=1000,
        timestamp_us=1_790_859_600_000_000,
    )
    task = context.schedule_context_compaction("alice", "tea.status", payload_budget_bytes=1000, compact=compact)
    assert task is not None and await task is True
    assert summary_call[1][1] == (
        "Earlier summary (publication times around 2026-10-01 12:00 UTC):\nTimer was started."
    )

    await context.update(
        "alice",
        "tea.status",
        "summary",
        "Timer is now at 6 minutes.",
        context_budget_bytes=1000,
        timestamp_us=1_790_859_600_000_000,
    )
    assert context.sections_with_recency("alice")[0][2] == "around 2026-10-01 13:00 UTC"


@pytest.mark.asyncio
async def test_incremental_context_budget_counts_rendered_publication_time_label():
    context = _ConversationContext()
    summary_budget = len("Source tea (publication time around 2026-10-01 12:00 UTC):\né".encode("utf-8"))
    await context.update(
        "alice",
        "tea",
        "summary",
        "é",
        context_budget_bytes=summary_budget,
        timestamp_us=1_790_856_000_000_000,
    )
    retained, dropped = await context.update(
        "alice",
        "tea",
        "incremental",
        "水",
        context_budget_bytes=summary_budget,
        timestamp_us=1_790_856_000_000_000,
    )

    assert (retained, dropped) == (False, True)
    assert context.sections("alice") == (("tea", "é"),)
    assert context.sections_with_recency("alice")[0][2] == "around 2026-10-01 12:00 UTC"


@pytest.mark.asyncio
async def test_context_topic_rejects_unscoped_update_and_oversized_summary_drops_old_source(caplog):
    conversation = Conversation(_LLM(), context_budget_bytes=1000)  # type: ignore[arg-type]
    await conversation._update_context("alice", "tea", ConversationContextUpdate(text="old"), mode="summary")
    conversation._context_budget_bytes = 5  # noqa: SLF001
    await conversation._update_context("alice", "tea", ConversationContextUpdate(text="é"), mode="summary")
    await conversation._update_context(None, "tea", ConversationContextUpdate(text="shared"), mode="summary")

    assert conversation._context.sections("alice") == ()
    assert "oversized context summary" in caplog.text
    assert "without participant" in caplog.text
    rendered_summary_size = len("Source tea (publication time unknown):\né".encode("utf-8"))
    assert f"size_bytes={rendered_summary_size}" in caplog.text


@pytest.mark.asyncio
async def test_summary_update_supersedes_inflight_compaction_and_departure_blocks_late_context():
    entered = asyncio.Event()
    never_finish = asyncio.Event()

    class _BlockingLLM(_LLM):
        async def chat(self, _messages, **_kwargs):
            entered.set()
            await never_finish.wait()
            return await super().chat(_messages)

    conversation = Conversation(_BlockingLLM())  # type: ignore[arg-type]
    await conversation._update_context(
        "alice", "tea", ConversationContextUpdate(text="timer started"), mode="incremental"
    )
    await entered.wait()

    await conversation._update_context(
        "alice", "tea", ConversationContextUpdate(text="timer is at 6 minutes"), mode="summary"
    )
    assert conversation._context.sections("alice") == (("tea", "timer is at 6 minutes"),)

    await conversation.participant_left("alice")
    await conversation._update_context(
        "alice", "tea", ConversationContextUpdate(text="late stale update"), mode="summary"
    )
    assert conversation._context.sections("alice") == ()
    await conversation.close()


@pytest.mark.asyncio
async def test_incremental_update_during_compaction_keeps_prefix_and_compacts_new_tail():
    context = _ConversationContext()
    await context.update("alice", "tea", "incremental", "first fact", context_budget_bytes=100)
    entered = asyncio.Event()
    finish_first = asyncio.Event()
    second_entered = asyncio.Event()
    calls = 0

    async def compact(_source, previous, entries):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await finish_first.wait()
            assert previous == ""
            assert [entry.user for entry in entries] == ["first fact"]
            return "summary of first fact"
        second_entered.set()
        assert "summary of first fact" in previous
        assert [entry.user for entry in entries] == ["second fact"]
        return "summary of both facts"

    first = context.schedule_context_compaction("alice", "tea", payload_budget_bytes=100, compact=compact)
    assert first is not None
    await entered.wait()
    await context.update("alice", "tea", "incremental", "second fact", context_budget_bytes=100)
    assert context.schedule_context_compaction("alice", "tea", payload_budget_bytes=100, compact=compact) is first
    finish_first.set()
    assert await first is True
    await second_entered.wait()
    second = context._participants["alice"].context_jobs["tea"]
    assert await second is True
    assert context.sections("alice") == (("tea", "summary of both facts"),)
