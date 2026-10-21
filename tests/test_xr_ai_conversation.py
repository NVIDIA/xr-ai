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

import xr_ai_conversation as conversation_package  # noqa: E402
from xr_ai_conversation import (  # noqa: E402
    Conversation,
    ConversationContextUpdate,
)
from xr_ai_conversation._context import _ConversationContext  # noqa: E402
from xr_ai_conversation._conversation import _messages  # noqa: E402
from xr_ai_routing import (  # noqa: E402
    Admission,
    AgentExchange,
    AgentRouter,
    AgentTurn,
    DestinationResult,
    RoutableAgent,
    RoutingAid,
)
from xr_ai_runtime import AgentRuntime, Topic  # noqa: E402


def test_conversation_explicitly_implements_routable_agent():
    assert Conversation.__bases__ == (RoutableAgent,)
    assert all(
        callable(Conversation.__dict__.get(method)) for method in ("routing_aid", "admit", "answer", "release")
    )


def test_conversation_package_exports_only_engine_types():
    assert conversation_package.__all__ == ["Conversation", "ConversationContextUpdate"]


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
    front_end = AgentRouter({"conversation": conversation}, _Decisions())  # type: ignore[arg-type]
    output = []

    async def emit(source, text):
        output.append((source, text))

    first = await front_end.handle(AgentTurn("alice", "turn-1", "Hello."), emit=emit)
    await conversation.participant_left("alice")
    second = await front_end.handle(AgentTurn("alice", "turn-2", "Say that again."), emit=emit)

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
            self.choices = iter(("conversation", "scene", "conversation", "conversation", "scene", "conversation"))

        async def decide(self, _state, questions, **_kwargs):
            choice = next(self.choices)
            assert choice in questions["destination"].criteria
            return DecisionResponse(
                answers={
                    "destination": DecisionAnswer(
                        choice, 1.0, {label: float(label == choice) for label in questions["destination"].criteria}
                    )
                },
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
    front_end = AgentRouter(
        {"conversation": conversation, "scene": Scene()},
        Decisions(),  # type: ignore[arg-type]
        context_budget_bytes=160,
    )

    async def emit(_source, _text):
        pass

    long_opening = "Please keep this opening exchange out of the front-end window. " * 4
    await front_end.handle(AgentTurn("alice", "turn-1", long_opening), emit=emit)
    await front_end.handle(AgentTurn("alice", "turn-2", "Add a blue cube."), emit=emit)
    await front_end.handle(AgentTurn("alice", "turn-3", "What did you add?"), emit=emit)

    context = llm.messages[1][1].content
    assert "Turn 2: user asked: Add a blue cube." in context
    assert "Turn 2: assistant said: “Added the blue cube to the room.”" in context

    await front_end.handle(AgentTurn("alice", "turn-4", "What color was it?"), emit=emit)
    repeated_context = llm.messages[2][1].content
    assert repeated_context.count("Added the blue cube to the room.") == 1
    assert "Turn 2: user asked: Add a blue cube." in repeated_context

    # The same scene result can happen again immediately. Causal exchange
    # identity, not timestamp distance or text equality, keeps both turns.
    await front_end.handle(AgentTurn("alice", "turn-5", "Add a blue cube."), emit=emit)
    await front_end.handle(AgentTurn("alice", "turn-6", "What did you add this time?"), emit=emit)
    latest_context = llm.messages[3][1].content
    assert latest_context.count("Added the blue cube to the room.") == 2


@pytest.mark.asyncio
async def test_unknown_seed_does_not_mask_a_later_identical_routed_exchange():
    class Decisions:
        def __init__(self):
            self.choices = iter(("conversation", "scene", "conversation"))

        async def decide(self, _state, questions, **_kwargs):
            choice = next(self.choices)
            assert choice in questions["destination"].criteria
            return DecisionResponse(
                answers={
                    "destination": DecisionAnswer(
                        choice, 1.0, {label: float(label == choice) for label in questions["destination"].criteria}
                    )
                },
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
    front_end = AgentRouter(
        {"conversation": conversation, "scene": Scene()},
        Decisions(),  # type: ignore[arg-type]
        context_budget_bytes=1_000,
    )

    async def emit(_source, _text):
        pass

    seed = AgentExchange("Add a blue cube.", "Added the blue cube to the room.")
    await front_end.handle(
        AgentTurn("alice", "turn-1", "What should I do next?", history=(seed,)), emit=emit
    )
    await front_end.handle(AgentTurn("alice", "turn-2", "Add a blue cube."), emit=emit)
    await front_end.handle(AgentTurn("alice", "turn-3", "What did I add?"), emit=emit)

    prompt_context = llm.messages[1][1].content
    assert prompt_context.count("Added the blue cube to the room.") == 2


@pytest.mark.asyncio
async def test_admission_release_text_and_rerouted_answer_keep_separate_history_identity():
    class Decisions:
        def __init__(self):
            self.choices = iter(("scene", "conversation", "conversation"))

        async def decide(self, _state, questions, **_kwargs):
            choice = next(self.choices)
            assert choice in questions["destination"].criteria
            return DecisionResponse(
                answers={
                    "destination": DecisionAnswer(
                        choice, 1.0, {label: float(label == choice) for label in questions["destination"].criteria}
                    )
                },
                model="fake",
                usage={},
                raw={},
            )

    class Scene:
        def routing_aid(self, _participant_id):
            return RoutingAid("scene", "Applies requested changes to the current scene.")

        async def admit(self, _turn, *, focused):
            return Admission("release", text="Leaving the scene task.")

        async def answer(self, _turn, *, emit):  # pragma: no cover - release skips answer
            raise AssertionError("released destination must not answer")

        async def release(self, _participant_id):
            pass

    llm = _LLM()
    conversation = Conversation(llm, context_budget_bytes=1_000)  # type: ignore[arg-type]
    front_end = AgentRouter(
        {"conversation": conversation, "scene": Scene()},
        Decisions(),  # type: ignore[arg-type]
        context_budget_bytes=1_000,
    )

    async def emit(_source, _text):
        pass

    await front_end.handle(AgentTurn("alice", "turn-1", "What should happen next?"), emit=emit)
    await front_end.handle(AgentTurn("alice", "turn-2", "What did we decide?"), emit=emit)

    prompt_context = llm.messages[1][1].content
    assert prompt_context.count("Leaving the scene task.") == 1
    assert prompt_context.count("Hello there.") == 1
    assert [item.assistant for item in front_end._history["alice"]] == [
        "Leaving the scene task.",
        "Hello there.",
        "Hello there.",
    ]


@pytest.mark.asyncio
async def test_optional_current_view_tool_uses_typed_frame_and_streaming_vision_requests():
    llm = _CurrentViewLLM()
    frames = _CurrentFrame()
    vision = _StreamingVision()
    conversation = Conversation(llm, frames=frames, vision=vision)  # type: ignore[arg-type]
    front_end = AgentRouter({"conversation": conversation}, _Decisions())  # type: ignore[arg-type]
    output = []

    async def emit(source, text):
        output.append((source, text))

    result = await front_end.handle(AgentTurn("alice", "view-1", "What color is the sign?"), emit=emit)

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
    front_end = AgentRouter({"conversation": conversation}, _Decisions())  # type: ignore[arg-type]

    async def emit(_source, _text):
        pass

    task = asyncio.create_task(
        front_end.handle(AgentTurn("alice", "view-cancel", "Look here."), emit=emit)
    )
    await asyncio.wait_for(vision.started.wait(), timeout=1)
    await front_end.interrupt("alice")
    with pytest.raises(asyncio.CancelledError):
        await task

    assert llm.closed.is_set()
    assert vision.closed.is_set()


def test_message_history_keeps_full_exchanges_and_latest_request_separate():
    history = tuple(AgentExchange(f"Question {index}", f"Answer {index}") for index in range(6))
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
        (AgentExchange("Earlier question", "Earlier answer"),),
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
        AgentExchange(f"Question {index}: " + "q" * 160, f"Answer {index}: " + "a" * 160) for index in range(6)
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
    # A repeated utterance/result at a distinct known time is new, while
    # replaying that exact timestamped window must not duplicate it.
    context.reconcile_transcript("alice", (("Scene request", "Scene reply", 2_000_000),))
    context.reconcile_transcript(
        "alice",
        (("Scene request", "Scene reply", 2_000_000),),
    )

    transcript = context.transcript("alice")
    assert [(entry.user, entry.assistant) for entry in transcript] == [
        ("First request", "First reply"),
        ("Scene request", "Scene reply"),
        ("Latest request", "Latest reply"),
        ("Scene request", "Scene reply"),
    ]
    assert [entry.timestamp_us for entry in transcript] == [10, 20, None, 2_000_000]

    unknown_seed = _ConversationContext()
    unknown_seed.reconcile_transcript("alice", (("Scene request", "Scene reply", None),))
    new_known_id = object()
    unknown_seed.reconcile_transcript(
        "alice",
        (("Scene request", "Scene reply", 30),),
        exchange_ids=(new_known_id,),
    )
    assert len(unknown_seed.transcript("alice")) == 2


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
        AgentExchange("The label is copper." + " x" * 160, "The label was observed." + " y" * 160),
        AgentExchange("Check the facilities later." + " z" * 160, "That request remains open." + " w" * 160),
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

    current = AgentExchange("What is the label?", "Hello there.")
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


@pytest.mark.asyncio
async def test_close_joins_departing_context_compaction_through_repeated_cancellation():
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    finish = asyncio.Event()
    settled = asyncio.Event()
    close_started = asyncio.Event()

    class BlockingLLM(_LLM):
        async def chat(self, _messages, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish.wait()
                settled.set()

    topic = Topic("test.context", ConversationContextUpdate)
    conversation = Conversation(BlockingLLM(), incremental_context_topics=(topic.name,))  # type: ignore[arg-type]
    runtime = AgentRuntime()
    runtime.register("context", conversation.context_subscriber())

    async def close():
        close_started.set()
        await conversation.close()

    async with runtime:
        await runtime.publish(topic, ConversationContextUpdate(text="A note"), participant_id="alice")
        await entered.wait()
        departure = asyncio.create_task(conversation.participant_left("alice"))
        await cleaning.wait()
        closing = asyncio.create_task(close())
        joining = asyncio.create_task(conversation.close())
        try:
            await close_started.wait()
            assert not closing.done(), "close must join the already-departing participant's real cleanup"
            for _ in range(2):
                departure.cancel()
                closing.cancel()
                checkpoint = asyncio.Event()
                asyncio.get_running_loop().call_soon(checkpoint.set)
                await checkpoint.wait()
                assert not departure.done()
                assert not closing.done()
                assert not joining.done()
                assert not settled.is_set()
            finish.set()
            results = await asyncio.gather(departure, closing, return_exceptions=True)
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert settled.is_set()
            await joining
            await conversation.close()
        finally:
            finish.set()
            await asyncio.gather(departure, closing, joining, return_exceptions=True)


@pytest.mark.asyncio
async def test_context_cleanup_keeps_participants_independent_and_observes_job_failure(caplog):
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    finish = asyncio.Event()
    settled = asyncio.Event()

    class BlockingLLM(_LLM):
        async def chat(self, _messages, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish.wait()
                settled.set()
                raise RuntimeError("compactor cleanup failed after termination")

    incremental = Topic("test.notes", ConversationContextUpdate)
    summary = Topic("test.summary", ConversationContextUpdate)
    conversation = Conversation(
        BlockingLLM(),  # type: ignore[arg-type]
        incremental_context_topics=(incremental.name,),
        summary_context_topics=(summary.name,),
    )
    runtime = AgentRuntime()
    runtime.register("context", conversation.context_subscriber())
    async with runtime:
        await runtime.publish(incremental, ConversationContextUpdate(text="A note"), participant_id="alice")
        await entered.wait()
        await runtime.publish(incremental, ConversationContextUpdate(text="another note"), participant_id="alice")
        dropping = asyncio.create_task(conversation.participant_left("alice"))
        await cleaning.wait()
        releasing = asyncio.create_task(conversation.release("alice"))
        try:
            with pytest.raises(RuntimeError, match="cleanup is still in progress"):
                conversation.activate_participant("alice")
            assert [text async for text in conversation.stream("Hello", "bob")] == ["Hello there."]
            await runtime.publish(summary, ConversationContextUpdate(text="late Alice update"), participant_id="alice")
            assert not dropping.done()
            assert not releasing.done()
            assert not settled.is_set()
            finish.set()
            await asyncio.gather(dropping, releasing)
            assert settled.is_set()
            assert caplog.text.count("conversation context compaction failed") == 1
            conversation.activate_participant("alice")
            await runtime.publish(summary, ConversationContextUpdate(text="new session"), participant_id="alice")
            assert [text async for text in conversation.stream("Remember?", "alice")] == ["Hello there."]
            await conversation.close()
        finally:
            finish.set()
            await asyncio.gather(dropping, releasing, return_exceptions=True)
            await conversation.close()


@pytest.mark.asyncio
async def test_clear_joins_retired_compaction_without_recancelling_its_finally():
    context = _ConversationContext()
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    finish = asyncio.Event()
    settled = asyncio.Event()

    async def compact(_source, _previous, _entries):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()
            settled.set()

    await context.update("alice", "notes", "incremental", "A note", context_budget_bytes=1000)
    job = context.schedule_context_compaction("alice", "notes", payload_budget_bytes=1000, compact=compact)
    assert job is not None
    await entered.wait()
    await context.update("alice", "notes", "summary", "Authoritative summary", context_budget_bytes=1000)
    await cleaning.wait()
    clearing = asyncio.create_task(context.clear("alice"))
    try:
        checkpoint = asyncio.Event()
        asyncio.get_running_loop().call_soon(checkpoint.set)
        await checkpoint.wait()
        assert not clearing.done()
        assert job.cancelling() == 1
        assert not settled.is_set()
        finish.set()
        await clearing
        assert settled.is_set()
        assert context.sections("alice") == ()
    finally:
        finish.set()
        await clearing
