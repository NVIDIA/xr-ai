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
from xr_ai_conversation._context import _BackgroundSection, _ContextRecord, _ConversationContext  # noqa: E402
from xr_ai_conversation._conversation import _ContextBlock, _messages  # noqa: E402
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
    assert all(callable(Conversation.__dict__.get(method)) for method in ("routing_aid", "admit", "answer", "release"))


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
    assert "NEWER MATERIAL, OLDEST FIRST:" in history_message
    assert "User: Hello.\nAssistant: “Hello there.”" in history_message
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
    conversation = Conversation(llm, dialogue_budget_bytes=1_000)  # type: ignore[arg-type]
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
    assert "User: Add a blue cube.\nAssistant: “Added the blue cube to the room.”" in context

    await front_end.handle(AgentTurn("alice", "turn-4", "What color was it?"), emit=emit)
    repeated_context = llm.messages[2][1].content
    assert repeated_context.count("Added the blue cube to the room.") == 1
    assert "User: Add a blue cube." in repeated_context

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
    conversation = Conversation(llm, dialogue_budget_bytes=1_000)  # type: ignore[arg-type]
    front_end = AgentRouter(
        {"conversation": conversation, "scene": Scene()},
        Decisions(),  # type: ignore[arg-type]
        context_budget_bytes=1_000,
    )

    async def emit(_source, _text):
        pass

    seed = AgentExchange("Add a blue cube.", "Added the blue cube to the room.")
    await front_end.handle(AgentTurn("alice", "turn-1", "What should I do next?", history=(seed,)), emit=emit)
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
    conversation = Conversation(llm, dialogue_budget_bytes=1_000)  # type: ignore[arg-type]
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
        _ = [chunk async for chunk in invalid_conversation.stream("Look here.", "alice")]
    assert invalid_frames.requests == []
    assert invalid_llm.closed.is_set()

    unavailable_llm = _CurrentViewLLM()
    unavailable_frames = _CurrentFrame(error=FrameUnavailable("No camera frame is available."))
    unavailable_conversation = Conversation(
        unavailable_llm,
        frames=unavailable_frames,
        vision=_StreamingVision(),
    )  # type: ignore[arg-type]
    output = [chunk async for chunk in unavailable_conversation.stream("Look here.", "alice")]

    assert output == ["No camera frame is available."]
    assert len(unavailable_frames.requests) == 1
    assert unavailable_llm.closed.is_set()


@pytest.mark.asyncio
async def test_current_view_rejects_an_empty_query_before_capturing_a_frame():
    llm = _CurrentViewLLM(arguments='{"query":""}')
    frames = _CurrentFrame()
    conversation = Conversation(llm, frames=frames, vision=_StreamingVision())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="invalid current_view arguments"):
        _ = [chunk async for chunk in conversation.stream("Look here.", "alice")]
    assert frames.requests == []


@pytest.mark.asyncio
async def test_camera_unavailable_reply_is_recorded_for_follow_ups_only_when_delivered():
    frames = _CurrentFrame(error=FrameUnavailable("No camera frame is available."))
    conversation = Conversation(_CurrentViewLLM(), frames=frames, vision=_StreamingVision())  # type: ignore[arg-type]

    # A consumer that stops before the reply is fully delivered records nothing.
    stream = conversation.stream("What color is the sign?", "alice")
    await stream.aclose()
    assert conversation._context.dialogue_snapshot("alice")[1] == ()

    assert [chunk async for chunk in conversation.stream("What color is the sign?", "alice")] == [
        "No camera frame is available."
    ]
    llm = _LLM()
    conversation._llm = llm
    _ = [chunk async for chunk in conversation.stream("Why?", "alice")]
    prompt = llm.messages[-1][-2].content
    assert "User: What color is the sign?\nAssistant: “No camera frame is available.”" in prompt
    await conversation.close()


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

    task = asyncio.create_task(front_end.handle(AgentTurn("alice", "view-cancel", "Look here."), emit=emit))
    await asyncio.wait_for(vision.started.wait(), timeout=1)
    await front_end.interrupt("alice")
    with pytest.raises(asyncio.CancelledError):
        await task

    assert llm.closed.is_set()
    assert vision.closed.is_set()


@pytest.mark.asyncio
async def test_prompt_puts_summaries_first_then_newer_material_in_time_order():
    """Layout eval: a later correction must follow the summary or update it supersedes."""

    minute = 60_000_000
    noon = 1_790_856_000_000_000
    conversation = Conversation(
        _LLM(),  # type: ignore[arg-type]
        summary_context_topics=("tea.status",),
        incremental_context_topics=("notes",),
    )
    context = conversation._context
    context.append("alice", "Correction: the meeting moved to Tuesday.", "Noted.", timestamp_us=noon + 60 * minute)
    memory = context._participants["alice"]
    memory.dialogue_background = _BackgroundSection("The meeting is on Monday.", -1, noon, noon, False)
    memory.backgrounds["notes"] = _BackgroundSection("Bench 3 must not be used.", 0, noon, noon, False)
    memory.context_order.append("notes")
    memory.context_records["notes"] = [_ContextRecord(1, "Bench 3 is usable again.", noon + 30 * minute)]
    await context.update(
        "alice",
        "tea.status",
        "summary",
        "Timer is running.",
        context_budget_bytes=1000,
        timestamp_us=noon + 45 * minute,
    )

    messages = _messages(
        "When is the meeting?",
        instructions="",
        vision_available=True,
        summaries=conversation._summary_blocks("alice"),
        newer=conversation._newer_blocks("alice"),
    )

    assert [message.role for message in messages] == ["system", "user", "user"]
    assert messages[-1].content == "When is the meeting?"
    # The system prompt carries no clock, so it stays a stable prefix.
    assert "Current time" not in messages[0].content
    rendered = messages[1].content
    order = [
        "SUMMARIES OF OLDER MATERIAL:",
        "Earlier conversation with this user (exchange completion times around 2026-10-01 12:00 UTC):\n"
        "The meeting is on Monday.",
        "Source notes, earlier updates (publication times around 2026-10-01 12:00 UTC):\nBench 3 must not be used.",
        "NEWER MATERIAL, OLDEST FIRST:",
        "Source notes, update published around 2026-10-01 12:30 UTC:\nBench 3 is usable again.",
        "Source tea.status, current state published around 2026-10-01 12:45 UTC:\nTimer is running.",
        "Exchange completed around 2026-10-01 13:00 UTC:\nUser: Correction: the meeting moved to Tuesday.",
        "Current time: around",
    ]
    missing = [text for text in order if text not in rendered]
    assert not missing, rendered
    positions = [rendered.index(text) for text in order]
    assert positions == sorted(positions)
    await conversation.close()


def test_prompt_budget_drops_oldest_newer_material_first_and_keeps_the_newest_exchange():
    summaries = (
        _ContextBlock("dialogue summary", None, "Earlier conversation with this user:\n" + "s" * 80),
        _ContextBlock("notes", None, "Source notes, earlier updates:\n" + "n" * 80),
    )
    newer = tuple(
        _ContextBlock("dialogue", index, f"Exchange {index}:\nUser: Question {index} " + "q" * 120)
        for index in range(6)
    )
    query = "Current question " + "z" * 300
    messages = _messages(
        query,
        instructions="keep the application instruction",
        vision_available=True,
        summaries=summaries,
        newer=newer,
        context_budget_bytes=650,
    )
    context = messages[-2].content

    assert messages[-1].content == query
    assert "keep the application instruction" in messages[0].content
    assert len(context.encode("utf-8")) <= 650
    assert "Question 5" in context
    assert "Question 0" not in context
    # Summaries outlive older raw material.
    assert "Earlier conversation with this user" in context

    # Under more pressure, source summaries go before the conversation summary.
    tight = _messages(
        query, instructions="", vision_available=True, summaries=summaries, newer=newer, context_budget_bytes=520
    )[-2].content
    assert "Question 5" in tight
    assert "Earlier conversation with this user" in tight
    assert "Source notes" not in tight


@pytest.mark.asyncio
async def test_private_dialogue_compaction_replaces_old_prefix_and_does_not_replay_it():
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

    async def compact(previous, entries, limit_bytes):
        assert previous == ""
        assert limit_bytes == 1000
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
    # Compacted exchanges are discarded, and a front-end window that still
    # contains them is not replayed.
    assert [entry.user for entry in context.transcript("alice")] == ["Keep the facilities check open."]
    context.reconcile_transcript(
        "alice",
        (
            ("The label says copper.", "I will remember copper."),
            ("Keep the facilities check open.", "I will keep it open."),
        ),
    )
    assert [entry.user for entry in context.transcript("alice")] == ["Keep the facilities check open."]


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
            assert "at most 175 bytes" in messages[0].content
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
    conversation = Conversation(llm, dialogue_budget_bytes=700, compaction_input_budget_bytes=4096)  # type: ignore[arg-type]
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
    assert "User: What is the label?\nAssistant: “Hello there.”" in prompt_context
    assert llm.messages[-1][-1].content == "What should I remember?"
    assert "The label is copper." not in prompt_context
    assert [entry.user for entry in conversation._context.transcript("alice")] == [
        "What is the label?",
        "What should I remember?",
    ]
    await conversation.close()


@pytest.mark.asyncio
async def test_oversized_exchange_is_clipped_and_later_dialogue_still_compacts():
    """Review finding: a 13 KB first request used to stall all later compaction."""

    requests = []

    async def compact(previous, entries, _limit_bytes):
        requests.append(len(previous.encode("utf-8")) + sum(entry.utf8_size for entry in entries))
        return f"summary {len(requests)}"

    context = _ConversationContext()
    context.append("alice", "initial request " + "x" * 13_000, "ok")
    for index in range(22):
        context.append("alice", f"preference {index}", "noted")
        task = context.schedule_dialogue_compaction(
            "alice", trigger_budget_bytes=300, payload_budget_bytes=600, summary_budget_bytes=100, compact=compact
        )
        while task is not None:
            await task
            task = context._participants["alice"].dialogue_job

    assert requests and max(requests) <= 600
    summary, recent = context.dialogue_snapshot("alice")
    assert summary == f"summary {len(requests)}"
    assert recent[-1].user == "preference 21"
    assert all("x" * 1000 not in entry.user for entry in recent)


@pytest.mark.asyncio
async def test_oversized_summary_is_condensed_then_truncated_and_compaction_continues(caplog):
    """Review findings: an oversized or near-budget summary used to block later updates."""

    calls = []

    async def compact(previous, entries, limit_bytes):
        calls.append((len(entries), limit_bytes))
        return "s" * 900 if len(calls) <= 2 else "short summary"

    def schedule():
        return context.schedule_dialogue_compaction(
            "alice", trigger_budget_bytes=512, payload_budget_bytes=4096, summary_budget_bytes=256, compact=compact
        )

    context = _ConversationContext()
    context.append("alice", "a" * 300, "ok")
    context.append("alice", "b" * 300, "ok")
    await schedule()
    # The merge overshoots, one condense request also overshoots, and the
    # result is truncated to the limit instead of being rejected.
    assert calls == [(1, 256), (0, 256)]
    assert context.dialogue_snapshot("alice")[0] == "s" * 256
    assert "condensing oversized context summary" in caplog.text

    for text in ("c", "d"):
        context.append("alice", text * 300, "correction")
    await schedule()
    assert context.dialogue_snapshot("alice")[0] == "short summary"
    assert [entry.user for entry in context.transcript("alice")] == ["d" * 300]


@pytest.mark.asyncio
async def test_failed_dialogue_compaction_retries_later_and_bounds_uncompacted_storage(caplog):
    results = iter(["", RuntimeError("model unavailable")])

    async def compact(_previous, _entries, _limit_bytes):
        result = next(results, "recovered summary")
        if isinstance(result, Exception):
            raise result
        return result

    context = _ConversationContext()

    async def turn(index):
        context.append("alice", f"question {index} " + "q" * 40, "a" * 40)
        task = context.schedule_dialogue_compaction(
            "alice", trigger_budget_bytes=100, payload_budget_bytes=1000, summary_budget_bytes=50, compact=compact
        )
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    await turn(0)
    await turn(1)  # empty summary: rejected, exchanges kept
    assert caplog.text.count("conversation context compaction failed") == 1
    assert len(context.transcript("alice")) == 2
    await turn(2)  # model error: exchanges kept
    assert caplog.text.count("conversation context compaction failed") == 2
    assert len(context.transcript("alice")) == 3

    await turn(3)  # retried on the next turn
    assert context.dialogue_snapshot("alice")[0] == "recovered summary"
    assert [entry.user[:10] for entry in context.transcript("alice")] == ["question 3"]

    results = iter([RuntimeError("model unavailable")] * 100)
    for index in range(4, 40):
        await turn(index)
    assert sum(entry.utf8_size for entry in context.transcript("alice")) <= 800
    assert context.transcript("alice")[-1].user.startswith("question 39")
    assert "dropping uncompacted context entry" in caplog.text


@pytest.mark.asyncio
async def test_dialogue_compaction_keeps_appended_tail_and_cancels_on_departure():
    context = _ConversationContext()
    context.append("alice", "Old question", "Old answer")
    context.append("alice", "Middle question", "Middle answer")
    first_entered = asyncio.Event()
    finish_first = asyncio.Event()
    second_entered = asyncio.Event()
    calls = 0

    async def blocked_compaction(previous, entries, _limit_bytes):
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
        topic_budget_bytes=160,
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

    job = conversation._context._participants["alice"].context_jobs.get(xr.name)
    if job is not None:
        await job
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

    async def compact(source, previous, entries, _limit_bytes):
        summary_call.append((source, previous, entries[0].timestamp_us))
        return "Timer was started."

    def schedule():
        return context.schedule_context_compaction(
            "alice",
            "tea.status",
            trigger_budget_bytes=1,
            payload_budget_bytes=1000,
            summary_budget_bytes=100,
            compact=compact,
        )

    task = schedule()
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
    task = schedule()
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
async def test_source_summary_near_budget_does_not_block_later_updates():
    """Review findings: a near-budget compacted summary used to discard every later update."""

    calls = []

    class _SummaryLLM(_LLM):
        async def chat(self, messages, **_kwargs):
            calls.append(messages[0].content)
            # The first merge overshoots and the condense request still does.
            content = "s" * 800 if len(calls) <= 2 else "Latest corrections applied."
            return ChatResponse(content=content, reasoning=None, tool_calls=None, finish_reason="stop", raw={})

    topic = "notes"
    conversation = Conversation(
        _SummaryLLM(),  # type: ignore[arg-type]
        incremental_context_topics=(topic,),
        topic_budget_bytes=1024,
        compaction_input_budget_bytes=4096,
    )  # type: ignore[arg-type]
    context = conversation._context

    async def publish(text):
        await conversation._update_context("alice", topic, ConversationContextUpdate(text=text), mode="incremental")
        while (job := context._participants["alice"].context_jobs.get(topic)) is not None:
            await job

    await publish("n" * 600)
    assert context.sections("alice") == ((topic, "s" * 256),)
    assert "at most 256 bytes" in calls[0]
    assert calls[1].startswith("Shorten this summary to at most 256 bytes")

    for index in range(3):
        await publish(f"correction {index} " + "c" * 290)
    # The first two corrections are merged; the newest stays verbatim below the trigger.
    assert len(calls) == 3
    assert context.sections("alice") == ((topic, "Latest corrections applied.\ncorrection 2 " + "c" * 290),)
    await conversation.close()


@pytest.mark.asyncio
async def test_incremental_update_larger_than_any_budget_is_clipped_and_compacted():
    requests = []

    async def compact(source, previous, entries, _limit_bytes):
        requests.append(len(previous.encode("utf-8")) + sum(entry.utf8_size for entry in entries))
        return f"{source} summary {len(requests)}"

    context = _ConversationContext()
    stored, trimmed = await context.update("alice", "log", "incremental", "line\n" * 5000, context_budget_bytes=100)
    assert (stored, trimmed) == (True, False)
    for index in range(5):
        await context.update("alice", "log", "incremental", f"event {index}", context_budget_bytes=100)
    task = context.schedule_context_compaction(
        "alice", "log", trigger_budget_bytes=50, payload_budget_bytes=300, summary_budget_bytes=25, compact=compact
    )
    while task is not None:
        await task
        task = context._participants["alice"].context_jobs.get("log")

    assert requests and max(requests) <= 300
    assert context.sections("alice") == (("log", f"log summary {len(requests)}"),)


@pytest.mark.asyncio
async def test_failed_source_compaction_keeps_updates_retries_and_bounds_storage(caplog):
    outcomes = iter([RuntimeError("model unavailable"), "", "recovered"])

    async def compact(_source, _previous, _entries, _limit_bytes):
        outcome = next(outcomes, RuntimeError("model unavailable"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    context = _ConversationContext()

    async def publish(text):
        await context.update("alice", "notes", "incremental", text, context_budget_bytes=1000)
        task = context.schedule_context_compaction(
            "alice",
            "notes",
            trigger_budget_bytes=50,
            payload_budget_bytes=1000,
            summary_budget_bytes=25,
            compact=compact,
        )
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    await publish("first note " * 6)
    await publish("second note")
    assert caplog.text.count("conversation context compaction failed") == 2
    assert context.sections("alice")[0][1].startswith("first note")
    await publish("third note")
    assert context.sections("alice") == (("notes", "recovered"),)

    for index in range(60):
        await publish(f"note {index} " + "x" * 30)
    records = context._participants["alice"].context_records["notes"]
    assert sum(record.utf8_size for record in records) <= 400
    assert records[-1].text.startswith("note 59")
    assert "dropping uncompacted context entry" in caplog.text


@pytest.mark.asyncio
async def test_context_topic_rejects_unscoped_update_and_oversized_summary_drops_old_source(caplog):
    conversation = Conversation(_LLM(), summary_context_topics=("tea",), topic_budget_bytes=1000)  # type: ignore[arg-type]
    await conversation._update_context("alice", "tea", ConversationContextUpdate(text="old"), mode="summary")
    conversation._topic_budgets["tea"] = 5  # noqa: SLF001
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

    conversation = Conversation(_BlockingLLM(), incremental_context_topics=("tea",), topic_budget_bytes=100)  # type: ignore[arg-type]
    await conversation._update_context(
        "alice", "tea", ConversationContextUpdate(text="timer started " * 4), mode="incremental"
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

    async def compact(_source, previous, entries, _limit_bytes):
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

    def schedule():
        return context.schedule_context_compaction(
            "alice", "tea", trigger_budget_bytes=1, payload_budget_bytes=100, summary_budget_bytes=100, compact=compact
        )

    first = schedule()
    assert first is not None
    await entered.wait()
    await context.update("alice", "tea", "incremental", "second fact", context_budget_bytes=100)
    assert schedule() is first
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
    conversation = Conversation(BlockingLLM(), incremental_context_topics=(topic.name,), topic_budget_bytes=100)  # type: ignore[arg-type]
    runtime = AgentRuntime()
    runtime.register("context", conversation.context_subscriber())

    async def close():
        close_started.set()
        await conversation.close()

    async with runtime:
        await runtime.publish(topic, ConversationContextUpdate(text="A note " * 10), participant_id="alice")
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
        topic_budget_bytes=200,
    )
    note = "A note " * 15
    runtime = AgentRuntime()
    runtime.register("context", conversation.context_subscriber())
    async with runtime:
        await runtime.publish(incremental, ConversationContextUpdate(text=note), participant_id="alice")
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

    async def compact(_source, _previous, _entries, _limit_bytes):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()
            settled.set()

    await context.update("alice", "notes", "incremental", "A note", context_budget_bytes=1000)
    job = context.schedule_context_compaction(
        "alice", "notes", trigger_budget_bytes=1, payload_budget_bytes=1000, summary_budget_bytes=100, compact=compact
    )
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


async def _settle(conversation, participant):
    memory = conversation._context._participants[participant]
    while jobs := [job for job in (memory.dialogue_job, *memory.context_jobs.values()) if job is not None]:
        await asyncio.gather(*jobs, return_exceptions=True)
        # Awaiting finished jobs does not yield; let their completion callbacks run.
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_dialogue_and_sources_together_stay_within_the_context_budget(caplog):
    """Live finding: per-log triggers let dialogue plus sources overflow, dropping an unsummarized correction."""

    llm = _LLM()
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        summary_context_topics=("status",),
        incremental_context_topics=("notes",),
        dialogue_budget_bytes=1000,
        topic_budget_bytes=1000,
    )
    for index in range(30):
        await conversation._update_context(
            "alice", "notes", ConversationContextUpdate(text=f"note {index} " + "n" * 120), mode="incremental"
        )
        await conversation._update_context(
            "alice", "status", ConversationContextUpdate(text=f"status {index} " + "s" * 300), mode="summary"
        )
        [chunk async for chunk in conversation.stream(f"question {index} " + "q" * 150, "alice")]
        await _settle(conversation, "alice")
        context = llm.messages[-1][-2].content
        assert len(context.encode("utf-8")) <= 3000

    assert "dropping old conversation context" not in caplog.text
    assert f"question 29 {'q' * 150}" in llm.messages[-1][-1].content
    await conversation.close()


@pytest.mark.asyncio
async def test_oversized_newest_exchange_is_clipped_in_the_prompt_instead_of_wiping_context():
    """Live finding: a 17 KB exchange used to remove the summary, sources and itself from the next prompt."""

    llm = _LLM()
    conversation = Conversation(
        llm, incremental_context_topics=("notes",), dialogue_budget_bytes=1500, topic_budget_bytes=1500
    )  # type: ignore[arg-type]
    await conversation._update_context(
        "alice", "notes", ConversationContextUpdate(text="Bench 3 is usable."), mode="incremental"
    )
    pasted_log = "Controller log:\n" + "heartbeat ok\n" * 1500 + "ERROR interlock tripped code E-4471"
    [chunk async for chunk in conversation.stream(pasted_log, "alice")]
    await _settle(conversation, "alice")
    [chunk async for chunk in conversation.stream("What was the error code?", "alice")]

    context = llm.messages[-1][-2].content
    assert len(context.encode("utf-8")) <= 3000
    assert "Controller log:" in context
    assert "E-4471" in context
    assert "bytes omitted" in context
    assert "Bench 3 is usable." in context
    await conversation.close()


@pytest.mark.asyncio
async def test_oversized_summary_topic_update_is_clipped_to_its_share(caplog):
    conversation = Conversation(_LLM(), summary_context_topics=("status",), topic_budget_bytes=200)  # type: ignore[arg-type]
    text = "Status begins. " + "detail " * 200 + "Status ends."
    await conversation._update_context("alice", "status", ConversationContextUpdate(text=text), mode="summary")

    ((source, stored),) = conversation._context.sections("alice")
    assert source == "status"
    assert stored.startswith("Status begins.") and stored.endswith("Status ends.")
    assert len(f"Source status (publication time unknown):\n{stored}".encode("utf-8")) <= 200
    assert "clipping oversized context summary" in caplog.text
    await conversation.close()


@pytest.mark.asyncio
async def test_uncompacted_source_dump_is_clipped_in_the_prompt_instead_of_dropped():
    """Live finding: a raw source update larger than the budget removed the whole source from the next prompt."""

    llm = _LLM()
    conversation = Conversation(
        llm, incremental_context_topics=("notes",), dialogue_budget_bytes=1500, topic_budget_bytes=1500
    )  # type: ignore[arg-type]
    await conversation._update_context(
        "alice", "notes", ConversationContextUpdate(text="Bench 3 is usable."), mode="incremental"
    )
    # Store a large raw update without letting compaction run before the next prompt.
    await conversation._context.update(
        "alice", "notes", "incremental", "dump " * 2000 + "Purge started.", context_budget_bytes=1500
    )
    [chunk async for chunk in conversation.stream("Status?", "alice")]

    context = llm.messages[-1][-2].content
    assert len(context.encode("utf-8")) <= 3000
    assert "Bench 3 is usable." in context
    assert "Purge started." in context
    assert "bytes omitted" in context
    await conversation.close()


@pytest.mark.asyncio
async def test_each_context_section_uses_its_configured_budget():
    llm = _LLM()
    with pytest.raises(ValueError, match="one entry per configured topic"):
        Conversation(llm, incremental_context_topics=("a", "b"), topic_budget_bytes={"a": 100})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integers"):
        Conversation(llm, incremental_context_topics=("a",), topic_budget_bytes={"a": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integers"):
        Conversation(llm, dialogue_budget_bytes=0)  # type: ignore[arg-type]

    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        incremental_context_topics=("small", "large"),
        dialogue_budget_bytes=2000,
        topic_budget_bytes={"small": 200, "large": 4000},
    )
    note = "a fact worth keeping " * 6
    for source in ("small", "large"):
        await conversation._update_context("alice", source, ConversationContextUpdate(text=note), mode="incremental")
    await _settle(conversation, "alice")

    # Only the small section passed half of its own budget and was compacted.
    assert conversation._context.sections("alice") == (
        ("small", "Compacted source facts."),
        ("large", note),
    )
    [chunk async for chunk in conversation.stream("Hello", "alice")]
    assert len(llm.messages[-1][-2].content.encode("utf-8")) <= 2000 + 200 + 4000
    await conversation.close()


@pytest.mark.asyncio
async def test_only_background_compaction_requests_opt_into_capped_thinking():
    class _RecordingLLM(_LLM):
        def __init__(self):
            super().__init__()
            self.chat_calls = []

        async def chat(self, messages, **kwargs):
            self.chat_calls.append(kwargs)
            return await super().chat(messages)

    with pytest.raises(ValueError, match="positive integers"):
        Conversation(_LLM(), compaction_thinking_budget=0)  # type: ignore[arg-type]

    for thinking_budget, expected in (
        (None, {}),
        (512, {"enable_thinking": True, "thinking_budget": 512, "max_tokens": 512 + 25}),
    ):
        llm = _RecordingLLM()
        conversation = Conversation(
            llm,  # type: ignore[arg-type]
            incremental_context_topics=("notes",),
            topic_budget_bytes=100,
            compaction_thinking_budget=thinking_budget,
        )
        await conversation._update_context(
            "alice", "notes", ConversationContextUpdate(text="n" * 80), mode="incremental"
        )
        await _settle(conversation, "alice")
        [chunk async for chunk in conversation.stream("Hello", "alice")]

        assert llm.chat_calls == [expected]
        # Answers keep the model's configured default.
        assert all("enable_thinking" not in call for call in llm.calls)
        await conversation.close()


@pytest.mark.asyncio
async def test_answers_stay_responsive_while_compaction_is_stuck_and_commit_later():
    release = asyncio.Event()
    entered = asyncio.Event()

    class _SlowCompactionLLM(_LLM):
        async def chat(self, messages, **_kwargs):
            entered.set()
            await release.wait()
            return ChatResponse(
                content="The user prefers nanometers.", reasoning=None, tool_calls=None, finish_reason="stop", raw={}
            )

    llm = _SlowCompactionLLM()
    conversation = Conversation(llm, dialogue_budget_bytes=400)  # type: ignore[arg-type]
    for index in range(4):
        [chunk async for chunk in conversation.stream(f"Preference {index}: " + "p" * 80, "alice")]
    await asyncio.wait_for(entered.wait(), 1)

    # Further turns complete while the compaction request is still pending.
    async def answer(text):
        return [chunk async for chunk in conversation.stream(text, "alice")]

    for index in range(4, 7):
        assert await asyncio.wait_for(answer(f"Preference {index}: " + "p" * 80), 1) == ["Hello there."]
    # Nothing is committed yet: every exchange stays stored verbatim, and the
    # prompt keeps the newest ones within the request budget.
    _, pending = conversation._context.dialogue_snapshot("alice")
    assert [entry.user[:12] for entry in pending] == [f"Preference {index}" for index in range(7)]
    prompt = llm.messages[-1][-2].content
    assert len(prompt.encode("utf-8")) <= 400
    assert "Preference 5" in prompt
    assert "SUMMARIES OF OLDER MATERIAL" not in prompt

    release.set()
    await _settle(conversation, "alice")
    summary, recent = conversation._context.dialogue_snapshot("alice")
    assert summary == "The user prefers nanometers."
    assert recent[-1].user.startswith("Preference 6")
    await conversation.close()


@pytest.mark.asyncio
async def test_retired_failing_compaction_does_not_touch_the_replaced_source():
    entered = asyncio.Event()

    async def compact(_source, _previous, _entries, _limit_bytes):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError("compactor failed while being cancelled") from None
        return "never"

    def schedule():
        return context.schedule_context_compaction(
            "alice",
            "notes",
            trigger_budget_bytes=1,
            payload_budget_bytes=1000,
            summary_budget_bytes=100,
            compact=compact,
        )

    context = _ConversationContext()
    await context.update("alice", "notes", "incremental", "old note", context_budget_bytes=1000)
    job = schedule()
    assert job is not None
    await entered.wait()
    # A replacement retires the in-flight job; newer records then arrive.
    await context.update("alice", "notes", "summary", "Replacement", context_budget_bytes=1000)
    await context.update("alice", "notes", "incremental", "newer note one", context_budget_bytes=1000)
    await context.update("alice", "notes", "incremental", "newer note two", context_budget_bytes=1000)
    await asyncio.gather(job, return_exceptions=True)
    await asyncio.sleep(0)

    # The retired job's failure neither evicted the newer records nor scheduled work.
    assert [record.text for record in context._participants["alice"].context_records["notes"]] == [
        "newer note one",
        "newer note two",
    ]
    assert "notes" not in context._participants["alice"].context_jobs
    await context.close()
