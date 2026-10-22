# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import xr_ai_conversation as conversation_package
import xr_ai_conversation._conversation as conversation_module
from xr_ai_conversation import Conversation, ConversationContextUpdate
from xr_ai_conversation._context import (
    _BackgroundSection,
    _ContextRecord,
    _ConversationContext,
    _TruncatedSummary,
)
from xr_ai_models import Capabilities, ChatResponse, ChatStreamEvent, DecisionAnswer, DecisionResponse
from xr_ai_routing import (
    Admission,
    AgentExchange,
    AgentRouter,
    AgentTurn,
    DestinationResult,
    RoutableAgent,
    RoutingAid,
    notify_response_started,
)
from xr_ai_runtime import AgentRuntime, Topic

_NOON_US = 1_790_856_000_000_000
_MINUTE_US = 60_000_000


def test_conversation_explicitly_implements_routable_agent():
    assert Conversation.__bases__ == (RoutableAgent,)
    assert all(callable(Conversation.__dict__.get(method)) for method in ("routing_aid", "admit", "answer", "release"))


def test_conversation_package_exports_only_engine_types():
    assert conversation_package.__all__ == ["Conversation", "ConversationContextUpdate"]


def test_conversation_has_no_tool_or_vision_parameters():
    parameters = inspect.signature(Conversation).parameters
    assert not {"frames", "vision", "tools"} & set(parameters)


class _Decisions:
    async def decide(self, *_args, **_kwargs):
        raise AssertionError("one registered destination does not need route inference")


def _response(content, finish_reason="stop"):
    return ChatResponse(content=content, reasoning=None, tool_calls=None, finish_reason=finish_reason, raw={})


class _LLM:
    capabilities = Capabilities()

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
        return _response("Compacted source facts.")


async def _settle(conversation, participant):
    memory = conversation._context._participants.get(participant)
    while memory is not None and (
        jobs := [job for job in (memory.dialogue_job, *memory.context_jobs.values()) if job is not None]
    ):
        await asyncio.gather(*jobs, return_exceptions=True)
        # Awaiting finished jobs does not yield; let their completion callbacks run.
        await asyncio.sleep(0)


def _schedule_dialogue(context, compact, *, trigger=1, payload=1000, summary=1000):
    return context.schedule_dialogue_compaction(
        "alice",
        trigger_budget_bytes=trigger,
        payload_budget_bytes=payload,
        summary_budget_bytes=summary,
        compact=compact,
    )


def _schedule_source(context, source, compact, *, trigger=1, payload=1000, summary=100):
    return context.schedule_context_compaction(
        "alice",
        source,
        trigger_budget_bytes=trigger,
        payload_budget_bytes=payload,
        summary_budget_bytes=summary,
        compact=compact,
    )


async def _drain(context, participant="alice"):
    memory = context._participants.get(participant)
    while memory is not None and (
        jobs := [job for job in (memory.dialogue_job, *memory.context_jobs.values()) if job is not None]
    ):
        await asyncio.gather(*jobs, return_exceptions=True)
        await asyncio.sleep(0)


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
    assert "User: Hello.\nAssistant: “Hello there.”" in history_message
    assert "max_tokens" not in llm.calls[0]
    assert "temperature" not in llm.calls[0]
    assert "enable_thinking" not in llm.calls[0]


@pytest.mark.asyncio
async def test_conversation_notifies_before_delivery_and_preserves_answer_context(monkeypatch):
    events = []

    class LLM(_LLM):
        def stream_events(self, _messages, **_kwargs):
            async def chunks():
                for text in (" ", "Hello", " there."):
                    yield ChatStreamEvent(kind="text", text=text)
            return chunks()

    def notify():
        events.append(("started", notify_response_started()))

    async def emit(source, text):
        events.append((source, text))

    monkeypatch.setattr(conversation_module, "notify_response_started", notify)
    conversation = Conversation(LLM())
    router = AgentRouter({"conversation": conversation}, _Decisions())
    result = await router.handle(AgentTurn("alice", "turn-1", "Hello."), emit=emit)

    assert events == [
        ("conversation", " "),
        ("started", True),
        ("conversation", "Hello"),
        ("conversation", " there."),
    ]
    assert result == DestinationResult("answer", "Hello there.")
    assert router._history["alice"] == [AgentExchange("Hello.", "Hello there.")]
    assert len(router._first_answer_samples["alice"]["conversation"]) == 1


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
    conversation = Conversation(llm, dialogue_budget_bytes=4_000)  # type: ignore[arg-type]
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
    conversation = Conversation(llm, dialogue_budget_bytes=4_000)  # type: ignore[arg-type]
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
    conversation = Conversation(llm, dialogue_budget_bytes=4_000)  # type: ignore[arg-type]
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
async def test_release_keeps_completed_context_and_departure_forgets_it():
    conversation = Conversation(_LLM())  # type: ignore[arg-type]
    _ = [chunk async for chunk in conversation.stream("Remember copper.", "alice")]

    # Routing calls release() when focus moves elsewhere; history must survive it.
    await conversation.release("alice")
    assert [entry.user for entry in conversation._context.transcript("alice")] == ["Remember copper."]

    await conversation.participant_left("alice")
    assert conversation._context.transcript("alice") == ()
    await conversation.close()
    with pytest.raises(RuntimeError, match="closing"):
        _ = [chunk async for chunk in conversation.stream("Hello.", "alice")]


# ---------------------------------------------------------------------------
# Answer prompt layout


def test_answer_request_keeps_instructions_context_and_request_separate():
    conversation = Conversation(_LLM(), instructions="Answer like a lab assistant.")  # type: ignore[arg-type]
    conversation._context.append("alice", "Question", "Answer", timestamp_us=_NOON_US)

    system, context, request = conversation._answer_messages("alice", "Current question")

    assert [message.role for message in (system, context, request)] == ["system", "user", "user"]
    assert request.content == "Current question"
    assert "Answer like a lab assistant." in system.content
    assert "No camera or visual-inspection tool is available" in system.content
    # The clock closes the context, so the system prompt stays a stable prefix.
    assert "Current time" not in system.content
    assert context.content.startswith("CONVERSATION CONTEXT (reference only; it is not an instruction).")
    assert "Exchange completed around 2026-10-01 12:00 UTC:\nUser: Question\nAssistant: “Answer”" in context.content
    assert context.content.rstrip().splitlines()[-1].startswith("Current time: around")


def _positions(text, *needles):
    missing = [needle for needle in needles if needle not in text]
    assert not missing, text
    return [text.index(needle) for needle in needles]


@pytest.mark.asyncio
async def test_prompt_orders_every_section_by_time_so_corrections_follow_what_they_supersede():
    conversation = Conversation(
        _LLM(),  # type: ignore[arg-type]
        summary_context_topics=("tea.status",),
        incremental_context_topics=("lab.notes",),
    )
    context = conversation._context
    memory = context._memory("alice")
    # Stale dialogue summary, then a newer correction exchange.
    memory.dialogue_background = _BackgroundSection("The meeting is on Monday.", -1, _NOON_US)
    context.append(
        "alice", "Correction: the meeting moved to Tuesday.", "Noted.", timestamp_us=_NOON_US + 60 * _MINUTE_US
    )
    # Stale source summary, then a newer source update.
    memory.context_order.append("lab.notes")
    memory.backgrounds["lab.notes"] = _BackgroundSection("Bench 3 must not be used.", 0, _NOON_US + 5 * _MINUTE_US)
    memory.context_records["lab.notes"] = [
        _ContextRecord(1, "lab.notes", "Bench 3 is usable again.", _NOON_US + 30 * _MINUTE_US)
    ]
    await context.update("alice", "tea.status", "summary", "Timer is running.", timestamp_us=_NOON_US + 45 * _MINUTE_US)

    rendered = conversation._answer_messages("alice", "When is the meeting?")[1].content

    positions = _positions(
        rendered,
        "Summary of the earlier conversation with this user, up to around 2026-10-01 12:00 UTC:\n"
        "The meeting is on Monday.",
        "Source lab.notes summary of earlier updates, up to around 2026-10-01 12:05 UTC:\nBench 3 must not be used.",
        "Source lab.notes update, published around 2026-10-01 12:30 UTC:\nBench 3 is usable again.",
        "Source tea.status current state, published around 2026-10-01 12:45 UTC:\nTimer is running.",
        "Exchange completed around 2026-10-01 13:00 UTC:\nUser: Correction: the meeting moved to Tuesday.",
        "Current time: around",
    )
    assert positions == sorted(positions)
    await conversation.close()


@pytest.mark.asyncio
async def test_summary_renders_first_with_its_time_when_another_section_has_older_raw_updates():
    """A summary's span can overlap another section's raw update; both show their times for the model."""

    conversation = Conversation(_LLM(), incremental_context_topics=("lab.notes",))  # type: ignore[arg-type]
    context = conversation._context
    memory = context._memory("alice")
    memory.context_order.append("lab.notes")
    memory.context_records["lab.notes"] = [
        _ContextRecord(0, "lab.notes", "Reference lamp in use is HL-1000-12.", _NOON_US)
    ]
    # The 12:40 lamp swap has been compacted into the dialogue summary.
    memory.dialogue_background = _BackgroundSection(
        "The user swapped lamps: use HL-2000-77 from now on.", 0, _NOON_US + 40 * _MINUTE_US
    )

    rendered = conversation._answer_messages("alice", "Which lamp should I use?")[1].content

    summary, update = _positions(
        rendered,
        "up to around 2026-10-01 12:40 UTC:\nThe user swapped lamps",
        "published around 2026-10-01 12:00 UTC:\nReference lamp in use is HL-1000-12.",
    )
    assert summary < update
    await conversation.close()


@pytest.mark.asyncio
async def test_each_section_is_bounded_so_a_busy_topic_cannot_evict_another():
    """Review finding: a topic over its budget while compaction was pending removed another topic's state."""

    release = asyncio.Event()

    class _StuckCompactionLLM(_LLM):
        async def chat(self, messages, **_kwargs):
            await release.wait()
            return _response("Notes summary.")

    llm = _StuckCompactionLLM()
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        summary_context_topics=("instrument.status",),
        incremental_context_topics=("notes",),
        topic_budget_bytes={"instrument.status": 400, "notes": 1000},
    )
    await conversation._update_context(
        "alice", "instrument.status", ConversationContextUpdate(text="Spectrometer ready."), mode="summary"
    )
    for index in range(10):
        await conversation._update_context(
            "alice", "notes", ConversationContextUpdate(text=f"note {index} " + "n" * 200), mode="incremental"
        )

    rendered = conversation._answer_messages("alice", "Status?")[1].content

    notes = [line for line in rendered.split("\n\n") if line.startswith("Source notes")]
    assert sum(len(block.encode("utf-8")) for block in notes) <= 1000
    assert "note 9 " in rendered  # the newest update is kept
    assert "Spectrometer ready." in rendered
    # Omitted updates stay stored until compaction folds them in.
    assert len(conversation._context._participants["alice"].context_records["notes"]) == 10
    release.set()
    await _settle(conversation, "alice")
    await conversation.close()


@pytest.mark.asyncio
async def test_oversized_newest_exchange_is_clipped_in_the_prompt_instead_of_wiping_context():
    llm = _LLM()
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        incremental_context_topics=("notes",),
        dialogue_budget_bytes=1500,
        topic_budget_bytes=1500,
    )
    await conversation._update_context(
        "alice", "notes", ConversationContextUpdate(text="Bench 3 is usable."), mode="incremental"
    )
    pasted_log = "Controller log:\n" + "heartbeat ok\n" * 1500 + "ERROR interlock tripped code E-4471"
    _ = [chunk async for chunk in conversation.stream(pasted_log, "alice")]
    await _settle(conversation, "alice")
    _ = [chunk async for chunk in conversation.stream("What was the error code?", "alice")]

    context = llm.messages[-1][-2].content
    assert len(context.encode("utf-8")) <= 3000 + 300
    assert "Controller log:" in context
    assert "E-4471" in context
    assert "bytes omitted" in context
    assert "Bench 3 is usable." in context
    await conversation.close()


@pytest.mark.asyncio
async def test_oversized_uncompacted_update_is_clipped_in_the_prompt_instead_of_dropped():
    llm = _LLM()
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        incremental_context_topics=("notes",),
        topic_budget_bytes=1500,
    )
    await conversation._update_context(
        "alice", "notes", ConversationContextUpdate(text="Bench 3 is usable."), mode="incremental"
    )
    # Store a large raw update without letting compaction run before the next prompt.
    await conversation._context.update("alice", "notes", "incremental", "dump " * 2000 + "Purge started.")
    _ = [chunk async for chunk in conversation.stream("Status?", "alice")]

    context = llm.messages[-1][-2].content
    assert "Purge started." in context
    assert "bytes omitted" in context
    await conversation.close()


@pytest.mark.asyncio
async def test_oversized_summary_topic_state_is_clipped_to_its_budget(caplog):
    conversation = Conversation(
        _LLM(),  # type: ignore[arg-type]
        summary_context_topics=("status",),
        topic_budget_bytes=200,
    )
    text = "Status begins. " + "detail " * 200 + "Status ends."
    await conversation._update_context("alice", "status", ConversationContextUpdate(text=text), mode="summary")

    ((source, stored),) = conversation._context.sections("alice")
    assert source == "status"
    assert stored.startswith("Status begins.") and stored.endswith("Status ends.")
    assert len(f"Source status current state, published at an unknown time:\n{stored}".encode("utf-8")) <= 200
    assert "clipping oversized context state" in caplog.text
    await conversation.close()


# ---------------------------------------------------------------------------
# Dialogue compaction


@pytest.mark.asyncio
async def test_private_dialogue_compaction_replaces_old_prefix_and_does_not_replay_it():
    context = _ConversationContext()
    window = (
        ("The label says copper.", "I will remember copper."),
        ("Keep the facilities check open.", "I will keep it open."),
    )
    context.reconcile_transcript("alice", window)
    context.reconcile_transcript("alice", window)
    assert len(context.transcript("alice")) == 2

    async def compact(previous, entries, limit_bytes):
        assert previous == ""
        assert limit_bytes == 1000
        assert [(entry.user, entry.assistant) for entry in entries] == [window[0]]
        return "The user identified a copper label."

    task = _schedule_dialogue(context, compact)
    assert task is not None
    assert await task is True
    summary, recent = context.dialogue_snapshot("alice")
    assert summary.text == "The user identified a copper label."
    assert [(entry.user, entry.assistant) for entry in recent] == [window[1]]
    # A front-end window that still contains the compacted exchange is not replayed.
    context.reconcile_transcript("alice", window)
    assert [entry.user for entry in context.transcript("alice")] == ["Keep the facilities check open."]


def test_private_transcript_reconciles_overlapping_windows_and_keeps_timestamp_provenance():
    context = _ConversationContext()
    context.reconcile_transcript("alice", (("First request", "First reply", 10), ("Scene request", "Scene reply", 20)))

    # The front end may send a recent suffix with one new completed exchange.
    context.reconcile_transcript(
        "alice", (("Scene request", "Scene reply", 20), ("Latest request", "Latest reply", None))
    )
    # A repeated utterance/result at a distinct known time is new, while
    # replaying that exact timestamped window must not duplicate it.
    context.reconcile_transcript("alice", (("Scene request", "Scene reply", 2_000_000),))
    context.reconcile_transcript("alice", (("Scene request", "Scene reply", 2_000_000),))

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
    unknown_seed.reconcile_transcript("alice", (("Scene request", "Scene reply", 30),), exchange_ids=(object(),))
    assert len(unknown_seed.transcript("alice")) == 2


@pytest.mark.asyncio
async def test_router_history_windows_larger_than_256_exchanges_are_not_replayed():
    """Review finding: a window longer than the remembered identities replayed completed dialogue."""

    llm = _LLM()
    conversation = Conversation(llm)  # type: ignore[arg-type]
    front_end = AgentRouter(
        {"conversation": conversation},
        _Decisions(),  # type: ignore[arg-type]
        context_budget_bytes=64_000,
    )

    async def emit(_source, _text):
        pass

    for index in range(300):
        await front_end.handle(AgentTurn("alice", f"turn-{index}", f"Turn {index}."), emit=emit)
        await _settle(conversation, "alice")

    assert len(front_end._history["alice"]) > 256
    assert len(conversation._context._participants["alice"].seen) == 300
    last_context = llm.messages[-1][1].content
    assert last_context.count("User: Turn 298.") == 1
    await conversation.close()


@pytest.mark.asyncio
async def test_conversation_uses_dialogue_summary_and_keeps_newer_exchanges_verbatim():
    class _DialogueLLM(_LLM):
        async def chat(self, messages, **_kwargs):
            assert "completed conversation history" in messages[0].content
            assert "at most 175 bytes" in messages[0].content
            content = messages[1].content
            if "User: Check the facilities later." in content:
                assert "The user identified a copper label." in content
                return _response("The user identified a copper label and left the facilities check open.")
            assert "User: The label is copper." in content
            return _response("The user identified a copper label.")

    llm = _DialogueLLM()
    conversation = Conversation(llm, dialogue_budget_bytes=700)  # type: ignore[arg-type]
    old_history = (
        AgentExchange("The label is copper." + " x" * 160, "The label was observed." + " y" * 160),
        AgentExchange("Check the facilities later." + " z" * 160, "That request remains open." + " w" * 160),
    )
    assert [chunk async for chunk in conversation.stream("What is the label?", "alice", history=old_history)] == [
        "Hello there."
    ]
    await _settle(conversation, "alice")
    assert conversation._context.dialogue_snapshot("alice")[0].text == (
        "The user identified a copper label and left the facilities check open."
    )

    current = AgentExchange("What is the label?", "Hello there.")
    _ = [chunk async for chunk in conversation.stream("Remember?", "alice", history=old_history + (current,))]
    prompt_context = llm.messages[-1][-2].content
    assert "The user identified a copper label and left the facilities check open." in prompt_context
    assert "User: What is the label?\nAssistant: “Hello there.”" in prompt_context
    assert "The label is copper." not in prompt_context
    await conversation.close()


def test_dialogue_trigger_is_half_the_section_budget_measured_as_rendered():
    conversation = Conversation(_LLM(), dialogue_budget_bytes=2000)  # type: ignore[arg-type]
    limits = conversation._compaction_limits(2000)
    assert limits["trigger_budget_bytes"] == 1000
    assert limits["summary_budget_bytes"] == 500

    context = _ConversationContext()
    context.append("alice", "ok", "Sure, done.", timestamp_us=_NOON_US)
    entry = context.transcript("alice")[0]
    # Short turns still carry their prompt framing.
    rendered = "Exchange completed around 2026-10-01 12:00 UTC:\nUser: ok\nAssistant: “Sure, done.”"
    assert entry.utf8_size == len(rendered.encode("utf-8"))


@pytest.mark.asyncio
async def test_short_turns_compact_before_the_prompt_omits_any_of_them(caplog):
    """Review finding: stored-size triggers let many short turns overflow the prompt before compaction."""

    conversation = Conversation(_LLM())  # type: ignore[arg-type]
    for index in range(200):
        _ = [chunk async for chunk in conversation.stream(f"ok {index}", "alice")]
        await _settle(conversation, "alice")
    assert "omitting uncompacted context" not in caplog.text
    assert conversation._context.dialogue_snapshot("alice")[0].text == "Compacted source facts."
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
        task = _schedule_dialogue(context, compact, trigger=600, payload=900, summary=100)
        while task is not None:
            await task
            await asyncio.sleep(0)
            task = context._participants["alice"].dialogue_job

    assert requests and max(requests) <= 900
    summary, recent = context.dialogue_snapshot("alice")
    assert summary.text == f"summary {len(requests)}"
    assert recent[-1].user == "preference 21"
    assert all("x" * 1000 not in entry.user for entry in recent)


@pytest.mark.asyncio
async def test_oversized_summary_is_condensed_then_clipped_keeping_its_newest_end(caplog):
    calls = []

    async def compact(previous, entries, limit_bytes):
        calls.append(len(entries))
        if entries:
            return "old facts " * 60 + "NEWEST-CORRECTION"
        return "condensed " * 60 + "NEWEST-CORRECTION"

    context = _ConversationContext()
    context.append("alice", "a" * 300, "ok")
    context.append("alice", "b" * 300, "ok")
    await _schedule_dialogue(context, compact, summary=256)

    # The merge overshoots, one condense request also overshoots, and the
    # result keeps its start and its newest end.
    assert calls == [1, 0]
    summary = context.dialogue_snapshot("alice")[0].text
    assert len(summary.encode("utf-8")) <= 256
    assert summary.startswith("condensed") and summary.endswith("NEWEST-CORRECTION")
    assert "condensing oversized context summary" in caplog.text

    async def condense_cut_off(previous, entries, _limit_bytes):
        if entries:
            return "merged " * 60 + "NEWEST"
        raise _TruncatedSummary("cut off")

    context.append("alice", "c" * 300, "ok")
    await _schedule_dialogue(context, condense_cut_off, summary=256)
    summary = context.dialogue_snapshot("alice")[0].text
    assert summary.startswith("merged") and summary.endswith("NEWEST")


@pytest.mark.asyncio
async def test_compaction_cut_off_at_the_token_limit_keeps_the_exchanges(caplog):
    """Review finding: a length-cut partial summary was committed and its exchanges discarded."""

    class _CutOffLLM(_LLM):
        async def chat(self, messages, **_kwargs):
            return _response("User's name is Al. User prefers tea. Cor", finish_reason="length")

    conversation = Conversation(_CutOffLLM(), dialogue_budget_bytes=400)  # type: ignore[arg-type]
    for index in range(6):
        _ = [chunk async for chunk in conversation.stream(f"Fact {index}: " + "f" * 60, "alice")]
        await _settle(conversation, "alice")

    summary, recent = conversation._context.dialogue_snapshot("alice")
    assert summary.text == ""
    assert len(recent) == 6
    assert "cut off at the output token limit" in caplog.text
    await conversation.close()


@pytest.mark.asyncio
async def test_failed_or_empty_compaction_keeps_exchanges_and_retries_later(caplog):
    results = iter(["", RuntimeError("model unavailable"), "recovered summary"])

    async def compact(_previous, _entries, _limit_bytes):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    context = _ConversationContext()

    async def turn(index):
        context.append("alice", f"question {index} " + "q" * 40, "a" * 40)
        task = _schedule_dialogue(context, compact, trigger=100, summary=50)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)

    await turn(0)
    await turn(1)  # empty summary: rejected, exchanges kept
    await turn(2)  # model error: exchanges kept
    assert caplog.text.count("conversation context compaction failed") == 2
    assert len(context.transcript("alice")) == 3
    await turn(3)  # retried on the next exchange
    assert context.dialogue_snapshot("alice")[0].text == "recovered summary"
    assert [entry.user[:10] for entry in context.transcript("alice")] == ["question 3"]


@pytest.mark.asyncio
async def test_dialogue_compaction_keeps_appended_tail_and_departure_cancels_it():
    first_entered = asyncio.Event()
    finish_first = asyncio.Event()
    second_entered = asyncio.Event()
    calls = 0

    async def compact(previous, entries, _limit_bytes):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            await finish_first.wait()
            assert [entry.user for entry in entries] == ["Old question"]
            return "Old question was answered."
        second_entered.set()
        assert "Old question was answered." in previous
        await asyncio.Event().wait()
        return "never completes"

    context = _ConversationContext()
    context.append("alice", "Old question", "Old answer")
    context.append("alice", "Middle question", "Middle answer")
    task = _schedule_dialogue(context, compact, payload=150)
    assert task is not None
    await first_entered.wait()
    context.append("alice", "New question", "New answer")
    finish_first.set()
    assert await task is True
    await second_entered.wait()
    summary, recent = context.dialogue_snapshot("alice")
    assert summary.text == "Old question was answered."
    assert [entry.user for entry in recent] == ["Middle question", "New question"]
    running = context._participants["alice"].dialogue_job
    await context.clear("alice")
    assert running.cancelled()
    assert context.transcript("alice") == ()


@pytest.mark.asyncio
async def test_answers_stay_responsive_while_compaction_is_stuck_and_commit_later():
    release = asyncio.Event()
    entered = asyncio.Event()

    class _SlowCompactionLLM(_LLM):
        async def chat(self, messages, **_kwargs):
            entered.set()
            await release.wait()
            return _response("The user prefers nanometers.")

    llm = _SlowCompactionLLM()
    conversation = Conversation(llm, dialogue_budget_bytes=600)  # type: ignore[arg-type]

    async def answer(text):
        return [chunk async for chunk in conversation.stream(text, "alice")]

    for index in range(4):
        await answer(f"Preference {index}: " + "p" * 80)
    await asyncio.wait_for(entered.wait(), 1)
    for index in range(4, 7):
        assert await asyncio.wait_for(answer(f"Preference {index}: " + "p" * 80), 1) == ["Hello there."]
    _, pending = conversation._context.dialogue_snapshot("alice")
    assert [entry.user[:12] for entry in pending] == [f"Preference {index}" for index in range(7)]
    assert "Preference 5" in llm.messages[-1][-2].content

    release.set()
    await _settle(conversation, "alice")
    summary, recent = conversation._context.dialogue_snapshot("alice")
    assert summary.text == "The user prefers nanometers."
    assert recent[-1].user.startswith("Preference 6")
    await conversation.close()


# ---------------------------------------------------------------------------
# Named topics


@pytest.mark.asyncio
async def test_runtime_named_context_topics_use_configured_modes_per_participant():
    llm = _LLM()
    tea = Topic("tea.status", ConversationContextUpdate)
    xr = Topic("xr.observations", ConversationContextUpdate)
    conversation = Conversation(
        llm,  # type: ignore[arg-type]
        summary_context_topics=(tea.name,),
        incremental_context_topics=(xr.name,),
        topic_budget_bytes=200,
    )
    runtime = AgentRuntime()
    runtime.register("conversation-context", conversation.context_subscriber())

    async with runtime:
        await runtime.publish(tea, {"text": "Tea timer is at 4 minutes."}, participant_id="alice")
        await runtime.publish(tea, {"text": "Timer is now at 6 minutes."}, participant_id="alice")
        await runtime.publish(xr, {"text": "The exhibit has three panels."}, participant_id="alice")
        await runtime.publish(xr, {"text": "One panel is interactive."}, participant_id="alice")
        await runtime.publish(tea, {"text": "A different timer is at 2 minutes."}, participant_id="bob")
        await _settle(conversation, "alice")

    assert conversation._context.sections("alice") == (
        ("tea.status", "Timer is now at 6 minutes."),
        ("xr.observations", "Compacted source facts."),
    )
    assert conversation._context.sections("bob") == (("tea.status", "A different timer is at 2 minutes."),)
    await conversation.close()


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


def test_each_section_budget_is_validated():
    llm = _LLM()
    with pytest.raises(ValueError, match="one entry per configured topic"):
        Conversation(llm, incremental_context_topics=("a", "b"), topic_budget_bytes={"a": 100})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integers"):
        Conversation(llm, incremental_context_topics=("a",), topic_budget_bytes={"a": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integers"):
        Conversation(llm, dialogue_budget_bytes=0)  # type: ignore[arg-type]
    # A compaction request must hold a summary plus at least as many bytes of new entries.
    with pytest.raises(ValueError, match="compaction_input_budget_bytes"):
        Conversation(llm, dialogue_budget_bytes=8000, compaction_input_budget_bytes=4255)  # type: ignore[arg-type]
    Conversation(llm, dialogue_budget_bytes=8000, compaction_input_budget_bytes=4256)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_each_topic_compacts_against_its_own_budget():
    conversation = Conversation(
        _LLM(),  # type: ignore[arg-type]
        incremental_context_topics=("small", "large"),
        topic_budget_bytes={"small": 200, "large": 4000},
    )
    note = "a fact worth keeping " * 6
    for source in ("small", "large"):
        await conversation._update_context("alice", source, ConversationContextUpdate(text=note), mode="incremental")
    await _settle(conversation, "alice")

    # Only the small topic passed half of its own budget and was compacted.
    assert conversation._context.sections("alice") == (("small", "Compacted source facts."), ("large", note))
    await conversation.close()


@pytest.mark.asyncio
async def test_source_summary_times_follow_compaction():
    summary_calls = []

    async def compact(source, previous, records, _limit_bytes):
        summary_calls.append((source, previous, records[0].timestamp_us))
        return "Timer was started."

    context = _ConversationContext()
    await context.update("alice", "tea.status", "incremental", "Timer started.", timestamp_us=_NOON_US)
    await _schedule_source(context, "tea.status", compact)
    assert summary_calls == [("tea.status", "", _NOON_US)]
    snapshot = context.source_snapshots("alice")[0]
    assert snapshot[1].text == "Timer was started." and snapshot[1].latest_us == _NOON_US

    await context.update(
        "alice", "tea.status", "incremental", "Timer extended.", timestamp_us=_NOON_US + 60 * _MINUTE_US
    )
    await _schedule_source(context, "tea.status", compact)
    assert summary_calls[1][1] == "Earlier summary (up to around 2026-10-01 12:00 UTC):\nTimer was started."
    assert context.source_snapshots("alice")[0][1].latest_us == _NOON_US + 60 * _MINUTE_US


@pytest.mark.asyncio
async def test_source_summary_near_budget_does_not_block_later_updates():
    """Review findings: an oversized or near-budget compacted summary used to discard later updates."""

    calls = []

    class _SummaryLLM(_LLM):
        async def chat(self, messages, **_kwargs):
            calls.append(messages[0].content)
            # The first merge overshoots and the condense request still does.
            return _response("s" * 800 if len(calls) <= 2 else "Latest corrections applied.")

    conversation = Conversation(
        _SummaryLLM(),  # type: ignore[arg-type]
        incremental_context_topics=("notes",),
        topic_budget_bytes=1024,
    )

    async def publish(text):
        await conversation._update_context("alice", "notes", ConversationContextUpdate(text=text), mode="incremental")
        await _settle(conversation, "alice")

    await publish("n" * 600)
    ((_, stored),) = conversation._context.sections("alice")
    assert len(stored.encode("utf-8")) <= 256
    assert "at most 256 bytes" in calls[0]
    assert calls[1].startswith("Shorten this summary to at most 256 bytes")

    for index in range(3):
        await publish(f"correction {index} " + "c" * 290)
    # The first two corrections are merged; the newest stays verbatim below the trigger.
    assert len(calls) == 3
    assert conversation._context.sections("alice") == (
        ("notes", "Latest corrections applied.\ncorrection 2 " + "c" * 290),
    )
    await conversation.close()


@pytest.mark.asyncio
async def test_incremental_update_larger_than_any_budget_is_clipped_and_compacted():
    requests = []

    async def compact(source, previous, records, _limit_bytes):
        requests.append(len(previous.encode("utf-8")) + sum(record.utf8_size for record in records))
        return f"{source} summary {len(requests)}"

    context = _ConversationContext()
    await context.update("alice", "log", "incremental", "line\n" * 5000)
    for index in range(5):
        await context.update("alice", "log", "incremental", f"event {index}")
    task = _schedule_source(context, "log", compact, trigger=50, payload=300, summary=25)
    while task is not None:
        await task
        await asyncio.sleep(0)
        task = context._participants["alice"].context_jobs.get("log")

    assert requests and max(requests) <= 300
    assert context.sections("alice") == (("log", f"log summary {len(requests)}"),)


@pytest.mark.asyncio
async def test_incremental_update_during_compaction_keeps_it_for_the_next_batch():
    entered = asyncio.Event()
    finish_first = asyncio.Event()
    calls = []

    async def compact(_source, previous, records, _limit_bytes):
        calls.append((previous, [record.text for record in records]))
        if len(calls) == 1:
            entered.set()
            await finish_first.wait()
            return "summary of first fact"
        return "summary of both facts"

    context = _ConversationContext()
    await context.update("alice", "tea", "incremental", "first fact")
    first = _schedule_source(context, "tea", compact)
    assert first is not None
    await entered.wait()
    await context.update("alice", "tea", "incremental", "second fact")
    assert _schedule_source(context, "tea", compact) is first
    finish_first.set()
    await first
    await _drain(context)

    assert calls[0] == ("", ["first fact"])
    assert "summary of first fact" in calls[1][0] and calls[1][1] == ["second fact"]
    assert context.sections("alice") == (("tea", "summary of both facts"),)


@pytest.mark.asyncio
async def test_context_update_without_participant_is_dropped(caplog):
    conversation = Conversation(_LLM(), summary_context_topics=("tea",))  # type: ignore[arg-type]
    await conversation._update_context(None, "tea", ConversationContextUpdate(text="shared"), mode="summary")
    assert conversation._context.sections("alice") == ()
    assert "without participant" in caplog.text
    await conversation.close()


# ---------------------------------------------------------------------------
# Lifecycle


@pytest.mark.asyncio
async def test_departure_and_close_cancel_compaction_and_keep_participants_independent(caplog):
    entered = asyncio.Event()

    class _BlockingLLM(_LLM):
        async def chat(self, messages, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

    topic = Topic("test.notes", ConversationContextUpdate)
    conversation = Conversation(
        _BlockingLLM(),  # type: ignore[arg-type]
        incremental_context_topics=(topic.name,),
        topic_budget_bytes=100,
    )
    runtime = AgentRuntime()
    runtime.register("context", conversation.context_subscriber())
    async with runtime:
        await runtime.publish(topic, ConversationContextUpdate(text="A note " * 10), participant_id="alice")
        await entered.wait()
        job = conversation._context._participants["alice"].context_jobs[topic.name]

        await conversation.participant_left("alice")
        assert job.cancelled()
        assert conversation._context.sections("alice") == ()
        # Another participant is unaffected.
        assert [text async for text in conversation.stream("Hello", "bob")] == ["Hello there."]

        await runtime.publish(topic, ConversationContextUpdate(text="A note " * 10), participant_id="carol")
        await conversation.close()
        assert conversation._context._participants == {}
    assert "compaction failed" not in caplog.text


# ---------------------------------------------------------------------------
# Eval case files


def _eval_common():
    path = Path(__file__).resolve().parents[1] / "agent-sdk" / "xr-ai-conversation" / "eval" / "_common.py"
    spec = importlib.util.spec_from_file_location("conversation_eval_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_case_files_are_well_formed_and_graders_reject_superseded_answers():
    common = _eval_common()
    for name in common.CASE_FILES:
        suite = common.load_cases(name)
        common.validate_suite(suite)
        for case in common.graded_cases(suite):
            for answer in case.get("accept", ()):
                assert common.matches(answer, case), (name, case["name"], answer)
            for answer in case.get("reject", ()):
                assert not common.matches(answer, case), (name, case["name"], answer)

    # Whole-word grading: "yes" must not match "yesterday".
    assert not common.matches("Yesterday", {"any_of": ["yes"]})
    assert common.matches("Yes.", {"any_of": ["yes"]})


@pytest.mark.asyncio
async def test_cancelled_answer_closes_the_model_stream_and_records_nothing():
    """The router cancels answer() on barge-in; the model stream must close and nothing be recorded."""

    first_chunk = asyncio.Event()
    stream_closed = asyncio.Event()

    class _BlockingStreamLLM(_LLM):
        def stream_events(self, messages, **_kwargs):
            async def events():
                try:
                    yield ChatStreamEvent(kind="text", text="Partial answer")
                    await asyncio.Event().wait()
                finally:
                    stream_closed.set()

            # Holding the stream, as a client could, means only an explicit close runs its finally.
            self.stream = events()
            return self.stream

    conversation = Conversation(_BlockingStreamLLM())  # type: ignore[arg-type]

    async def emit(_source, _text):
        # Barge-in arrives while the first chunk is being delivered, with the model stream paused at it.
        first_chunk.set()
        await asyncio.Event().wait()

    answering = asyncio.create_task(conversation.answer(AgentTurn("alice", "turn-1", "Tell me a story."), emit=emit))
    await asyncio.wait_for(first_chunk.wait(), 1)
    answering.cancel()
    with pytest.raises(asyncio.CancelledError):
        await answering

    assert stream_closed.is_set()
    assert conversation._context.transcript("alice") == ()
    await conversation.close()


def _conflict_eval():
    eval_dir = Path(__file__).resolve().parents[1] / "agent-sdk" / "xr-ai-conversation" / "eval"
    sys.path.insert(0, str(eval_dir))
    try:
        import conflict_eval
    finally:
        sys.path.remove(str(eval_dir))
    return conflict_eval


@pytest.mark.asyncio
async def test_every_conflict_case_seeds_the_shape_it_claims():
    """A budget change must not leave a conflict case seeding no summary and testing nothing."""

    conflict_eval = _conflict_eval()
    suite = conflict_eval.load_cases("conflicts.yaml")
    for case in suite["cases"]:
        clock = conflict_eval._Clock()
        with patch("time.time_ns", clock):
            conversation, runtime = await conflict_eval._seed(
                conflict_eval.ScriptedLLM(_LLM()), case, clock, suite["step_minutes"]
            )
        try:
            assert conflict_eval._seeding_errors(conversation, case.get("expect", {})) == [], case["name"]
        finally:
            await conversation.close()
            await runtime.stop()
