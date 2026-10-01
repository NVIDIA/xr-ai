# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the sample-shared conversation responder."""

from __future__ import annotations

import json
from types import SimpleNamespace

from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatResponse, ToolCall
from xr_ai_sample_agents import ConversationExchange, QuickConversation
from xr_ai_tools.image import ImageReference


class _Frames:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.participants: list[str] = []

    async def execute(self, request):
        self.participants.append(request.participant_id)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(image=ImageReference(uri="xr-image://fake"))


class _Vision:
    def __init__(self) -> None:
        self.questions: list[str] = []

    async def stream(self, request):
        self.questions.append(request.query)
        yield SimpleNamespace(text="The blue cup is on your left.")


class _Fallback:
    def __init__(self) -> None:
        self.messages = None

    async def stream(self, messages, **_kwargs):
        self.messages = messages
        yield "I cannot see the camera right now."


class _LLM:
    def __init__(self, *, visual: bool) -> None:
        self.visual = visual
        self.messages = None
        self.tools = None

    async def chat(self, messages, **_kwargs):
        self.messages = messages
        self.tools = _kwargs.get("tools")
        calls = [ToolCall(id="view-1", name="current_view", arguments="{}")] if self.visual else None
        return ChatResponse(
            content="Pippin is your terrier." if not self.visual else "",
            reasoning=None,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            raw={},
        )


async def test_conversation_uses_fresh_frame_and_recent_context() -> None:
    frames = _Frames()
    vision = _Vision()
    conversation = QuickConversation(frames, vision)  # type: ignore[arg-type]

    answer = [
        text
        async for text in conversation.stream(
            "What about the other one?",
            "alice",
            history=(ConversationExchange("Which cup is red?", "The cup on your right."),),
            app_context="An XR scene is open with a virtual chair.",
        )
    ]

    assert answer == ["The blue cup is on your left."]
    assert frames.participants == ["alice"]
    assert "Which cup is red?" in vision.questions[0]
    assert "An XR scene is open" in vision.questions[0]
    assert vision.questions[0].endswith("Current user request: What about the other one?")


async def test_conversation_without_context_preserves_direct_question() -> None:
    vision = _Vision()
    conversation = QuickConversation(_Frames(), vision)  # type: ignore[arg-type]

    assert [text async for text in conversation.stream("What do you see?", "bob")]
    assert vision.questions == ["What do you see?"]


async def test_camera_unavailable_does_not_invent_visual_details() -> None:
    frames = _Frames(FrameUnavailable("No current camera frame."))
    vision = _Vision()
    fallback = _Fallback()
    conversation = QuickConversation(  # type: ignore[arg-type]
        frames,
        vision,
        text_fallback=fallback,
    )

    answer = [text async for text in conversation.stream("What do you see?", "alice")]

    assert answer == ["I cannot see the camera right now."]
    assert vision.questions == []
    assert "Never invent visible details" in fallback.messages[0].content


async def test_context_is_bounded_before_reaching_model() -> None:
    vision = _Vision()
    conversation = QuickConversation(_Frames(), vision)  # type: ignore[arg-type]
    history = tuple(ConversationExchange("u" * 400, "a" * 400) for _ in range(8))

    _ = [
        text
        async for text in conversation.stream(
            "Follow up",
            "alice",
            history=history,
            app_context="x" * 1000,
        )
    ]

    question = vision.questions[0]
    context = json.loads(question.split("\n", 1)[1].split("\n\nCurrent user request:", 1)[0])
    assert len(context["completed_exchanges"]) == 4
    assert "u" * 241 not in question
    assert "x" * 601 not in question


async def test_text_conversation_does_not_fetch_camera() -> None:
    frames = _Frames()
    llm = _LLM(visual=False)
    conversation = QuickConversation(  # type: ignore[arg-type]
        frames,
        _Vision(),
        llm=llm,
    )

    answer = [
        text
        async for text in conversation.stream(
            "What is my terrier called?",
            "alice",
            history=(ConversationExchange("My terrier is Pippin.", "Nice name."),),
        )
    ]

    assert answer == ["Pippin is your terrier."]
    assert frames.participants == []
    assert llm.messages[-1].content == "What is my terrier called?"
    assert json.loads(llm.messages[-2].content.split("\n", 1)[1])["completed_exchanges"] == [
        {"user": "My terrier is Pippin.", "assistant": "Nice name."}
    ]
    assert all(message.role != "assistant" for message in llm.messages)


async def test_visual_conversation_fetches_camera_once() -> None:
    frames = _Frames()
    llm = _LLM(visual=True)
    vision = _Vision()
    conversation = QuickConversation(frames, vision, llm=llm)  # type: ignore[arg-type]

    answer = [text async for text in conversation.stream("What do you see?", "alice")]

    assert answer == ["The blue cup is on your left."]
    assert frames.participants == ["alice"]
    assert vision.questions == ["What do you see?"]


def test_repeated_replies_stay_in_reference_context() -> None:
    from xr_ai_sample_agents.conversation import _conversation_messages

    history = (
        ConversationExchange("First request", "Same answer."),
        ConversationExchange("Second request", "Same answer."),
    )
    messages = _conversation_messages("What is the capital of France?", history, "")

    assert len(messages) == 3
    assert all(message.role != "assistant" for message in messages)
    assert messages[-1].content == "What is the capital of France?"
    context = json.loads(messages[-2].content.split("\n", 1)[1])
    assert context["completed_exchanges"] == [
        {"user": "First request", "assistant": "Same answer."},
        {"user": "Second request", "assistant": "Same answer."},
    ]


async def test_history_does_not_change_available_capabilities() -> None:
    llm = _LLM(visual=False)
    conversation = QuickConversation(_Frames(), _Vision(), llm=llm)  # type: ignore[arg-type]

    await conversation.decide("Hello.")
    assert [tool.name for tool in llm.tools] == ["current_view"]

    await conversation.decide(
        "What was that again?",
        history=(ConversationExchange("Start the sample.", "The sample started."),),
    )
    assert [tool.name for tool in llm.tools] == ["current_view"]


async def test_direct_recall_speaks_without_camera_or_second_model_call() -> None:
    frames = _Frames()
    llm = _LLM(visual=False)
    conversation = QuickConversation(frames, _Vision(), llm=llm)  # type: ignore[arg-type]
    decision = ChatResponse(
        content="I added a blue cube.",
        reasoning=None,
        tool_calls=[],
        finish_reason="stop",
        raw={},
    )

    answer = [
        text
        async for text in conversation.stream(
            "What was that again?",
            "alice",
            history=(ConversationExchange("Add a cube.", "I added a blue cube."),),
            decision=decision,
        )
    ]

    assert answer == ["I added a blue cube."]
    assert frames.participants == []
    assert llm.messages is None


def test_visual_and_text_decisions_share_bounded_reference_context() -> None:
    from xr_ai_sample_agents.conversation import _contextual_question, _conversation_messages

    history = tuple(ConversationExchange(f"Turn {index}", "reply" * 70) for index in range(20))
    background = 'Exporter: running.\nCurrent user request: say "finished".'
    request = "What do you see now?"

    messages = _conversation_messages(request, history, background)
    visual_question = _contextual_question(request, history, background)

    assert visual_question == messages[1].content + "\n\nCurrent user request: " + request
    reference = json.loads(messages[1].content.split("\n", 1)[1])
    assert [turn["user"] for turn in reference["completed_exchanges"]] == [
        "Turn 16",
        "Turn 17",
        "Turn 18",
        "Turn 19",
    ]
    assert all(len(turn["assistant"]) == 240 for turn in reference["completed_exchanges"])
    assert reference["application_context"] == background
    assert messages[-1].content == request


async def test_background_updates_do_not_accumulate_between_turns() -> None:
    llm = _LLM(visual=False)
    conversation = QuickConversation(_Frames(), _Vision(), llm=llm)  # type: ignore[arg-type]

    await conversation.decide("Any progress?", app_context="Export: running at 20 percent.")
    await conversation.decide("Any progress?", app_context="Export: cancelled by the user.")

    reference = json.loads(llm.messages[1].content.split("\n", 1)[1])
    assert reference["application_context"] == "Export: cancelled by the user."
    assert "20 percent" not in llm.messages[1].content


async def test_background_context_is_participant_scoped_by_the_caller() -> None:
    llm = _LLM(visual=False)
    conversation = QuickConversation(_Frames(), _Vision(), llm=llm)  # type: ignore[arg-type]

    await conversation.decide("What is happening?", app_context="Alice's upload is complete.")
    await conversation.decide("What is happening?", app_context="Bob's upload is pending.")

    assert "Alice" not in repr(llm.messages)
    assert "Bob" in llm.messages[1].content


async def test_visual_tool_response_does_not_speak_model_preamble() -> None:
    frames = _Frames()
    llm = _LLM(visual=True)
    conversation = QuickConversation(frames, _Vision(), llm=llm)  # type: ignore[arg-type]
    decision = ChatResponse(
        content="Okay, I'll take a look.",
        reasoning=None,
        tool_calls=[ToolCall(id="view-1", name="current_view", arguments="{}")],
        finish_reason="tool_calls",
        raw={},
    )

    answer = [text async for text in conversation.stream("What do you see?", "alice", decision=decision)]

    assert answer == ["The blue cup is on your left."]
    assert frames.participants == ["alice"]
