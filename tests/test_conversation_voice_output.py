# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise routed Conversation output through the actual typed runtime and voice consumer."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections import deque

import pytest
import xr_ai_routing._router as router_module
from xr_ai_conversation import Conversation, ConversationContextUpdate
from xr_ai_interaction import ResponseOutput, UserQuery
from xr_ai_models import ChatStreamEvent, DecisionAnswer, DecisionResponse
from xr_ai_routing import Admission, AgentExchange, AgentRouter, AgentTurn, RoutingAid
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe

QUERIES = Topic("routing-test.queries", UserQuery)
OUTPUT = Topic("routing-test.output", ResponseOutput, telemetry="none")
CONTEXT = Topic("routing-test.context", ConversationContextUpdate)


class _Output(Agent):
    def __init__(self):
        super().__init__()
        self.items = []
        self.changed = asyncio.Event()

    @subscribe(OUTPUT)
    async def receive(self, value: ResponseOutput, ctx: RuntimeContext):
        self.items.append((value, ctx.metadata))
        self.changed.set()

    async def wait(self, count):
        while len(self.items) < count:
            self.changed.clear()
            await self.changed.wait()



class _Decisions:
    def __init__(self, *choices):
        self.choices = deque(choices)
        self.calls = []

    async def decide(self, state, questions, **_kwargs):
        self.calls.append((state, questions))
        key = next(iter(questions))
        choice = self.choices.popleft() if key == "destination" else "slow"
        return DecisionResponse(
            {key: DecisionAnswer(choice, 1.0, {choice: 1.0})},
            "fake",
            {},
            {},
        )



async def _control(_turn, _source, _text):
    return None


async def _standalone_started():
    return None



async def test_configured_control_output_preserves_concurrent_participant_turn_identity():
    alice_entered, release_alice = asyncio.Event(), asyncio.Event()
    delivered_turns = []

    class Clarifier:
        def routing_aid(self, _participant):
            return RoutingAid("conversation", "Ask for a specific request")

        async def admit(self, turn, *, focused):
            return Admission("clarify", text=f"Clarify {turn.query}")

        async def answer(self, _turn, *, started):
            raise AssertionError("clarification does not start destination work")

        async def release(self, _participant):
            pass

    runtime = AgentRuntime()
    output = runtime.register("output", _Output())

    async def publish(turn, source, text):
        if turn.participant_id == "alice":
            alice_entered.set()
            await release_alice.wait()
        delivered_turns.append((turn, source))
        await runtime.publish(
            OUTPUT, ResponseOutput(kind="reply", text=text, request_id=turn.turn_id, timestamp_us=turn.timestamp_us),
            participant_id=turn.participant_id, source="router",
        )

    router = AgentRouter({"conversation": Clarifier()}, None, output=publish)
    alice = AgentTurn("alice", "alice-turn", "Alice request", (AgentExchange("earlier", "Alice context"),), 11)
    bob = AgentTurn("bob", "bob-turn", "Bob request", timestamp_us=22)
    async with runtime:
        first = asyncio.create_task(router.handle(alice))
        try:
            await alice_entered.wait()
            await asyncio.wait_for(router.handle(bob), 1)
            assert [(value.timestamp_us, metadata.participant_id) for value, metadata in output.items] == [(22, "bob")]
            release_alice.set()
            await first
            assert delivered_turns == [(bob, "conversation"), (alice, "conversation")]
            assert [metadata.participant_id for _, metadata in output.items] == ["bob", "alice"]
            assert [value.text for value, _ in output.items] == ["Clarify Bob request", "Clarify Alice request"]
            assert [value.timestamp_us for value, _ in output.items] == [22, 11]
        finally:
            release_alice.set()
            await asyncio.gather(first, return_exceptions=True)
            await router.stop()


async def test_conversation_context_topic_and_direct_output_retain_complete_history():
    class LLM:
        def __init__(self):
            self.messages = []

        def stream_events(self, messages):
            self.messages.append(messages)

            async def response():
                yield ChatStreamEvent(kind="text", text="answer one ")
                yield ChatStreamEvent(kind="text", text="answer two")

            return response()

    llm = LLM()
    runtime = AgentRuntime()
    output = runtime.register("output", _Output())
    conversation = Conversation(
        llm, summary_context_topics=(CONTEXT.name,), runtime=runtime, output_topic=OUTPUT,
    )
    runtime.register("context", conversation.context_subscriber())
    router = AgentRouter({"conversation": conversation}, None, output=_control)
    async with runtime:
        try:
            await runtime.publish(
                CONTEXT, ConversationContextUpdate(text="explicit application context"), participant_id="alice"
            )
            await router.handle(AgentTurn("alice", "first", "first request", timestamp_us=1234))
            await router.handle(AgentTurn("alice", "followup", "followup", timestamp_us=1234))
            assert "first request" in str(llm.messages[1])
            assert "answer one answer two" in str(llm.messages[1])
            assert router._history["alice"] == [
                AgentExchange("first request", "answer one answer two"),
                AgentExchange("followup", "answer one answer two"),
            ]
            assert [value.kind for value, _ in output.items] == ["reply"] * 6
            assert [value.request_id for value, _ in output.items] == ["first"] * 3 + ["followup"] * 3
            assert [value.final for value, _ in output.items] == [False, False, True] * 2
            assert output.items[0][0].response_id == output.items[2][0].response_id
            assert output.items[0][0].response_id != output.items[3][0].response_id
            assert "explicit application context" in str(llm.messages[0])
            assert all(meta.source == "conversation" for _, meta in output.items)
            assert all(value.timestamp_us == 1234 for value, _ in output.items)
            assert len(output.items) == 6
        finally:
            await router.stop()
            await conversation.close()



async def test_conversations_publish_directly_to_shared_typed_topic_with_own_source():
    class LLM:
        def stream_events(self, _messages):
            async def response():
                yield ChatStreamEvent(kind="reasoning", reasoning="private reasoning")
                yield ChatStreamEvent(kind="text", text=" ")
                yield ChatStreamEvent(kind="text", text="spoken ")
                yield ChatStreamEvent(kind="text", text="answer")

            return response()

    runtime = AgentRuntime()
    output = runtime.register("output", _Output())
    conversations = [
        Conversation(LLM(), routing=RoutingAid(name, name), runtime=runtime, output_topic=OUTPUT)
        for name in ("general", "specialist")
    ]
    async with runtime:
        try:
            for index, conversation in enumerate(conversations):
                result = await conversation.answer(
                    AgentTurn("alice", f"turn-{index}", "request", timestamp_us=17), started=_standalone_started,
                )
                assert result.text == "spoken answer"
            assert [value.text for value, _ in output.items] == ["spoken ", "answer", ""] * 2
            assert [value.kind for value, _ in output.items] == ["reply"] * 6
            assert [value.final for value, _ in output.items] == [False, False, True] * 2
            for index, (value, metadata) in enumerate(output.items):
                assert metadata.participant_id == "alice"
                assert metadata.source == ("general" if index < 3 else "specialist")
                assert value.request_id == f"turn-{index // 3}"
                assert value.timestamp_us == 17
        finally:
            for conversation in conversations:
                await conversation.close()



@pytest.mark.parametrize("failed_frame", [1, 2])
async def test_conversation_does_not_retry_or_finalize_failed_typed_publication(failed_frame):
    closed = asyncio.Event()

    class LLM:
        def stream_events(self, _messages):
            async def response():
                try:
                    for text in ("first", "second", "third"):
                        yield ChatStreamEvent(kind="text", text=text)
                finally:
                    closed.set()

            return response()

    class FailingOutput(_Output):
        @subscribe(OUTPUT)
        async def receive(self, value: ResponseOutput, ctx: RuntimeContext):
            await super().receive(value, ctx)
            if len(self.items) == failed_frame:
                raise RuntimeError("output delivery failed")

    runtime = AgentRuntime()
    output = runtime.register("output", FailingOutput())
    conversation = Conversation(LLM(), runtime=runtime, output_topic=OUTPUT)
    async with runtime:
        try:
            with pytest.raises(ExceptionGroup):
                await conversation.answer(AgentTurn("alice", "turn", "request"), started=_standalone_started)
            assert len(output.items) == failed_frame
            assert not any(value.final for value, _ in output.items)
            assert closed.is_set()
            assert conversation._store._context.transcript("alice") == ()
        finally:
            await conversation.close()



async def test_failed_final_frame_does_not_commit_the_exchange():
    class LLM:
        def stream_events(self, _messages):
            async def response():
                yield ChatStreamEvent(kind="text", text="answer")

            return response()

    class FailingFinal(_Output):
        @subscribe(OUTPUT)
        async def receive(self, value: ResponseOutput, ctx: RuntimeContext):
            await super().receive(value, ctx)
            if value.final:
                raise RuntimeError("final delivery failed")

    runtime = AgentRuntime()
    runtime.register("output", FailingFinal())
    conversation = Conversation(LLM(), runtime=runtime, output_topic=OUTPUT)
    async with runtime:
        try:
            with pytest.raises(ExceptionGroup):
                await conversation.answer(AgentTurn("alice", "turn", "request"), started=_standalone_started)
            assert conversation._store._context.transcript("alice") == ()
        finally:
            await conversation.close()


async def test_interrupted_final_frame_does_not_commit_the_exchange():
    final_entered = asyncio.Event()

    class LLM:
        def stream_events(self, _messages):
            async def response():
                yield ChatStreamEvent(kind="text", text="answer")

            return response()

    class BlockingFinal(_Output):
        @subscribe(OUTPUT)
        async def receive(self, value: ResponseOutput, ctx: RuntimeContext):
            await super().receive(value, ctx)
            if value.final:
                final_entered.set()
                await asyncio.Event().wait()

    runtime = AgentRuntime()
    runtime.register("output", BlockingFinal())
    conversation = Conversation(LLM(), runtime=runtime, output_topic=OUTPUT)
    async with runtime:
        try:
            answering = asyncio.create_task(
                conversation.answer(AgentTurn("alice", "turn", "request"), started=_standalone_started)
            )
            await asyncio.wait_for(final_entered.wait(), timeout=1)
            answering.cancel()
            with pytest.raises(asyncio.CancelledError):
                await answering
            assert conversation._store._context.transcript("alice") == ()
        finally:
            await conversation.close()


async def test_answer_shares_request_id_with_router_preamble():
    class LLM:
        def stream_events(self, _messages):
            async def response():
                yield ChatStreamEvent(kind="text", text="answer")

            return response()

    class Preambled(Conversation):
        async def admit(self, _turn, *, focused):
            return Admission("accept", preamble="One moment.")

    runtime = AgentRuntime()
    output = runtime.register("output", _Output())

    async def publish_control(turn, source, text):
        await runtime.publish(
            OUTPUT, ResponseOutput(kind="reply", text=text, request_id=turn.turn_id, timestamp_us=turn.timestamp_us),
            participant_id=turn.participant_id, source=source,
        )

    conversation = Preambled(LLM(), runtime=runtime, output_topic=OUTPUT)
    router = AgentRouter({"conversation": conversation}, None, output=publish_control)
    async with runtime:
        try:
            await router.handle(AgentTurn("alice", "request", "request"))
            assert [(value.kind, value.request_id, value.text) for value, _ in output.items] == [
                ("reply", "request", "One moment."),
                ("reply", "request", "answer"),
                ("reply", "request", ""),
            ]
            assert output.items[0][0].response_id is None
            assert output.items[1][0].response_id == output.items[2][0].response_id is not None
        finally:
            await router.stop()
            await conversation.close()


@pytest.mark.parametrize("partial", [False, True])
async def test_model_failure_is_reported_once_and_partial_stream_is_finalized(partial):
    closed = asyncio.Event()
    notifications = []
    control = []

    async def record_control(_turn, source, text):
        control.append((source, text))

    class Recording(Conversation):
        async def answer(self, turn, *, started):
            async def recorded():
                await started()
                notifications.append("started")

            return await super().answer(turn, started=recorded)

    class LLM:
        def stream_events(self, _messages):
            async def response():
                try:
                    if partial:
                        yield ChatStreamEvent(kind="text", text="Partial answer")
                    raise RuntimeError("model failed")
                finally:
                    closed.set()

            return response()

    runtime = AgentRuntime()
    output = runtime.register("output", _Output())

    conversation = Recording(LLM(), runtime=runtime, output_topic=OUTPUT)
    router = AgentRouter({"conversation": conversation}, None, output=record_control)
    async with runtime:
        try:
            with pytest.raises(RuntimeError, match="model failed"):
                await router.handle(AgentTurn("alice", "request", "request"))
            assert closed.is_set()
            assert notifications == (["started"] if partial else [])
            assert router._history["alice"] == []
            assert conversation._store._context.transcript("alice") == ()
            assert all(meta.source == "conversation" for _, meta in output.items)
            if partial:
                assert [value.text for value, _ in output.items] == ["Partial answer", ""]
                assert output.items[-1][0].final
                assert control == []
            else:
                assert output.items == []
                assert control == [("conversation", router_module._FAILURE_TEXT)]
        finally:
            await router.stop()
            await conversation.close()



async def test_finite_ack_does_not_block_direct_answer_in_bounded_voice_consumer(monkeypatch):
    from xr_ai_models import ChatResponse
    from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceAgent
    from xr_ai_voice import _runtime as voice_module

    class Session:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.accepted = asyncio.Event()
            self.spoken = []

        async def enqueue_reply(self, _participant, response, **_kwargs):
            self.queue.put_nowait(response)
            self.accepted.set()

        async def consume(self):
            while True:
                response = await self.queue.get()
                try:
                    text = response if isinstance(response, str) else "".join([chunk async for chunk in response])
                    self.spoken.append(text)
                finally:
                    self.queue.task_done()

    session = Session()
    monkeypatch.setattr(voice_module, "_VoiceSession", lambda **_kwargs: session)
    runtime = AgentRuntime()
    runtime.register(
        "voice",
        VoiceAgent(
            query_topic=QUERIES,
            stt=object(),
            tts=object(),
            vad=object(),
            voice_gate=object(),
            response_capacity=1,
        ),
    )

    class LLM:
        def stream_events(self, _messages):
            async def response():
                await session.accepted.wait()
                for _ in range(8):
                    yield ChatStreamEvent(kind="text", text="answer ")

            return response()

    class Ack:
        async def chat(self, *_args, **_kwargs):
            return ChatResponse("Let me think through the requested details.", None, None, "stop", {})

    conversation = Conversation(LLM(), runtime=runtime, output_topic=VOICE_OUTPUT_TOPIC)

    async def emit_control(turn, _source, text):
        assert (turn.participant_id, turn.turn_id, turn.timestamp_us) == ("alice", "request", 928)
        await runtime.publish(
            VOICE_OUTPUT_TOPIC,
            ResponseOutput(kind="reply", text=text, request_id=turn.turn_id, timestamp_us=turn.timestamp_us),
            participant_id=turn.participant_id, source="router",
        )

    router = AgentRouter(
        {"conversation": conversation}, _Decisions(), output=emit_control,
        acknowledgment_llm=Ack(), acknowledgment_delay_secs=10,
    )

    consumer = asyncio.create_task(session.consume())
    async with runtime:
        try:
            await asyncio.wait_for(
                router.handle(AgentTurn("alice", "request", "provide the details", timestamp_us=928)), 1,
            )
            await asyncio.wait_for(session.queue.join(), 1)
            assert session.spoken == ["Let me think through the requested details. ", "answer " * 8]
            assert router._history["alice"] == [AgentExchange("provide the details", ("answer " * 8).strip())]
        finally:
            await router.stop()
            await conversation.close()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)



@pytest.mark.parametrize(
    ("package", "excluded"),
    [
        ("xr_ai_interaction", ("xr_ai_voice", "xr_ai_runtime")),
        ("xr_ai_routing", ("xr_ai_voice", "xr_ai_runtime", "xr_ai_interaction")),
        ("xr_ai_conversation", ("xr_ai_voice",)),
    ],
)
def test_pure_package_import_does_not_load_voice_media(package, excluded):
    result = subprocess.run(
        [sys.executable, "-c", f"import sys, {package}; assert not set({excluded!r}) & set(sys.modules)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr



@pytest.mark.parametrize("after_cancel", ["yield", "return", "error"])
@pytest.mark.parametrize("partial", [False, True])
async def test_conversation_drops_late_model_output_after_interruption(after_cancel, partial):
    started, closed = asyncio.Event(), asyncio.Event()

    class LLM:
        def stream_events(self, _messages):
            async def response():
                try:
                    if partial:
                        yield ChatStreamEvent(kind="text", text="partial")
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        if after_cancel == "yield":
                            yield ChatStreamEvent(kind="text", text="late after interruption")
                        elif after_cancel == "error":
                            raise RuntimeError("failed after interruption") from None
                        else:
                            return
                finally:
                    closed.set()

            return response()

    runtime = AgentRuntime()
    output = runtime.register("output", _Output())

    conversation = Conversation(LLM(), runtime=runtime, output_topic=OUTPUT)
    router = AgentRouter({"conversation": conversation}, None, output=_control)
    async with runtime:
        try:
            answering = asyncio.create_task(
                router.handle(AgentTurn("alice", "request", "request"))
            )
            await started.wait()
            await router.interrupt("alice")
            with pytest.raises(asyncio.CancelledError):
                await answering
            assert closed.is_set()
            # An opened stream is closed by an empty final frame and receives no late text.
            assert [(value.text, value.final) for value, _ in output.items] == (
                [("partial", False), ("", True)] if partial else []
            )
            assert router._history["alice"] == []
            assert conversation._store._context.transcript("alice") == ()
        finally:
            await router.stop()
            await conversation.close()
