# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early advice is optional; final ASR input always has a non-model path."""

# ruff: noqa: E402
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from xr_ai_voice import TranscriptUpdate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/conversation-example/worker"))
from conversation_example_worker._semantic_turns import _Policy
from conversation_example_worker._semantic_voice import _ClefVoicePolicy


class _Decisions:
    def __init__(self, choice="yes", confidence=0.95, error=None):
        self.choice, self.confidence, self.error = choice, confidence, error
        self.calls = []
        self.entered = asyncio.Event()
        self.release = None
        self.cancelled = asyncio.Event()

    async def decide(self, state, questions, **kwargs):
        self.calls.append(state)
        self.entered.set()
        try:
            if self.release is not None:
                await self.release.wait()
            if self.error is not None:
                raise self.error
            return SimpleNamespace(answers={"ready": SimpleNamespace(choice=self.choice, confidence=self.confidence)})
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def _update(text, revision=0, segment=0, *, final=False, pid="user", epoch="source"):
    return TranscriptUpdate(pid, epoch, segment, revision, text, 123_000, final)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice,score,error",
    [
        ("no", 0.99, None),
        ("yes", 0.6409, None),
        ("yes", 0.99, RuntimeError("backend failed")),
        ("unknown", 0.99, None),
        ("yes", float("nan"), None),
    ],
)
async def test_early_failure_uncertainty_and_low_confidence_never_veto_final(choice, score, error):
    d = _Decisions(choice, score, error)
    p = _ClefVoicePolicy(d, _Policy(stable_s=0))
    p.update(_update("What is augmented reality"))
    assert await p.decide("user") == []
    p.update(_update("What is augmented reality", revision=1, final=True))
    (action,) = await p.decide("user")
    assert action.kind == "submit" and action.text == "What is augmented reality"
    assert len(d.calls) == 1
    assert await p.decide("user") == []


@pytest.mark.asyncio
async def test_final_input_never_calls_clef_even_without_any_early_result():
    d = _Decisions(error=AssertionError("final must not call Clef"))
    p = _ClefVoicePolicy(d)
    p.update(_update("What is this?", final=True))
    assert (await p.decide("user"))[0].text == "What is this?"
    assert d.calls == []


@pytest.mark.asyncio
async def test_confident_early_endpoint_deduplicates_final_and_preserves_new_suffix():
    d = _Decisions()
    p = _ClefVoicePolicy(d)
    p.update(_update("Explain AR"))
    (first,) = await p.decide("user")
    p.update(_update("Explain AR.", revision=1, final=True))
    assert await p.decide("user") == []
    p.update(_update("Explain AR and VR", revision=2, final=True))
    (second,) = await p.decide("user")
    assert second.text == "and VR" and first.request_id != second.request_id
    assert len(d.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,suppressed",
    [
        ("um uh", True),
        ("hmm", False),
        ("yes", False),
        ("no", False),
        ("Um, what is AR?", False),
        ("What does uh mean?", False),
    ],
)
async def test_final_filter_is_strict_even_if_a_model_would_reject_meaningful_text(text, suppressed):
    d = _Decisions(error=AssertionError("final must not call model"))
    p = _ClefVoicePolicy(d)
    p.update(_update(text, final=True))
    actions = await p.decide("user")
    assert bool(actions) is not suppressed
    if actions:
        assert actions[0].text == text
    assert d.calls == []


@pytest.mark.asyncio
async def test_early_timeout_does_not_block_final_input():
    d = _Decisions()
    d.release = asyncio.Event()
    p = _ClefVoicePolicy(d, _Policy(decision_timeout_s=0.01))
    p.update(_update("Explain AR"))
    assert await p.decide("user") == []
    assert d.cancelled.is_set()
    p.update(_update("Explain AR", revision=1, final=True))
    assert (await p.decide("user"))[0].text == "Explain AR"


@pytest.mark.asyncio
async def test_revised_or_reset_input_invalidates_awaiting_early_advice():
    d = _Decisions()
    d.release = asyncio.Event()
    p = _ClefVoicePolicy(d)
    p.update(_update("Explain AR"))
    pending = asyncio.create_task(p.decide("user"))
    await d.entered.wait()
    p.update(_update("", revision=1))
    d.release.set()
    assert await pending == []
    p.reset("user")
    p.update(_update("Explain VR", final=True, epoch="new-source"))
    assert (await p.decide("user"))[0].text == "Explain VR"


@pytest.mark.asyncio
async def test_large_final_input_is_not_truncated_or_blocked_by_early_model_limit():
    p = _ClefVoicePolicy(_Decisions(), _Policy(max_chars=4))
    p.update(_update("Explain augmented reality"))
    assert await p.decide("user") == []
    p.update(_update("Explain augmented reality", revision=1, final=True))
    assert (await p.decide("user"))[0].text == "Explain augmented reality"


@pytest.mark.asyncio
async def test_native_final_is_retained_when_next_partial_arrives_before_decision():
    d = _Decisions(choice="no")
    p = _ClefVoicePolicy(d)
    p.update(_update("What is augmented reality?", final=True))
    p.update(_update("And how does", segment=1))
    (first,) = await p.decide("user")
    assert first.text == "What is augmented reality?"
    assert d.calls == []
    assert await p.decide("user") == []
    assert len(d.calls) == 1
    p.update(_update("And how does it work?", segment=1, revision=1, final=True))
    assert (await p.decide("user"))[0].text == "And how does it work?"


@pytest.mark.asyncio
async def test_burst_of_native_finals_is_retained_without_model_work():
    d = _Decisions(error=AssertionError("native endpoint must not call model"))
    p = _ClefVoicePolicy(d)
    for segment in range(4):
        p.update(_update(f"Question {segment}", segment=segment, final=True))
    assert [a.text for a in await p.decide("user")] == [f"Question {i}" for i in range(4)]
    assert await p.decide("user") == [] and d.calls == []
    p.update(_update("Retired query", segment=4, final=True))
    p.reset("user")
    assert await p.decide("user") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("final,label", [(False, "Clef"), (True, "VAD")])
@pytest.mark.parametrize("delivery_fails", [False, True])
async def test_admitted_query_sends_sample_owned_source_message_without_changing_query(final, label, delivery_fails):
    from unittest.mock import AsyncMock

    from conversation_example_worker._endpoint_feedback import _EndpointFeedback
    from conversation_example_worker._topics import USER_QUERY_TOPIC
    from xr_ai_interaction import UserQuery
    from xr_ai_runtime import Agent, AgentRuntime, subscribe

    policy = _ClefVoicePolicy(_Decisions())
    policy.update(_update("What is augmented reality?", final=final))
    (action,) = await policy.decide("user")
    received = []

    class Receiver(Agent):
        @subscribe(USER_QUERY_TOPIC)
        async def query(self, query: UserQuery, ctx):
            received.append(query.text)

    send = AsyncMock(side_effect=RuntimeError("closed") if delivery_fails else None)
    runtime = AgentRuntime()
    runtime.register("feedback", _EndpointFeedback(policy, SimpleNamespace(send_return_data=send)))
    runtime.register("receiver", Receiver())
    async with runtime:
        await runtime.publish(
            USER_QUERY_TOPIC,
            UserQuery(text=action.text, timestamp_us=action.timestamp_us, request_id=action.request_id),
            participant_id="user",
        )
        message = send.call_args.args[0]
        assert message.topic == "voice.user-query" and message.participant_id == "user"
        assert message.pts_us == action.timestamp_us
        assert message.data.decode() == f"User [{label}]: What is augmented reality?"
        assert received == ["What is augmented reality?"]
        assert policy.take_admission_source("user", action.request_id) is None
        await runtime.publish(
            USER_QUERY_TOPIC, UserQuery(text="Typed input", timestamp_us=1), participant_id="user",
        )
        assert send.await_count == 1
        assert received[-1] == "Typed input"
