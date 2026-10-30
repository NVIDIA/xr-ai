# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot alert ownership and output boundaries; scripted decisions are not model evaluations."""

from __future__ import annotations

import asyncio
import gc
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger
from test_conversation_example_agents import _Frames, _next_call
from xr_ai_conversation import ContextManager, ConversationContextUpdate
from xr_ai_hub import FrameUnavailable
from xr_ai_models import Capabilities, ChatResponse
from xr_ai_routing import AgentExchange, AgentTurn
from xr_ai_voice import VOICE_OUTPUT_TOPIC

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agent-samples/conversation-example/worker"))

from conversation_example_worker._alert_agent import VisualAlertAgent  # noqa: E402


class _Output:
    def __init__(self, *, fail_background=False):
        self.messages = []
        self.background = asyncio.Event()
        self.description = asyncio.Event()
        self.fail_background = fail_background

    async def publish(self, _topic, message, **metadata):
        self.messages.append((message, metadata))
        if message.kind == "update" and not message.urgent:
            self.description.set()
        if message.kind == "update" and message.urgent:
            self.background.set()
            if self.fail_background:
                raise RuntimeError("output unavailable")


class _VLM:
    def __init__(self):
        self.calls = []
        self.observed_images = []
        self.entered = asyncio.Event()
        self.resume = None
        self.closed = asyncio.Event()
        self.error = None
        self.block_observation = False
        self.observation_started = asyncio.Event()
        self.observation_cleanup = asyncio.Event()
        self.release_observation_cleanup = asyncio.Event()
        self.observation_cleaned = asyncio.Event()
        self.cleanup_error = None

    async def ask_images(self, images, question, **kwargs):
        self.observed_images.append((tuple(images), question))
        self.observation_started.set()
        if self.block_observation:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.observation_cleanup.set()
                await self.release_observation_cleanup.wait()
                self.observation_cleaned.set()
                if self.cleanup_error is not None:
                    raise self.cleanup_error from None
                raise
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content="The lamp is blue.")

    async def stream_images(self, images, question, **kwargs):
        self.calls.append((images, question))
        self.entered.set()
        try:
            if self.error is not None:
                raise self.error
            yield "The lamp is blue."
            if self.resume is not None:
                await self.resume.wait()
        finally:
            self.closed.set()


class _Images:
    def __init__(self, *, unavailable=False):
        self.unavailable = unavailable
        self.resolved = []

    def resolve(self, reference):
        if self.unavailable:
            raise LookupError("image unavailable")
        self.resolved.append(reference)
        return reference.uri.encode()


_READY = json.dumps({"status": "ready", "condition": "the lamp is blue", "relevance": "the lamp"})
_UNCLEAR = json.dumps({"status": "unclear"})


class _LLM:
    """Answer every alert compilation with the next scripted text."""

    capabilities = Capabilities()

    def __init__(self, *answers):
        self.answers = deque(answers or (_READY,))
        self.requests = []

    async def chat(self, messages, **_kwargs):
        self.requests.append(messages)
        content = self.answers[0] if len(self.answers) == 1 else self.answers.popleft()
        return ChatResponse(content=content, reasoning=None, tool_calls=None, finish_reason="stop", raw={})


class _Decisions:
    """Script each decision path: control, relevance screen, direct image check, observation check."""

    def __init__(
        self,
        *controls,
        relevance="yes",
        fast=("no", 1.0),
        backup=("no", 1.0),
        synchronize=False,
        fast_gate=None,
        ignore_fast_cancel=False,
        fast_cancel_error=None,
    ):
        self.controls = deque(controls)
        self.relevance = deque(relevance if isinstance(relevance, list) else [relevance])
        self.fast = deque(fast if isinstance(fast, list) else [fast])
        self.backup = deque(backup if isinstance(backup, list) else [backup])
        self.calls = []
        self.fast_called = asyncio.Event()
        self.backup_called = asyncio.Event()
        self.synchronize = synchronize
        self.relevance_called = asyncio.Event()
        self.both_ready = asyncio.Event()
        self.ready_count = 0
        self.fast_gate = fast_gate
        self.ignore_fast_cancel = ignore_fast_cancel
        self.fast_cancel_error = fast_cancel_error

    def paths(self):
        return [call[0]["decision_path"] for call in self.calls if "decision_path" in call[0]]

    async def decide(self, state, questions, *, images=(), timeout=None):
        self.calls.append((state, questions, tuple(images), timeout))
        if "alert_action" in questions:
            choice = self.controls.popleft()
            confidence = 1.0
        elif "relevant" in questions:
            self.relevance_called.set()
            choice = self.relevance[0] if len(self.relevance) == 1 else self.relevance.popleft()
            confidence = 1.0
        else:
            path = state["decision_path"]
            (self.fast_called if path == "fast-clef" else self.backup_called).set()
            results = self.fast if path == "fast-clef" else self.backup
            choice, confidence = results[0] if len(results) == 1 else results.popleft()
            if path == "fast-clef" and self.fast_gate is not None:
                try:
                    await self.fast_gate.wait()
                except asyncio.CancelledError:
                    if self.fast_cancel_error is not None:
                        raise self.fast_cancel_error from None
                    if not self.ignore_fast_cancel:
                        raise
            if self.synchronize:
                self.ready_count += 1
                if self.ready_count == 2:
                    self.both_ready.set()
                await self.both_ready.wait()
            await asyncio.sleep(0)
        key = next(iter(questions))
        return SimpleNamespace(
            answers={key: SimpleNamespace(choice=choice, confidence=confidence, probabilities={choice: confidence})},
            model="scripted",
            usage={},
            raw={},
        )


def _agent(
    decisions,
    frames,
    images=None,
    *,
    output=None,
    vlm=None,
    llm=None,
    context=None,
    check_interval_secs=1.0,
    confirm_interval_secs=3.0,
):
    output = output or _Output()
    sleeps = []
    llm = llm or _LLM()

    async def sleep(delay):
        sleeps.append(delay)
        await asyncio.sleep(0)

    agent = VisualAlertAgent(
        frames,
        images or _Images(),
        decisions,
        vlm=vlm or _VLM(),
        llm=llm,
        context=context or ContextManager(llm),
        runtime=output,
        output_topic=VOICE_OUTPUT_TOPIC,
        check_interval_secs=check_interval_secs,
        confirm_interval_secs=confirm_interval_secs,
        sleep=sleep,
    )
    agent.sleeps = sleeps
    return agent, output


async def _start(agent, participant="alice", request="Alert me when the lamp is blue.", turn_id="start", history=()):
    turn = AgentTurn(participant, turn_id, request, history=tuple(history))
    assert (await agent.admit(turn, focused=False)).disposition == "accept"

    async def started():
        return None

    await agent.answer(turn, started=started)


def _open():
    event = asyncio.Event()
    event.set()
    return event


def _spoken(output):
    return [message.text for message, _ in output.messages if message.text]


@pytest.mark.asyncio
async def test_alert_speaks_once_after_matching_and_never_publishes_observations():
    gate = _open()
    frames = _Frames((gate, None))
    images = _Images()
    decisions = _Decisions("start", fast=("no", 1.0), backup=("yes", 0.9))
    agent, output = _agent(decisions, frames, images)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(job.task, 1)
    assert not agent._jobs
    await agent.stop()
    alerts = [(message, metadata) for message, metadata in output.messages if message.text == "Alert"]
    assert len(alerts) == 1
    notice = alerts[0][0]
    assert notice.kind == "update" and notice.urgent
    descriptions = [message for message, _ in output.messages if message.kind == "update" and not message.urgent]
    assert [description.text for description in descriptions] == ["The lamp is blue."]
    assert alerts[0][1] == {"participant_id": "alice", "source": "alerts"}
    assert len(frames.calls) == 1 and len(images.resolved) == 1
    assert not agent._jobs
    assert frames.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["frame", "media"])
async def test_unavailable_evidence_stops_without_alert_or_condition_decision(failure):
    gate = _open()
    failure_value = FrameUnavailable("no frame") if failure == "frame" else None
    frames = _Frames((gate, failure_value), (gate, failure_value))
    images = _Images(unavailable=failure == "media")
    decisions = _Decisions("start")
    agent, output = _agent(decisions, frames, images)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(job.task, 1)
    assert not agent._jobs
    await agent.stop()
    assert not any(message.text == "Alert" for message, _ in output.messages)
    assert output.messages[-1][0].text == "I couldn't check the visual alert. It has stopped."
    assert len(decisions.calls) == 1
    assert not agent._jobs


@pytest.mark.asyncio
async def test_cancel_and_declined_fragment_preserve_participant_isolation():
    frames = _Frames()
    agent, output = _agent(_Decisions("start", "start", "decline", "cancel"), frames)
    await _start(agent, "alice")
    await _next_call(frames)
    await _start(agent, "bob")
    await _next_call(frames)
    alice_job, bob_job = agent._jobs["alice"], agent._jobs["bob"]
    assert (
        await agent.admit(AgentTurn("alice", "fragment", "A lamp over there."), focused=False)
    ).disposition == "decline"
    assert agent._jobs["alice"] is alice_job
    await _start(agent, "alice", "Cancel the visual alert.", "cancel")
    assert alice_job.task.done()
    assert agent._jobs["bob"] is bob_job and not bob_job.task.done()
    await agent.participant_left("bob")
    await agent.stop()
    assert frames.active == 0
    assert not any(message.text == "Alert" for message, _ in output.messages)


@pytest.mark.asyncio
async def test_shutdown_during_start_ack_cannot_create_a_late_job():
    agent, _ = _agent(_Decisions("start"), _Frames())
    turn = AgentTurn("alice", "start", "Alert me when the parcel is visible.")
    await agent.admit(turn, focused=False)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def started():
        entered.set()
        await resume.wait()

    pending = asyncio.create_task(agent.answer(turn, started=started))
    await entered.wait()
    await agent.stop()
    resume.set()
    await pending
    assert not agent._jobs


@pytest.mark.asyncio
async def test_failed_alert_publication_is_not_retried_as_another_message():
    gate = _open()
    agent, output = _agent(
        _Decisions("start", fast=("yes", 0.99)),
        _Frames((gate, None)),
        output=_Output(fail_background=True),
    )
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(job.task, 1)
    assert not agent._jobs
    await agent.stop()
    assert len([message for message, _ in output.messages if message.kind == "update" and message.urgent]) == 1
    assert not agent._jobs


@pytest.mark.asyncio
async def test_cancel_drains_followup_after_notice_without_more_spoken_output():
    vlm = _VLM()
    vlm.resume = asyncio.Event()
    agent, output = _agent(_Decisions("start", fast=("yes", 0.99)), _Frames((_open(), None)), vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(vlm.entered.wait(), 1)
    assert agent._jobs["alice"] is job
    await agent.participant_left("alice")
    await asyncio.wait_for(vlm.closed.wait(), 1)
    assert job.task.done() and not agent._jobs
    # The description is one finite update, so a description cut off by cancellation is never spoken.
    assert [message.text for message, _ in output.messages if message.kind == "update"] == ["Alert"]


@pytest.mark.asyncio
async def test_description_service_failure_does_not_repeat_the_initial_alert():
    vlm = _VLM()
    vlm.error = RuntimeError("description unavailable")
    agent, output = _agent(_Decisions("start", fast=("yes", 0.99)), _Frames((_open(), None)), vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(job.task, 1)
    messages = [message for message, _ in output.messages]
    assert sum(message.text == "Alert" for message in messages) == 1
    assert messages[-1].final and not agent._jobs


@pytest.mark.asyncio
async def test_direct_no_allows_observation_positive_on_the_same_frame():
    frames = _Frames((_open(), None))
    vlm = _VLM()
    decisions = _Decisions("start", fast=("no", 1.0), backup=("yes", 1.0))
    agent, output = _agent(decisions, frames, vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(job.task, 1)
    assert sum(message.text == "Alert" for message, _ in output.messages) == 1
    assert {"fast-clef", "cosmos-clef"} <= set(decisions.paths())
    assert len(vlm.observed_images) == 1
    # The description uses the frame that triggered the alert.
    assert vlm.observed_images[0][0][0] == vlm.calls[0][0][0]
    assert len(frames.calls) == 1 and frames.active == 0 and not agent._jobs


@pytest.mark.asyncio
async def test_fast_positive_survives_observation_service_failure():
    vlm = _VLM()
    vlm.error = RuntimeError("Cosmos unavailable")
    decisions = _Decisions("start", fast=("yes", 1.0), backup=("yes", 1.0))
    agent, output = _agent(decisions, _Frames((_open(), None)), vlm=vlm)
    await _start(agent)
    await asyncio.wait_for(agent._jobs["alice"].task, 1)
    assert sum(message.text == "Alert" for message, _ in output.messages) == 1
    assert not agent._jobs


@pytest.mark.asyncio
async def test_both_confirmations_failing_retires_the_alert():
    vlm = _VLM()
    vlm.error = RuntimeError("Cosmos unavailable")

    class FailingDirect(_Decisions):
        async def decide(self, state, questions, **kwargs):
            if state.get("decision_path") == "fast-clef":
                raise RuntimeError("Clef unavailable")
            return await super().decide(state, questions, **kwargs)

    agent, output = _agent(FailingDirect("start"), _Frames((_open(), None)), vlm=vlm)
    await _start(agent)
    await asyncio.wait_for(agent._jobs["alice"].task, 1)
    assert _spoken(output)[-1] == "I couldn't check the visual alert. It has stopped."
    assert not agent._jobs


@pytest.mark.asyncio
async def test_both_paths_positive_publish_exactly_one_alert():
    decisions = _Decisions("start", fast=("yes", 1.0), backup=("yes", 1.0), synchronize=True)
    agent, output = _agent(decisions, _Frames((_open(), None)))
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(job.task, 1)
    assert decisions.both_ready.is_set()
    assert sum(message.text == "Alert" for message, _ in output.messages) == 1
    assert sum(message.text == "The lamp is blue." for message, _ in output.messages) == 1
    assert not agent._jobs


def _slow_sibling_cleanup():
    """Direct check confirms while the Cosmos check is stuck in cancellation cleanup."""
    vlm = _VLM()
    vlm.block_observation = True
    fast_gate = asyncio.Event()
    decisions = _Decisions("start", fast=("yes", 1.0), backup=("no", 1.0), fast_gate=fast_gate)
    return vlm, fast_gate, decisions


@pytest.mark.asyncio
async def test_alert_is_announced_after_the_sibling_check_finishes():
    vlm, fast_gate, decisions = _slow_sibling_cleanup()
    frames = _Frames((_open(), None))
    agent, output = _agent(decisions, frames, vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(vlm.observation_started.wait(), 1)
    fast_gate.set()
    await asyncio.wait_for(vlm.observation_cleanup.wait(), 1)
    assert not any(message.text == "Alert" for message, _ in output.messages)
    vlm.release_observation_cleanup.set()
    await asyncio.wait_for(job.task, 1)
    assert sum(message.text == "Alert" for message, _ in output.messages) == 1
    assert not agent._jobs and frames.active == 0


@pytest.mark.asyncio
async def test_departure_during_sibling_cleanup_waits_for_that_cleanup():
    vlm, fast_gate, decisions = _slow_sibling_cleanup()
    agent, output = _agent(decisions, _Frames((_open(), None)), vlm=vlm)
    await _start(agent)
    await asyncio.wait_for(vlm.observation_started.wait(), 1)
    fast_gate.set()
    await asyncio.wait_for(vlm.observation_cleanup.wait(), 1)
    for _ in range(20):
        await asyncio.sleep(0)

    # Departure cancels the watch a second time while the losing check is still cleaning up.
    leaving = asyncio.create_task(agent.participant_left("alice"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert not leaving.done() and not vlm.observation_cleaned.is_set()
    vlm.release_observation_cleanup.set()
    await asyncio.wait_for(leaving, 1)

    assert vlm.observation_cleaned.is_set()
    assert not agent._jobs
    assert not any(message.text == "Alert" for message, _ in output.messages)


@pytest.mark.asyncio
async def test_losing_check_failing_during_cleanup_is_logged_not_left_unretrieved():
    # The observation check wins; the direct check, still waiting on Clef, fails while cancelling.
    decisions = _Decisions(
        "start",
        fast=("no", 1.0),
        backup=("yes", 1.0),
        fast_gate=asyncio.Event(),
        fast_cancel_error=RuntimeError("Clef cleanup failed"),
    )
    unretrieved = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: unretrieved.append(context))
    logs = []
    sink = logger.add(logs.append, format="{message}", level="WARNING")
    try:
        agent, output = _agent(decisions, _Frames((_open(), None)))
        await _start(agent)
        job = agent._jobs["alice"]
        await asyncio.wait_for(job.task, 1)
        del job
        gc.collect()
    finally:
        logger.remove(sink)
        loop.set_exception_handler(None)

    assert sum(message.text == "Alert" for message, _ in output.messages) == 1
    assert any("alert-fast-clef" in line and "Clef cleanup failed" in line for line in logs), logs
    assert not unretrieved


@pytest.mark.asyncio
async def test_late_yes_after_replacement_cannot_claim_or_publish():
    frames = _Frames((_open(), None), (_open(), None))
    fast_gate = asyncio.Event()
    decisions = _Decisions(
        "start", "start", fast=("yes", 1.0), backup=("no", 1.0), fast_gate=fast_gate, ignore_fast_cancel=True
    )
    agent, output = _agent(decisions, frames)
    await _start(agent, turn_id="first")
    old_job = agent._jobs["alice"]
    await asyncio.wait_for(decisions.fast_called.wait(), 1)
    await _start(agent, turn_id="replacement")
    new_job = agent._jobs["alice"]
    assert old_job is not new_job and old_job.task.done()
    await agent.participant_left("alice")
    fast_gate.set()
    await asyncio.sleep(0)
    assert not any(message.text == "Alert" for message, _ in output.messages)
    assert new_job.task.done() and not agent._jobs and frames.active == 0


@pytest.mark.asyncio
async def test_only_a_screened_frame_runs_cosmos_and_the_condition_checks():
    gate = _open()
    frames = _Frames((gate, None), (gate, None), (gate, None))
    vlm = _VLM()
    # Hold the direct check until Cosmos has started, so both condition checks run on the screened frame.
    fast_gate = asyncio.Event()
    decisions = _Decisions(
        "start", relevance=["no", "no", "yes"], fast=("yes", 1.0), backup=("yes", 1.0), fast_gate=fast_gate
    )
    agent, output = _agent(decisions, frames, vlm=vlm)
    await _start(agent)
    await asyncio.wait_for(vlm.observation_started.wait(), 1)
    await asyncio.wait_for(decisions.backup_called.wait(), 1)
    fast_gate.set()
    await asyncio.wait_for(agent._jobs["alice"].task, 1)
    paths = decisions.paths()
    assert paths.count("relevance") == 3
    assert paths[:3] == ["relevance", "relevance", "relevance"]
    assert {"fast-clef", "cosmos-clef"} == set(paths[3:])
    assert len(vlm.observed_images) == 1
    assert sum(message.text == "Alert" for message, _ in output.messages) == 1


@pytest.mark.asyncio
async def test_unscreened_frames_never_run_cosmos_or_condition_checks():
    gate = _open()
    frames = _Frames((gate, None), (gate, None), (gate, None))
    vlm = _VLM()
    decisions = _Decisions("start", relevance="no", fast=("yes", 1.0), backup=("yes", 1.0))
    agent, output = _agent(decisions, frames, vlm=vlm)
    await _start(agent)
    for _ in range(3):
        await _next_call(frames)
    await _next_call(frames)
    await agent.participant_left("alice")
    assert set(decisions.paths()) == {"relevance"}
    assert not vlm.observed_images
    assert not any(message.text == "Alert" for message, _ in output.messages)


@pytest.mark.asyncio
async def test_checks_are_paced_and_a_screened_frame_waits_longer():
    gate = _open()
    frames = _Frames((gate, None), (gate, None), (gate, None))
    decisions = _Decisions(
        "start", relevance=["no", "yes", "yes"], fast=[("no", 1.0), ("yes", 1.0)], backup=("no", 1.0)
    )
    agent, _ = _agent(decisions, frames, check_interval_secs=1.0, confirm_interval_secs=5.0)
    await _start(agent)
    await asyncio.wait_for(agent._jobs["alice"].task, 1)
    # Frame 1 is screened out, frame 2 is screened in but not confirmed, frame 3 confirms.
    assert agent.sleeps == pytest.approx([1.0, 5.0], abs=0.5)


@pytest.mark.parametrize(
    ("check", "confirm"),
    [(-1.0, 3.0), (2.0, 1.0)],
)
def test_alert_intervals_must_be_ordered_and_nonnegative(check, confirm):
    with pytest.raises(ValueError, match="intervals"):
        _agent(_Decisions(), _Frames(), check_interval_secs=check, confirm_interval_secs=confirm)


@pytest.mark.asyncio
async def test_request_is_compiled_with_recent_exchanges_and_saved_context():
    llm = _LLM(
        json.dumps({"status": "ready", "condition": "the red cable is plugged in", "relevance": "the red cable"})
    )
    context = ContextManager(llm, summary_context_topics=("notes",))
    await context.update("alice", "notes", ConversationContextUpdate(text="Watch for the red cable."), mode="summary")
    decisions = _Decisions("start", relevance="no")
    agent, output = _agent(decisions, _Frames((_open(), None)), llm=llm, context=context)
    history = (AgentExchange("My spare part is the red cable.", "Noted."),)

    await _start(agent, request="Alert me about what I mentioned in my notes.", history=history)

    prompt = "\n".join(message.content for message in llm.requests[0])
    assert "Alert me about what I mentioned in my notes." in prompt
    assert "My spare part is the red cable." in prompt
    assert "Watch for the red cable." in prompt
    assert "Watch for the red cable" not in "\n".join(m.content for m in llm.requests[0][:1])
    assert _spoken(output) == ["I'll alert you once when the red cable is plugged in."]
    # The compiled watch, not the literal request, drives the checks.
    job = agent._jobs["alice"]
    assert (job.alert.condition, job.alert.relevance) == ("the red cable is plugged in", "the red cable")
    await asyncio.wait_for(decisions.relevance_called.wait(), 1)
    relevance_call = next(call for call in decisions.calls if "relevant" in call[1])
    assert relevance_call[0]["relevance"] == "the red cable"
    await agent.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("compiled", [_UNCLEAR, "not json", json.dumps({"status": "ready", "condition": "x"})])
async def test_unresolvable_request_starts_no_alert_and_keeps_a_running_one(compiled):
    llm = _LLM(_READY, compiled)
    frames = _Frames()
    agent, output = _agent(_Decisions("start", "start"), frames, llm=llm)
    await _start(agent, turn_id="first")
    running = agent._jobs["alice"]
    await _next_call(frames)

    await _start(agent, request="Alert me about that.", turn_id="second")

    assert agent._jobs["alice"] is running and not running.task.done()
    assert _spoken(output)[-1] == "I couldn't tell what to watch for. Please say what should trigger the alert."
    await agent.stop()


@pytest.mark.asyncio
async def test_departure_during_compilation_leaves_no_alert_running():
    entered, resume = asyncio.Event(), asyncio.Event()

    class SlowLLM(_LLM):
        async def chat(self, messages, **kwargs):
            entered.set()
            await resume.wait()
            return await super().chat(messages, **kwargs)

    agent, _ = _agent(_Decisions("start"), _Frames(), llm=SlowLLM())
    turn = AgentTurn("alice", "start", "Alert me when the lamp is blue.")
    await agent.admit(turn, focused=False)

    async def started():
        return None

    answering = asyncio.create_task(agent.answer(turn, started=started))
    await entered.wait()
    leaving = asyncio.create_task(agent.participant_left("alice"))
    await asyncio.sleep(0)
    resume.set()
    await asyncio.gather(answering, leaving)

    assert not agent._jobs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_text", "control", "reply"),
    [
        ("Cancel the visual alert.", "cancel", "The visual alert is cancelled."),
        ("Alert me when the door opens.", "start", "I'll alert you once when the lamp is blue."),
    ],
    ids=["cancel", "replace"],
)
async def test_cancelling_or_replacing_an_alert_withdraws_its_unspoken_updates(request_text, control, reply):
    decisions = _Decisions("start", control, fast=("yes", 1.0))
    agent, output = _agent(decisions, _Frames((_open(), None), (_open(), None)))
    await _start(agent)
    # The job has finished and left _jobs, but its description may still be queued for speech.
    await asyncio.wait_for(agent._jobs["alice"].task, 1)
    assert not agent._jobs

    await _start(agent, request=request_text, turn_id="second")

    kinds = [(message.kind, message.text) for message, _ in output.messages]
    after_description = kinds[kinds.index(("update", "The lamp is blue.")) + 1 :]
    # The second request takes back the queued description before its own reply.
    assert after_description.index(("withdraw", "")) < after_description.index(("reply", reply))
    await agent.stop()
