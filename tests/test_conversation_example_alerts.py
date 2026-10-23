# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot alert ownership and output boundaries; scripted decisions are not model evaluations."""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import nemo_relay
import pytest
from test_conversation_example_agents import _Frames, _next_call
from xr_ai_hub import FrameUnavailable
from xr_ai_routing import AgentTurn
from xr_ai_tools.vision import ImageQueryResult, ImageQueryTool
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

    async def ask_images(self, images, question, **kwargs):
        self.observed_images.append((tuple(images), question))
        self.observation_started.set()
        if self.block_observation:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.observation_cleanup.set()
                await self.release_observation_cleanup.wait()
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


class _HedgedDecisions:
    def __init__(
        self,
        *controls,
        fast=("no", 1.0),
        backup=("no", 1.0),
        synchronize=False,
        fast_gate=None,
        ignore_fast_cancel=False,
    ):
        self.controls = deque(controls)
        self.fast = deque(fast if isinstance(fast, list) else [fast])
        self.backup = deque(backup if isinstance(backup, list) else [backup])
        self.calls = []
        self.fast_called = asyncio.Event()
        self.backup_called = asyncio.Event()
        self.synchronize = synchronize
        self.both_ready = asyncio.Event()
        self.ready_count = 0
        self.fast_gate = fast_gate
        self.ignore_fast_cancel = ignore_fast_cancel

    async def decide(self, state, questions, *, images=(), timeout=None):
        self.calls.append((state, questions, tuple(images), timeout))
        if "alert_action" in questions:
            choice = self.controls.popleft()
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


def _agent(decisions, frames, images=None, *, output=None, vlm=None):
    output = output or _Output()
    capabilities = []

    async def publish(participant, update):
        capabilities.append((participant, update.text))

    agent = VisualAlertAgent(
        frames,
        images or _Images(),
        decisions,
        publish,
        vlm=vlm or _VLM(),
        runtime=output,
        output_topic=VOICE_OUTPUT_TOPIC,
    )
    return agent, output, capabilities


@pytest.fixture(autouse=True)
def _run_relay_calls_with_scripted_vlm(monkeypatch):
    async def stream_execute(_name, request, handler, _on_chunk, _on_complete, **kwargs):
        return handler(request)

    original_init = ImageQueryTool.__init__

    def init(tool, *, images, vlm, system_prompt=""):
        original_init(tool, images=images, vlm=vlm, system_prompt=system_prompt)
        tool._test_images = images
        tool._test_vlm = vlm
        tool._test_prompt = system_prompt

    async def execute(tool, request):
        image = tool._test_images.resolve(request.image)
        response = await tool._test_vlm.ask_images([image], request.query, system_prompt=tool._test_prompt)
        return ImageQueryResult(text=response.content)

    monkeypatch.setattr(nemo_relay.llm, "stream_execute", stream_execute)
    monkeypatch.setattr(ImageQueryTool, "__init__", init)
    monkeypatch.setattr(ImageQueryTool, "execute", execute)


async def _start(agent, participant="alice", request="Alert me when the lamp is blue.", turn_id="start"):
    turn = AgentTurn(participant, turn_id, request)
    assert (await agent.admit(turn, focused=False)).disposition == "accept"

    async def started():
        return None

    await agent.answer(turn, started=started)


def _open():
    event = asyncio.Event()
    event.set()
    return event


@pytest.mark.asyncio
async def test_alert_speaks_once_after_matching_and_never_publishes_observations():
    gate = _open()
    frames = _Frames((gate, None), (gate, None))
    images = _Images()
    decisions = _HedgedDecisions("start", fast=("no", 1.0), backup=("yes", 0.9))
    agent, output, capabilities = _agent(decisions, frames, images)
    await agent.participant_joined("alice")
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(job.task, 1)
    assert not agent._jobs
    await agent.stop()
    alerts = [(message, metadata) for message, metadata in output.messages if message.text == "Alert"]
    assert len(alerts) == 1
    assert alerts[0][0].text == "Alert"
    notice = alerts[0][0]
    assert notice.kind == "update" and notice.urgent
    descriptions = [message for message, _ in output.messages if message.kind == "update" and not message.urgent]
    assert [description.text for description in descriptions] == ["The lamp is blue."]
    assert alerts[0][1] == {"participant_id": "alice", "source": "alerts"}
    assert len(frames.calls) >= 2 and len(images.resolved) == 2
    assert len(capabilities) == 1
    assert "lamp" not in capabilities[0][1]
    assert not agent._jobs
    assert frames.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["frame", "media"])
async def test_unavailable_evidence_stops_without_alert_or_condition_decision(failure):
    gate = _open()
    failure_value = FrameUnavailable("no frame") if failure == "frame" else None
    frames = _Frames((gate, failure_value), (gate, failure_value))
    images = _Images(unavailable=failure == "media")
    decisions = _HedgedDecisions("start")
    agent, output, _ = _agent(decisions, frames, images)
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
    agent, output, _ = _agent(_HedgedDecisions("start", "start", "decline", "cancel"), frames)
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
    agent, _, _ = _agent(_HedgedDecisions("start"), _Frames())
    turn = AgentTurn("alice", "start", "Alert me when the parcel is visible.")
    await agent.admit(turn, focused=False)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def started():
        entered.set()
        await resume.wait()
        return True

    pending = asyncio.create_task(agent.answer(turn, started=started))
    await entered.wait()
    await agent.stop()
    resume.set()
    await pending
    assert not agent._jobs


@pytest.mark.asyncio
async def test_failed_alert_publication_is_not_retried_as_another_message():
    gate = _open()
    agent, output, _ = _agent(
        _HedgedDecisions("start", fast=("yes", 0.99)),
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
    agent, output, _ = _agent(_HedgedDecisions("start", fast=("yes", 0.99)), _Frames((_open(), None)), vlm=vlm)
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
    agent, output, _ = _agent(_HedgedDecisions("start", fast=("yes", 0.99)), _Frames((_open(), None)), vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(job.task, 1)
    messages = [message for message, _ in output.messages]
    assert sum(message.text == "Alert" for message in messages) == 1
    assert "unavailable" in "".join(message.text or "" for message in messages).lower()
    assert messages[-1].final and not agent._jobs


@pytest.mark.asyncio
async def test_direct_no_allows_backup_positive_with_its_own_frame():
    gate = _open()
    frames = _Frames((gate, None), (gate, None))
    vlm = _VLM()
    decisions = _HedgedDecisions("start", fast=("no", 1.0), backup=("yes", 1.0))
    agent, output, _ = _agent(decisions, frames, vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(job.task, 1)
    alert_messages = [message for message, _ in output.messages if message.text == "Alert"]
    assert len(alert_messages) == 1
    assert any(call[0].get("decision_path") == "fast-clef" for call in decisions.calls)
    assert any(call[0].get("decision_path") == "cosmos-clef" for call in decisions.calls)
    assert len(vlm.observed_images) == 1
    observed_image = vlm.observed_images[0][0][0]
    described_images = vlm.calls[0][0]
    assert observed_image == described_images[0]
    assert frames.active == 0 and not agent._jobs


@pytest.mark.asyncio
async def test_fast_positive_survives_backup_service_failure():
    gate = _open()
    frames = _Frames((gate, None), (gate, None))
    vlm = _VLM()
    vlm.error = RuntimeError("Cosmos unavailable")
    decisions = _HedgedDecisions("start", fast=("yes", 1.0), backup=("yes", 1.0))
    agent, output, _ = _agent(decisions, frames, vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(job.task, 1)
    assert sum(message.text == "Alert" for message, _ in output.messages) == 1
    assert "unavailable" in "".join(message.text or "" for message, _ in output.messages).lower()
    assert frames.active == 0 and not agent._jobs


@pytest.mark.asyncio
async def test_both_paths_positive_publish_exactly_one_alert():
    gate = _open()
    frames = _Frames((gate, None), (gate, None))
    vlm = _VLM()
    decisions = _HedgedDecisions("start", fast=("yes", 1.0), backup=("yes", 1.0), synchronize=True)
    agent, output, _ = _agent(decisions, frames, vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(job.task, 1)
    assert decisions.both_ready.is_set()
    assert sum(message.text == "Alert" for message, _ in output.messages) == 1
    assert sum(message.text == "The lamp is blue." for message, _ in output.messages) == 1
    assert not agent._jobs and frames.active == 0


@pytest.mark.asyncio
async def test_direct_no_and_backup_no_leave_job_pending_until_cancelled():
    gate = asyncio.Event()
    gate.set()
    frames = _Frames((gate, None), (gate, None))
    decisions = _HedgedDecisions("start", fast=("no", 1.0), backup=("no", 1.0))
    agent, output, _ = _agent(decisions, frames)
    await _start(agent)
    await _next_call(frames)
    await _next_call(frames)
    await asyncio.wait_for(decisions.fast_called.wait(), 1)
    await asyncio.wait_for(decisions.backup_called.wait(), 1)
    await agent.participant_left("alice")
    assert not any(message.text == "Alert" for message, _ in output.messages)
    assert frames.active == 0 and not agent._jobs


@pytest.mark.asyncio
async def test_alert_is_published_before_sibling_cancellation_cleanup_finishes():
    gate = _open()
    frames = _Frames((gate, None), (gate, None))
    vlm = _VLM()
    vlm.block_observation = True
    fast_gate = asyncio.Event()
    decisions = _HedgedDecisions("start", fast=("yes", 1.0), backup=("no", 1.0), fast_gate=fast_gate)
    agent, output, _ = _agent(decisions, frames, vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(vlm.observation_started.wait(), 1)
    fast_gate.set()
    await asyncio.wait_for(vlm.observation_cleanup.wait(), 1)
    assert any(message.text == "Alert" for message, _ in output.messages)
    assert not job.task.done()
    vlm.release_observation_cleanup.set()
    await asyncio.wait_for(job.task, 1)
    assert not agent._jobs and frames.active == 0


@pytest.mark.asyncio
async def test_late_yes_after_replacement_cannot_claim_or_publish():
    gate = _open()
    frames = _Frames((gate, None), (gate, None), (gate, None), (gate, None))
    fast_gate = asyncio.Event()
    decisions = _HedgedDecisions(
        "start", "start", fast=("yes", 1.0), backup=("no", 1.0), fast_gate=fast_gate, ignore_fast_cancel=True
    )
    agent, output, _ = _agent(decisions, frames)
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
