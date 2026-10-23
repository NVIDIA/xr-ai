# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot alert ownership and output boundaries; scripted decisions are not model evaluations."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from test_conversation_example_agents import _Decisions, _Frames, _next_call
from xr_ai_hub import FrameUnavailable
from xr_ai_routing import AgentTurn
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
        if message.text and (message.response_id or "").endswith(":description"):
            self.description.set()
        if (message.response_id or "").startswith("alert:"):
            self.background.set()
            if self.fail_background:
                raise RuntimeError("output unavailable")


class _VLM:
    def __init__(self):
        self.calls = []
        self.entered = asyncio.Event()
        self.resume = None
        self.closed = asyncio.Event()
        self.error = None

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
        return b"scripted image input, not submitted to a real model"


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


async def _start(agent, participant="alice", request="Alert me when the lamp is blue.", turn_id="start"):
    turn = AgentTurn(participant, turn_id, request)
    assert (await agent.admit(turn, focused=False)).disposition == "accept"

    async def started():
        return True

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
    decisions = _Decisions("start", "no", "yes")
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
    assert notice.final and notice.interrupt
    chunks = [message for message, _ in output.messages if message.response_id == notice.response_id + ":description"]
    assert chunks[0].text == "The lamp is blue." and not chunks[0].final
    assert chunks[-1].final and not chunks[-1].text
    assert all(not chunk.interrupt for chunk in chunks)
    assert alerts[0][1] == {"participant_id": "alice", "source": "alerts"}
    assert len(frames.calls) == len(images.resolved) == 2
    assert len(capabilities) == 1
    assert "lamp" not in capabilities[0][1]
    assert not agent._jobs
    assert frames.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["frame", "media"])
async def test_unavailable_evidence_stops_without_alert_or_condition_decision(failure):
    gate = _open()
    frames = _Frames((gate, FrameUnavailable("no frame") if failure == "frame" else None))
    images = _Images(unavailable=failure == "media")
    decisions = _Decisions("start")
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
    agent, output, _ = _agent(_Decisions("start", "start", "decline", "cancel"), frames)
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
    agent, _, _ = _agent(_Decisions("start"), _Frames())
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
        _Decisions("start", "yes"),
        _Frames((gate, None)),
        output=_Output(fail_background=True),
    )
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.background.wait(), 1)
    await asyncio.wait_for(job.task, 1)
    assert not agent._jobs
    await agent.stop()
    assert len([message for message, _ in output.messages if (message.response_id or "").startswith("alert:")]) == 1
    assert not agent._jobs


@pytest.mark.asyncio
async def test_cancel_drains_followup_after_notice_without_more_spoken_output():
    vlm = _VLM()
    vlm.resume = asyncio.Event()
    agent, output, _ = _agent(_Decisions("start", "yes"), _Frames((_open(), None)), vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(output.description.wait(), 1)
    assert agent._jobs["alice"] is job
    assert [message.text for message, _ in output.messages][-2:] == ["Alert", "The lamp is blue."]
    await agent.participant_left("alice")
    await asyncio.wait_for(vlm.closed.wait(), 1)
    assert job.task.done() and not agent._jobs
    assert output.messages[-1][0].final and not output.messages[-1][0].text


@pytest.mark.asyncio
async def test_description_service_failure_does_not_repeat_the_initial_alert():
    vlm = _VLM()
    vlm.error = RuntimeError("description unavailable")
    agent, output, _ = _agent(_Decisions("start", "yes"), _Frames((_open(), None)), vlm=vlm)
    await _start(agent)
    job = agent._jobs["alice"]
    await asyncio.wait_for(job.task, 1)
    messages = [message for message, _ in output.messages]
    assert sum(message.text == "Alert" for message in messages) == 1
    assert "unavailable" in "".join(message.text or "" for message in messages).lower()
    assert messages[-1].final and not agent._jobs
