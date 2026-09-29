# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOP commands, focus and silence through the shared conversation path."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import nemo_relay
import pytest
import yaml
from test_workflow_recorder_controls import _SAMPLE
from test_workflow_recorder_controls import demo as demo
from workflow_recorder_worker.events import INTERRUPTED_TOPIC, PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC
from xr_ai_models import ChatResponse, ToolCall
from xr_ai_runtime import Agent
from xr_ai_sample_agents.front_end import FRONT_END_QUERY_TOPIC
from xr_ai_tools.image import ImageReference
from xr_ai_voice import UserQuery, VoiceInterrupted, VoiceParticipantJoined, VoiceParticipantLeft


async def test_app_registers_shared_front_end_and_preserves_recording_voice_hook(monkeypatch, tmp_path):
    from workflow_recorder_worker import app
    from workflow_recorder_worker.config import load_config

    config = replace(
        load_config(_SAMPLE / "yaml/workflow_recorder_worker.yaml"),
        artifacts_dir=tmp_path / "artifacts", guides_dir=tmp_path / "guides",
    )
    for factory in ("make_llm", "make_stt", "make_tts", "make_vlm"):
        monkeypatch.setattr(app, factory, Mock(return_value=Mock()))
    monkeypatch.setattr(app, "setup_logging", Mock())
    monkeypatch.setattr(app, "HubVoiceTransport", Mock())

    class Voice(Agent):
        def __init__(self, **kwargs):
            super().__init__()
            self.options = kwargs

        async def run(self, runtime):
            assert self.options["query_topic"] is FRONT_END_QUERY_TOPIC
            assert self.options["interrupted_topic"] is INTERRUPTED_TOPIC
            assert self.options["stop_ack_enabled"]("user") is True
            assert self.options["interrupt_on_supersede"] is True
            assert "conversation" in runtime._agents
            assert "voice-aggregation" in runtime._agents

    monkeypatch.setattr(app, "VoiceAgent", Voice)
    await app.run_app(config)


@pytest.mark.parametrize("demo", [True], indirect=True)
class TestConversationIntegration:
    async def connect(self, demo, pid="user"):
        await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), pid)
        await asyncio.sleep(0.01)

    async def test_idle_chat_and_command_bypass(self, demo):
        await self.connect(demo)
        await demo.say("What is two plus two?")
        assert demo.speech[-1] == ("user", "An ordinary answer.")
        assert demo.front._history["user"][-1].user == "What is two plus two?"
        demo.frame.execute.assert_not_awaited()
        assert demo.llm.chat.call_args.kwargs["enable_thinking"] is False
        demo.llm.chat.reset_mock()
        await demo.say("list guides")
        assert demo.speech[-1][1] == "No valid guides are available yet."
        demo.llm.chat.assert_not_awaited()

    async def test_idle_app_delegation_does_not_invent_recording_commands(self, demo):
        await self.connect(demo)
        demo.llm.chat.return_value = ChatResponse(
            content="", reasoning=None, tool_calls=[ToolCall(id="sop", name="sop_guide", arguments="{}")],
            finish_reason="tool_calls", raw={},
        )
        await demo.say("How do I record a demonstration?")
        assert not demo.recorder.is_recording("user")
        assert "start guide" in demo.speech[-1][1]
        assert demo.front._history["user"][-1].user == "How do I record a demonstration?"

    async def test_idle_visual_answer_uses_fresh_frame_and_shared_stream(self, demo):
        await self.connect(demo)
        demo.llm.chat.return_value = ChatResponse(
            content="", reasoning=None, tool_calls=[ToolCall(id="view", name="current_view", arguments="{}")],
            finish_reason="tool_calls", raw={},
        )
        demo.frame.execute.side_effect = None
        demo.frame.execute.return_value = SimpleNamespace(image=ImageReference(uri="xr-image://fake"))
        await demo.say("What color is this cup?")
        demo.frame.execute.assert_awaited_once()
        assert demo.frame.execute.call_args.args[0].participant_id == "user"
        assert demo.front._history["user"][-1].assistant == "A blue cup."
        assert all(output.kind == "result" for output in demo.outputs)
        assert demo.outputs[-2].response_id == demo.outputs[-1].response_id
        assert demo.outputs[-1].final
        assert demo.outputs[-1].turn_id is not None

    async def test_repeated_recordings_remain_silent_and_keep_narration(self, demo):
        await self.connect(demo)
        await demo.say("Hello")
        history = tuple(demo.front._history["user"])
        demo.llm.chat.reset_mock()
        directories = []
        for _ in range(2):
            await demo.say("start recording")
            assert demo.speech[-1][1] == "Recording started."
            state = demo.recorder._sessions["user"]
            directories.append(state.directory)
            count = len(demo.speech)
            narration = ["Fit the axle", "What do you see?", "next", "stop guide"]
            for text in narration:
                await demo.say(text)
            assert len(demo.speech) == count
            assert demo.engine.has_focus("user")
            await demo.say("stop recording")
            assert demo.speech[-1][1].startswith("Recording ended.")
            assert not demo.engine.has_focus("user")
            rows = [json.loads(line) for line in (state.directory / "transcript.jsonl").read_text().splitlines()]
            assert [row["text"] for row in rows] == narration
        assert directories[0] != directories[1]
        assert tuple(demo.front._history["user"]) == history
        demo.llm.chat.assert_not_awaited()

    async def test_guide_focus_and_approval_preserve_execution(self, demo):
        await self.connect(demo)
        guide = yaml.safe_load((_SAMPLE / "skills/recording-to-guide/references/example.guide.yaml").read_text())
        directory = demo.catalog._guides_dir
        directory.mkdir()
        path = directory / "test.guide.yaml"
        path.write_text(yaml.safe_dump(guide))
        await demo.catalog._scan()
        await demo.say(f"start guide {guide['task']['id']}")
        assert not demo.engine.has_focus("user")
        assert "draft" in demo.speech[-1][1]
        guide["task"]["status"] = "approved"
        path.write_text(yaml.safe_dump(guide))
        await demo.catalog._scan()
        demo.engine._tick = AsyncMock()
        await demo.say(f"start guide {guide['task']['id']}")
        session = demo.engine._sessions["user"]
        assert demo.engine.has_focus("user")
        await demo.say("next")
        assert demo.engine._sessions["user"].step_id == session.workflow.start_step
        assert "not complete" in demo.speech[-1][1]
        demo.engine._answer_step_question = AsyncMock(return_value="Follow this guide step.")
        await demo.say("What do you see?")
        demo.engine._answer_step_question.assert_awaited_once_with(session, "What do you see?")
        assert demo.front._history["user"][-1].assistant == "Follow this guide step."
        demo.llm.chat.assert_not_awaited()
        await demo.say("start recording")
        await demo.say("stop recording")
        assert demo.engine._sessions["user"] is session
        assert demo.engine.has_focus("user")
        await demo.say("stop guide")
        assert not demo.engine.has_focus("user")
        await demo.say("Hello again")
        demo.llm.chat.assert_awaited_once()

    async def test_start_recording_cancels_inflight_generic_turn(self, demo):
        await self.connect(demo)
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def blocked_decision(*_args, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        demo.llm.chat.side_effect = blocked_decision
        await demo.publish(FRONT_END_QUERY_TOPIC, UserQuery(text="What do you see?", timestamp_us=1))
        await asyncio.wait_for(entered.wait(), 2)
        await demo.say("start recording")
        assert cancelled.is_set()
        assert demo.speech[-1][1] == "Recording started."
        assert "user" not in demo.front._history

    async def test_interruption_and_departure_are_participant_local(self, demo):
        for pid in ("user", "other"):
            await self.connect(demo, pid)
            await demo.say("Hello", pid)
        await demo.say("start recording")
        await demo.publish(INTERRUPTED_TOPIC, VoiceInterrupted())
        assert demo.recorder.is_recording("user")
        await demo.say("Hello again", "other")
        assert len(demo.front._history["other"]) == 2
        await demo.publish(PARTICIPANT_LEFT_TOPIC, VoiceParticipantLeft())
        assert not demo.recorder.is_recording("user")
        assert "user" not in demo.front._history
        assert len(demo.front._history["other"]) == 2
        await self.connect(demo)
        await demo.say("Hello")
        assert len(demo.front._history["user"]) == 1

    async def test_stop_finalization_survives_interruption_and_serializes_restart(self, demo, monkeypatch):
        await self.connect(demo)
        await demo.say("start recording")
        original = demo.recorder._wait_for_media
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked_wait(state):
            entered.set()
            await release.wait()
            await original(state)

        monkeypatch.setattr(demo.recorder, "_wait_for_media", blocked_wait)
        stop = asyncio.create_task(
            demo.publish(FRONT_END_QUERY_TOPIC, UserQuery(text="stop recording", timestamp_us=1)),
            context=nemo_relay.fork_asyncio_context(),
        )
        await asyncio.wait_for(entered.wait(), 2)
        await demo.publish(INTERRUPTED_TOPIC, VoiceInterrupted())
        start = asyncio.create_task(
            demo.publish(FRONT_END_QUERY_TOPIC, UserQuery(text="start recording", timestamp_us=2)),
            context=nemo_relay.fork_asyncio_context(),
        )
        try:
            await asyncio.sleep(0.01)
            assert not stop.done() and not start.done()
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(stop, start), 2)
        assert demo.recorder.is_recording("user")
