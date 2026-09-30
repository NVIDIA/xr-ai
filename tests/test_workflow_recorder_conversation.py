# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOP commands, focus and silence through the shared conversation path."""

import asyncio
import json
import sys
import time
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


@pytest.mark.parametrize("shutdown", ["exit", "cancel"])
async def test_app_finalizes_recording_before_voice_transport_closes(monkeypatch, tmp_path, shutdown):
    from device_io_hub.capture._service import CaptureService
    from device_io_hub.capture.config import CaptureConfig
    from workflow_recorder_worker import app
    from workflow_recorder_worker.config import load_config
    from xr_ai_hub._capture import CAPTURE_STOP_TOPIC
    from xr_ai_voice import _session as session_module
    from xr_ai_voice._types import VoiceQuery

    config = replace(
        load_config(_SAMPLE / "yaml/workflow_recorder_worker.yaml"),
        artifacts_dir=tmp_path / "artifacts", guides_dir=tmp_path / "guides",
        media_capture_dir=tmp_path / "captures",
    )
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", SimpleNamespace())
    capture = CaptureService(CaptureConfig(
        out_dir=str(config.media_capture_dir), session_mode="explicit", max_total_bytes=0,
    ))
    commands = []
    closed = False

    async def send(message):
        if closed:
            raise RuntimeError("hub endpoint is closed")
        commands.append(message.topic)
        await capture._on_agent_data(message)

    def close_transport():
        nonlocal closed
        closed = True

    transport = Mock(
        endpoint=Mock(send_return_data=AsyncMock(side_effect=send), mark_ready=AsyncMock()),
        send_return_data=AsyncMock(side_effect=send),
        shutdown=Mock(side_effect=close_transport), wait_until_started=AsyncMock(),
    )
    monkeypatch.setattr(app, "HubVoiceTransport", Mock(return_value=transport))
    monkeypatch.setattr(app, "setup_logging", Mock())
    for factory in ("make_llm", "make_stt", "make_tts", "make_vlm"):
        monkeypatch.setattr(app, factory, Mock(return_value=Mock(close=AsyncMock())))
    async def wait_for_frame(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(app, "CurrentFrameTool", Mock(return_value=Mock(execute=AsyncMock(
        side_effect=wait_for_frame,
    ))))
    # Bound a regression's missing-manifest wait; successful finalization does
    # not rely on a shortened timeout or capture-service shutdown.
    monkeypatch.setattr("workflow_recorder_worker.recorder._MEDIA_FINALIZE_TIMEOUT_S", 0.2)
    recorded = asyncio.Event()
    finish = asyncio.Event()
    callbacks = {}

    class Pipeline:
        async def cancel(self):
            finish.set()

    def build_pipeline(**kwargs):
        callbacks.update(kwargs)
        return object(), Pipeline()

    class Runner:
        async def run(self, pipeline):
            io = callbacks["io_processor"]
            await io._on_participant_joined("user")
            voice = io._input_sink.__self__
            await asyncio.gather(*voice._lifecycle_tasks)
            # Drive the real VoiceAgent input/transcript bridge. No departure
            # event occurs: only this worker is shutting down.
            await io._input_sink(VoiceQuery(participant_id="user", text="start recording", timestamp_us=1))
            await callbacks["on_final_transcript"]("user", "Attach a wheel", time.time_ns() // 1000)
            # Establish captured narration before requesting worker shutdown.
            await voice._transcript_queue.join()
            recorded.set()
            await finish.wait()

    monkeypatch.setattr(session_module, "_build_voice_pipeline", build_pipeline)
    monkeypatch.setattr(session_module, "PipelineRunner", Runner)
    # The pipeline is mocked, but the application, VoiceAgent.run, and session
    # cleanup are real, including the transport close that caused the defect.
    monkeypatch.setattr(session_module._VoiceIOProcessor, "enqueue_response", AsyncMock())
    task = asyncio.create_task(app.run_app(config), context=nemo_relay.fork_asyncio_context())
    try:
        async with asyncio.timeout(2):
            while not recorded.is_set():
                if task.done():
                    await task
                await asyncio.sleep(0.001)
        if shutdown == "cancel":
            task.cancel()
        else:
            finish.set()
        result, = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
        assert isinstance(result, asyncio.CancelledError) if shutdown == "cancel" else result is None
        packet_path, = (config.artifacts_dir / "sessions").glob("*/packet.json")
        packet = json.loads(packet_path.read_text())
        assert packet["status"] == "complete"
        assert packet["media_capture"]["control_status"] == "complete"
        assert packet["narration_status"] == "complete"
        assert packet["counts"]["transcripts"] == 1
        rows = [json.loads(row) for row in (packet_path.parent / "transcript.jsonl").read_text().splitlines()]
        assert [row["text"] for row in rows] == ["Attach a wheel"]
        assert commands.count(CAPTURE_STOP_TOPIC) == 1
        assert closed
    finally:
        finish.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await capture.stop()


@pytest.mark.parametrize("transition", [None, "next", "skip", "stop guide", "restart guide", "finish"])
async def test_observation_completion_notices_do_not_survive_guide_transitions(demo, transition):
    guide = yaml.safe_load((_SAMPLE / "skills/recording-to-guide/references/example.guide.yaml").read_text())
    guide["task"]["status"] = "approved"
    step = guide["steps"][0]
    step["evidence"].pop("commit")
    step["evidence"]["consecutive"] = 1
    if transition == "finish":
        step["next"] = None
        guide["steps"] = [step]
    demo.catalog._guides_dir.mkdir(parents=True, exist_ok=True)
    (demo.catalog._guides_dir / "test.guide.yaml").write_text(yaml.safe_dump(guide))
    await demo.catalog._scan()
    await demo.engine._start("user", guide["task"]["id"])
    session = demo.engine._sessions["user"]
    demo.engine._trigger = AsyncMock(return_value=(True, "workspace clear"))
    committed, resume = asyncio.Event(), asyncio.Event()

    async def model_reply(*args, **kwargs):
        if not session.notices:
            return ChatResponse(content="", reasoning=None, finish_reason="tool_calls", raw={}, tool_calls=[
                ToolCall(id="commit", name="workflow__commit", arguments=json.dumps({
                    "updates": {"workspace_clear": True},
                })),
            ])
        committed.set()
        await resume.wait()
        return ChatResponse(content="Done.", reasoning=None, finish_reason="stop", raw={}, tool_calls=None)

    demo.engine._llm.chat.side_effect = model_reply
    tick = asyncio.create_task(demo.engine._tick(session), context=nemo_relay.fork_asyncio_context())
    try:
        await asyncio.wait_for(committed.wait(), 2)
        assert session.step.is_complete(session.state)
        if transition:
            await demo.engine._route("next" if transition == "finish" else transition, "user")
        resume.set()
        await asyncio.wait_for(tick, 2)
        assert (step["messages"]["complete"] in [text for _, text in demo.speech]) is (transition is None)
        assert not session.notices
    finally:
        resume.set()
        await asyncio.gather(tick, return_exceptions=True)


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

        async def run(self, runtime, *, before_close):
            assert self.options["query_topic"] is FRONT_END_QUERY_TOPIC
            assert self.options["interrupted_topic"] is INTERRUPTED_TOPIC
            assert self.options["stop_ack_enabled"]("user") is True
            assert self.options["interrupt_on_supersede"] is True
            assert "conversation" in runtime._agents
            assert "voice-aggregation" in runtime._agents
            await before_close()

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

    @pytest.mark.parametrize("cleanup", ["conversation", "engine", "aggregation"])
    async def test_departure_cannot_restart_a_suspended_dispatch(self, demo, monkeypatch, cleanup):
        await self.connect(demo)
        entered, resume = asyncio.Event(), asyncio.Event()
        owner, name = {
            "conversation": (demo.front, "interrupted"),
            "engine": (demo.engine, "interrupted"),
            "aggregation": (demo.aggregation, "release"),
        }[cleanup]
        original = getattr(owner, name)
        calls = 0

        async def blocked_cleanup(participant_id):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await resume.wait()
            await original(participant_id)

        monkeypatch.setattr(owner, name, blocked_cleanup)
        dispatch = asyncio.create_task(
            demo.publish(FRONT_END_QUERY_TOPIC, UserQuery(text="Hello", timestamp_us=1)),
            context=nemo_relay.fork_asyncio_context(),
        )
        await asyncio.wait_for(entered.wait(), 2)
        departure = asyncio.create_task(
            demo.publish(PARTICIPANT_LEFT_TOPIC, VoiceParticipantLeft()), context=nemo_relay.fork_asyncio_context(),
        )
        try:
            async with asyncio.timeout(2):
                while demo.engine.is_connected("user"):
                    await asyncio.sleep(0)
        finally:
            resume.set()
            await asyncio.wait_for(asyncio.gather(dispatch, departure), 2)
        # Drain a wrongly-created task too, so the old implementation fails
        # deterministically rather than escaping into fixture teardown.
        if task := demo.front._tasks.get("user"):
            await task
        demo.llm.chat.assert_not_awaited()
        assert "user" not in demo.front._tasks
        assert not demo.front._pending and not demo.front._history and not demo.front._spoken
        await self.connect(demo)
        await demo.say("Hello after reconnect")
        demo.llm.chat.assert_awaited_once()

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
