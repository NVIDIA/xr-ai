# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit recording boundaries and participant-local speech controls."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import nemo_relay
import pytest
import pytest_asyncio
from xr_ai_models import load_models_config
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    VOICE_TRANSCRIPT_TOPIC,
    UserQuery,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
    VoiceTranscript,
)

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/workflow-recorder"
sys.path.insert(0, str(_SAMPLE / "worker"))

from workflow_recorder_worker._workflow_engine import SopEngineAgent  # noqa: E402
from workflow_recorder_worker.catalog import GuideCatalog  # noqa: E402
from workflow_recorder_worker.events import (  # noqa: E402
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    RECORDING_COMMAND,
    USER_QUERY_TOPIC,
)
from workflow_recorder_worker.recorder import RecorderAgent  # noqa: E402


def test_sample_model_config_resolves_presets():
    load_models_config(_SAMPLE / "yaml/models.json")


class _Speech(Agent):
    def __init__(self):
        super().__init__()
        self.messages = []

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def output(self, message: VoiceOutput, ctx: RuntimeContext):
        self.messages.append((ctx.metadata.participant_id, message.text))


@pytest_asyncio.fixture
async def demo(tmp_path):
    catalog = GuideCatalog(tmp_path / "guides", tmp_path / "index.json", interval_s=1)
    frame = Mock()

    async def wait_for_frame(*_args, **_kwargs):
        await asyncio.Event().wait()

    frame.execute = AsyncMock(side_effect=wait_for_frame)
    recorder = RecorderAgent(
        sessions_dir=tmp_path / "sessions",
        current_frame=frame,
        images=Mock(),
        query_image=Mock(),
        guide_catalog=catalog,
        capture_fps=2,
        caption_interval_s=5,
    )
    engine = SopEngineAgent(
        catalog=catalog,
        llm=Mock(chat=AsyncMock()),
        current_frame=frame,
        image_query=Mock(),
        vision_timeout_s=1,
        recorder=recorder,
    )
    speech = _Speech()
    runtime = AgentRuntime()
    runtime.register("recorder", recorder)
    runtime.register("sop-engine", engine)
    runtime.register("speech", speech)
    engine.bind_runtime(runtime)

    async def publish(topic, event, pid="user"):
        await runtime.publish(topic, event, participant_id=pid, source="test")

    async def say(text, pid="user"):
        timestamp = time.time_ns() // 1000
        await publish(VOICE_TRANSCRIPT_TOPIC, VoiceTranscript(text=text, timestamp_us=timestamp), pid)
        await publish(USER_QUERY_TOPIC, UserQuery(text=text, timestamp_us=timestamp), pid)
        if turn := engine._turns.get(pid):
            await turn

    async with runtime:
        try:
            yield SimpleNamespace(
                recorder=recorder, engine=engine, speech=speech.messages,
                publish=publish, say=say, root=tmp_path / "sessions", frame=frame,
            )
        finally:
            await engine.stop()
            await recorder.stop()


@pytest.mark.asyncio
async def test_connect_help_once_and_no_automatic_recording(demo):
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    assert len(demo.speech) == 1
    assert "start recording" in demo.speech[0][1]
    assert "finish recording" in demo.speech[0][1]
    assert "list guides" in demo.speech[0][1]
    assert not demo.recorder.is_recording("user")
    assert not list(demo.root.iterdir())
    demo.frame.execute.assert_not_called()
    await demo.say("finish recording")
    assert len(demo.speech) == 1
    for text in ("hello", "What can you do?", "put the block in the box"):
        await demo.say(text)
        assert demo.speech[-1][1] == "Say list guides, or say start guide followed by a guide ID."
    assert len(demo.speech) == 4
    assert demo.speech.count(demo.speech[0]) == 1
    assert not list(demo.root.iterdir())
    await demo.say("list guides")
    assert demo.speech[-1][1] == "No valid guides are available yet."


@pytest.mark.asyncio
async def test_multiple_recordings_are_silent_and_finalize_separate_packets(demo):
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    directories = []
    for index in range(2):
        before = len(demo.speech)
        await demo.say("Start recording.")
        assert demo.speech[-1] == ("user", "Recording started.")
        state = demo.recorder._sessions["user"]
        directories.append(state.directory)
        tasks = tuple(state.tasks)
        await demo.say("start recording")
        assert demo.recorder._sessions["user"] is state
        narration = [f"Attach wheel {index + 1}", "list guides", "next", "stop guide"]
        for text in narration:
            await demo.say(text)
        assert len(demo.speech) == before + 1
        assert "user" not in demo.engine._monitors
        await demo.say("Finish recording!")
        assert not demo.recorder.is_recording("user")
        assert all(task.done() for task in tasks)
        packet = json.loads((state.directory / "packet.json").read_text())
        assert packet["status"] == "complete"
        assert packet["ended_at"] is not None
        assert packet["counts"]["transcripts"] == len(narration)
        rows = [json.loads(line) for line in (state.directory / "transcript.jsonl").read_text().splitlines()]
        assert [row["text"] for row in rows] == narration
        assert (state.directory / "summary.md").exists()
        assert len(demo.speech) == before + 2
        assert demo.speech[-1] == ("user", "Recording ended. " + demo.speech[0][1])
        await demo.say("finish recording")
        assert len(demo.speech) == before + 2
        await demo.say("some idle speech")
        assert len(demo.speech) == before + 3
        assert demo.speech[-1][1] == "Say list guides, or say start guide followed by a guide ID."
    assert directories[0] != directories[1]


@pytest.mark.asyncio
async def test_disconnect_finalizes_without_help_and_reconnect_is_idle(demo):
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    await demo.say("start recording")
    state = demo.recorder._sessions["user"]
    await demo.publish(PARTICIPANT_LEFT_TOPIC, VoiceParticipantLeft())
    assert json.loads((state.directory / "packet.json").read_text())["status"] == "complete"
    assert len(demo.speech) == 2
    await demo.say("start recording")
    assert not demo.recorder.is_recording("user")
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    assert len(demo.speech) == 3
    assert not demo.recorder.is_recording("user")


@pytest.mark.asyncio
async def test_recording_pauses_guide_without_changing_guide_state(demo):
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    pinned = object()
    demo.engine._sessions["user"] = pinned
    demo.engine._tick = AsyncMock()
    demo.engine._answer_step_question = AsyncMock(return_value="Current step answer.")
    pending = asyncio.create_task(asyncio.Event().wait())
    demo.engine._turns["user"] = pending
    await demo.say("start recording")
    assert pending.cancelled()
    await demo.say("skip")
    assert demo.engine._sessions["user"] is pinned
    assert len(demo.speech) == 2
    await demo.say("finish recording")
    assert "user" in demo.engine._monitors
    assert demo.engine._sessions["user"] is pinned
    await demo.say("What should I do?")
    demo.engine._answer_step_question.assert_awaited_once_with(pinned, "What should I do?")
    assert demo.speech[-1][1] == "Current step answer."


@pytest.mark.asyncio
async def test_recording_is_participant_local(demo):
    for pid in ("user", "other"):
        await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), pid)
    await demo.say("start recording", "user")
    await demo.say("finish recording", "other")
    assert demo.recorder.is_recording("user")
    assert not demo.recorder.is_recording("other")
    await demo.say("list guides", "other")
    assert demo.speech[-1] == ("other", "No valid guides are available yet.")


@pytest.mark.asyncio
async def test_idle_hint_is_spoken_before_and_after_but_not_during_recording(demo):
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    await demo.say("What can you do?")
    assert demo.speech[-1][1] == "Say list guides, or say start guide followed by a guide ID."
    assert len(demo.speech) == 2
    await demo.say("start recording")
    for question in ("What can you do?", "Describe what you see."):
        await demo.say(question)
    assert len(demo.speech) == 3
    assert demo.speech[-1] == ("user", "Recording started.")
    demo.engine._llm.chat.assert_not_awaited()
    await demo.say("finish recording")
    await demo.say("What can you do?")
    demo.engine._llm.chat.assert_not_awaited()
    assert demo.speech[-1][1] == "Say list guides, or say start guide followed by a guide ID."
    assert len(demo.speech) == 5
    assert not demo.engine._sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["What can you do?", "Describe what you see.", "What is two plus two?"])
async def test_idle_questions_preserve_original_capabilities(demo, question):
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    await demo.say(question)
    assert demo.speech[-1][1] == "Say list guides, or say start guide followed by a guide ID."
    demo.engine._llm.chat.assert_not_awaited()
    demo.frame.execute.assert_not_awaited()
    demo.engine._image_query.execute.assert_not_called()
    assert not list(demo.root.iterdir())


@pytest.mark.asyncio
async def test_concurrent_start_and_disconnect_do_not_leave_a_recording(demo):
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    await asyncio.gather(
        asyncio.create_task(demo.say("start recording"), context=nemo_relay.fork_asyncio_context()),
        asyncio.create_task(
            demo.publish(PARTICIPANT_LEFT_TOPIC, VoiceParticipantLeft()),
            context=nemo_relay.fork_asyncio_context(),
        ),
    )
    assert not demo.recorder.is_recording("user")
    assert "user" not in demo.engine._monitors
    assert demo.speech[1:] in ([], [("user", "Recording started.")])


@pytest.mark.asyncio
async def test_delayed_control_transcript_is_not_recorded(demo):
    await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
    await demo.publish(USER_QUERY_TOPIC, UserQuery(text="start recording", timestamp_us=1))
    state = demo.recorder._sessions["user"]
    await demo.publish(VOICE_TRANSCRIPT_TOPIC, VoiceTranscript(text="start recording", timestamp_us=1))
    assert state.transcript_count == 0
    await demo.say("finish recording")


@pytest.mark.parametrize(
    "text", [
        "start workflow lego", "end recording", "then finish recording", "finish recording after this",
        "start workflow", "finish workflow",
    ],
)
def test_recording_commands_do_not_match_guide_selectors_or_narration(text):
    assert RECORDING_COMMAND.fullmatch(text) is None
