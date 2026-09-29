# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit recording boundaries and participant-local speech controls."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import nemo_relay
import pytest
import pytest_asyncio
from PIL import Image
from xr_ai_hub import DataMessage
from xr_ai_hub._capture import CAPTURE_START_TOPIC, CAPTURE_STOP_TOPIC, CAPTURE_STT_TOPIC, CAPTURE_TTS_TOPIC
from xr_ai_models import load_models_config
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
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
from workflow_recorder_worker.config import load_config  # noqa: E402
from workflow_recorder_worker.events import (  # noqa: E402
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    RECORDING_COMMAND,
    USER_QUERY_TOPIC,
)
from workflow_recorder_worker.recorder import RecorderAgent, _export_narration  # noqa: E402


def test_sample_model_config_resolves_presets():
    load_models_config(_SAMPLE / "yaml/models.json")


def test_sample_uses_shared_capture_configuration():
    from device_io_hub.capture.config import load_capture_config

    worker = load_config(_SAMPLE / "yaml/workflow_recorder_worker.yaml")
    capture = load_capture_config(_SAMPLE / "yaml/media_capture.yaml")
    assert worker.media_capture_dir == Path(capture.out_dir)
    assert capture.session_mode == "explicit"
    assert capture.profile == "raw"
    assert capture.max_total_bytes == 0
    assert worker.capture_fps == 2
    assert worker.caption_interval_s == 5


def test_narration_projection_preserves_schema_timestamps_and_source(tmp_path):
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "speech.jsonl"
    destination = tmp_path / "sop.jsonl"
    manifest.write_text(json.dumps({"transcript": source.name}))
    rows = [
        {"source": "agent", "pts_us": 1, "text": "Recording started."},
        {"source": "user", "pts_us": 2, "text": " Start recording! "},
        {"source": "user", "pts_us": 1_234_567, "text": "  Place the café cup.  "},
        {"source": "user", "pts_us": 1_234_568, "text": "then stop recording"},
        {"source": "user", "pts_us": 1_234_569, "text": "Place the café cup."},
        {"source": "user", "pts_us": 3_000_000, "text": "STOP RECORDING."},
        {"source": "user", "pts_us": 4_000_000, "text": "  "},
    ]
    original = "\n".join(json.dumps(row) for row in rows) + "\n"
    source.write_text(original)
    assert _export_narration(manifest, destination) == 3
    exported = [json.loads(line) for line in destination.read_text().splitlines()]
    assert exported[0] == {
        "transcript_id": 1, "timestamp_us": 1_234_567,
        "timestamp": "1970-01-01T00:00:01.234567+00:00", "text": "Place the café cup.",
    }
    assert [row["transcript_id"] for row in exported] == [1, 2, 3]
    assert [row["text"] for row in exported] == ["Place the café cup.", "then stop recording", "Place the café cup."]
    assert source.read_text() == original
    # Re-export replaces the derived view, rather than duplicating narration.
    assert _export_narration(manifest, destination) == 3
    assert len(destination.read_text().splitlines()) == 3


@pytest.mark.parametrize("content", [
    "not json\n",
    '{"source":"user","text":"missing timestamp"}\n',
    '{"source":"user","text":"bad timestamp","pts_us":true}\n',
])
def test_invalid_narration_does_not_replace_existing_export(tmp_path, content):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"transcript": "speech.jsonl"}))
    (tmp_path / "speech.jsonl").write_text(content)
    destination = tmp_path / "sop.jsonl"
    destination.write_text("previous export")
    with pytest.raises((ValueError, KeyError)):
        _export_narration(manifest, destination)
    assert destination.read_text() == "previous export"


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
    targets = {}

    async def capture_command(message):
        if message.topic == CAPTURE_START_TOPIC:
            targets[message.participant_id] = json.loads(message.data)["target"]
            bundle = tmp_path / "captures" / targets[message.participant_id] / "bundle"
            bundle.mkdir(parents=True)
            (bundle / "transcript.jsonl").touch()
        elif message.topic == CAPTURE_STOP_TOPIC:
            bundle = tmp_path / "captures" / targets.pop(message.participant_id) / "bundle"
            (bundle / "manifest.json").write_text(json.dumps({
                "complete": True, "transcript": "transcript.jsonl",
            }))
        elif message.topic in (CAPTURE_STT_TOPIC, CAPTURE_TTS_TOPIC) and message.participant_id in targets:
            bundle = tmp_path / "captures" / targets[message.participant_id] / "bundle"
            source = "user" if message.topic == CAPTURE_STT_TOPIC else "agent"
            with (bundle / "transcript.jsonl").open("a") as stream:
                stream.write(json.dumps({
                    "source": source, "pts_us": message.pts_us, "text": message.data.decode(),
                }) + "\n")

    capture_endpoint = Mock(send_return_data=AsyncMock(side_effect=capture_command))
    recorder = RecorderAgent(
        sessions_dir=tmp_path / "sessions",
        capture_endpoint=capture_endpoint,
        media_capture_dir=tmp_path / "captures",
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
        # VoiceAgent forwards the final STT to main's capture before publishing
        # the runtime transcript. The SOP recorder no longer subscribes to it.
        await capture_endpoint.send_return_data(DataMessage(pid, CAPTURE_STT_TOPIC, timestamp, text.encode()))
        await publish(VOICE_TRANSCRIPT_TOPIC, VoiceTranscript(text=text, timestamp_us=timestamp), pid)
        await publish(USER_QUERY_TOPIC, UserQuery(text=text, timestamp_us=timestamp), pid)
        if turn := engine._turns.get(pid):
            await turn

    async with runtime:
        try:
            yield SimpleNamespace(
                recorder=recorder, engine=engine, speech=speech.messages,
                publish=publish, say=say, root=tmp_path / "sessions", frame=frame,
                capture_endpoint=capture_endpoint,
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
    assert "stop recording" in demo.speech[0][1]
    assert "list guides" in demo.speech[0][1]
    assert not demo.recorder.is_recording("user")
    assert not list(demo.root.iterdir())
    demo.frame.execute.assert_not_called()
    demo.capture_endpoint.send_return_data.assert_not_awaited()
    await demo.say("stop recording")
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
        assert not (state.directory / "transcript.jsonl").exists()
        assert state.transcript_count == 0
        assert len(demo.speech) == before + 1
        assert "user" not in demo.engine._monitors
        await demo.say("Stop recording!")
        assert not demo.recorder.is_recording("user")
        assert all(task.done() for task in tasks)
        packet = json.loads((state.directory / "packet.json").read_text())
        assert packet["status"] == "complete"
        assert packet["ended_at"] is not None
        assert packet["counts"]["transcripts"] == len(narration)
        assert packet["media_capture"]["control_status"] == "complete"
        assert Path(packet["media_capture"]["manifest"]).is_file()
        assert Path(packet["media_capture"]["directory"]).name == state.session_id
        rows = [json.loads(line) for line in (state.directory / "transcript.jsonl").read_text().splitlines()]
        assert [row["text"] for row in rows] == narration
        assert (state.directory / "summary.md").exists()
        assert len(demo.speech) == before + 2
        assert demo.speech[-1] == ("user", "Recording ended. " + demo.speech[0][1])
        await demo.say("stop recording")
        assert len(demo.speech) == before + 2
        await demo.say("some idle speech")
        assert len(demo.speech) == before + 3
        assert demo.speech[-1][1] == "Say list guides, or say start guide followed by a guide ID."
    assert directories[0] != directories[1]
    messages = [
        call.args[0] for call in demo.capture_endpoint.send_return_data.await_args_list
        if call.args[0].topic in (CAPTURE_START_TOPIC, CAPTURE_STOP_TOPIC)
    ]
    assert [message.topic for message in messages] == [CAPTURE_START_TOPIC, CAPTURE_STOP_TOPIC] * 2
    for message, directory in zip(messages[::2], directories, strict=True):
        payload = json.loads(message.data)
        assert message.participant_id == "user"
        assert payload["target"] == directory.name
        assert payload["metadata"]["sop_packet"] == str(directory / "packet.json")
        assert payload["metadata"]["sop_session_id"] == directory.name


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
async def test_capture_start_failure_does_not_leave_an_active_packet(demo):
    demo.capture_endpoint.send_return_data.side_effect = RuntimeError("capture unavailable")
    with pytest.raises(RuntimeError, match="capture unavailable"):
        await demo.recorder.start_recording("user")
    assert not demo.recorder.is_recording("user")
    packet = json.loads(next(demo.root.glob("*/packet.json")).read_text())
    assert packet["status"] == "failed"
    assert packet["media_capture"]["control_status"] == "start_failed"
    demo.frame.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_capture_stop_failure_still_preserves_packet(demo, monkeypatch):
    monkeypatch.setattr("workflow_recorder_worker.recorder._MEDIA_FINALIZE_TIMEOUT_S", 0.02)
    await demo.recorder.start_recording("user")
    state = demo.recorder._sessions["user"]
    demo.capture_endpoint.send_return_data.side_effect = RuntimeError("hub closed")
    await demo.recorder.finish_recording("user")
    assert not demo.recorder.is_recording("user")
    assert all(task.done() for task in state.tasks)
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["media_capture"]["control_status"] == "finalization_failed"
    assert packet["narration_status"] == "failed"
    assert packet["status"] == "incomplete"
    assert "hub closed" in (state.directory / "errors.jsonl").read_text()


@pytest.mark.asyncio
async def test_narration_is_recovered_when_capture_finalizes_despite_stop_send_failure(demo):
    await demo.recorder.start_recording("user")
    state = demo.recorder._sessions["user"]
    await demo.say("Attach the shovel")
    finalize = demo.capture_endpoint.send_return_data.side_effect

    async def disconnected(message):
        await finalize(message)
        raise RuntimeError("hub closed")

    demo.capture_endpoint.send_return_data.side_effect = disconnected
    await demo.recorder.finish_recording("user")
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["narration_status"] == "complete"
    assert packet["counts"]["transcripts"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_narration_export_failure_marks_packet_incomplete(demo, missing):
    await demo.recorder.start_recording("user")
    state = demo.recorder._sessions["user"]
    bundle = state.media_directory / "bundle"
    if missing:
        (bundle / "transcript.jsonl").unlink()
    else:
        (bundle / "transcript.jsonl").write_text("malformed\n")
    await demo.recorder.finish_recording("user")
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["status"] == "incomplete"
    assert packet["narration_status"] == "failed"
    assert packet["counts"]["transcripts"] == 0
    assert not (state.directory / "transcript.jsonl").exists()
    assert "narration_export" in (state.directory / "errors.jsonl").read_text()


@pytest.mark.asyncio
async def test_empty_capture_exports_empty_narration(demo):
    await demo.recorder.start_recording("user")
    state = demo.recorder._sessions["user"]
    await demo.recorder.finish_recording("user")
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["status"] == packet["narration_status"] == "complete"
    assert packet["counts"]["transcripts"] == 0
    assert (state.directory / "transcript.jsonl").read_text() == ""


@pytest.mark.asyncio
async def test_incomplete_capture_keeps_available_narration(demo):
    await demo.recorder.start_recording("user")
    state = demo.recorder._sessions["user"]
    await demo.say("Attach the shovel")
    (state.media_directory / "bundle" / "manifest.json").write_text(json.dumps({
        "complete": False, "incomplete_reason": "return_traffic_drain_timeout",
        "transcript": "transcript.jsonl",
    }))
    demo.capture_endpoint.send_return_data.side_effect = None
    await demo.recorder.finish_recording("user")
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["status"] == packet["narration_status"] == "incomplete"
    assert packet["counts"]["transcripts"] == 1
    assert json.loads((state.directory / "transcript.jsonl").read_text())["text"] == "Attach the shovel"


@pytest.mark.asyncio
async def test_shared_narration_stays_participant_local(demo):
    states = {}
    for pid in ("user", "other"):
        await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), pid)
        await demo.say("start recording", pid)
        states[pid] = demo.recorder._sessions[pid]
    for pid in states:
        await demo.say(f"Narration by {pid}", pid)
    for pid, state in states.items():
        await demo.say("stop recording", pid)
        row = json.loads((state.directory / "transcript.jsonl").read_text())
        assert row["text"] == f"Narration by {pid}"


@pytest.mark.asyncio
@pytest.mark.parametrize("incomplete", [False, True])
async def test_missing_or_incomplete_capture_is_not_reported_complete(demo, monkeypatch, incomplete):
    monkeypatch.setattr("workflow_recorder_worker.recorder._MEDIA_FINALIZE_TIMEOUT_S", 0.02)
    await demo.recorder.start_recording("user")
    state = demo.recorder._sessions["user"]
    if incomplete:
        bundle = state.media_directory / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "manifest.json").write_text(json.dumps({
            "complete": False, "incomplete_reason": "return_traffic_drain_timeout",
        }))
    demo.capture_endpoint.send_return_data.side_effect = None
    await demo.recorder.finish_recording("user")
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["status"] == "incomplete"  # Images and captions are still retained.
    assert packet["media_capture"]["control_status"] == "finalization_failed"
    assert (state.directory / "errors.jsonl").exists()
    assert not demo.recorder.is_recording("user")


@pytest.mark.asyncio
async def test_shutdown_stops_each_participants_capture_once(demo):
    for pid in ("user", "other"):
        await demo.recorder.start_recording(pid)
    await demo.recorder.stop()
    await demo.recorder.stop()
    messages = [call.args[0] for call in demo.capture_endpoint.send_return_data.await_args_list]
    assert [(message.participant_id, message.topic) for message in messages] == [
        ("user", CAPTURE_START_TOPIC), ("other", CAPTURE_START_TOPIC),
        ("user", CAPTURE_STOP_TOPIC), ("other", CAPTURE_STOP_TOPIC),
    ]


@pytest.mark.asyncio
async def test_shared_capture_preserves_sop_jpegs_and_frame_linked_captions(demo):
    await demo.recorder.start_recording("user")
    state = demo.recorder._sessions["user"]
    for task in state.tasks:
        task.cancel()
    await asyncio.gather(*state.tasks, return_exceptions=True)
    images = ImageRegistry()
    image_bytes = io.BytesIO()
    Image.new("RGB", (32, 24), color="orange").save(image_bytes, format="JPEG")
    frame = ImageFrame(
        image=images.put(image_bytes.getvalue()), timestamp_us=state.started_at_us,
        sequence=1, width=32, height=24, participant_id="user",
    )
    demo.recorder._images = images
    demo.frame.execute.side_effect = None
    demo.frame.execute.return_value = frame
    demo.recorder._query_image.execute = AsyncMock(return_value=SimpleNamespace(
        available=True, text=json.dumps({
            "activity": "Assembly", "phase": "Attach wheel",
            "caption": "A wheel is attached", "delta": "Wheel added",
        }),
    ))
    await demo.recorder._capture(state)
    await demo.recorder._capture(state)  # The same source frame is still deduplicated.
    await demo.recorder._caption(state)
    await demo.recorder.finish_recording("user")
    packet = json.loads((state.directory / "packet.json").read_text())
    assert packet["capture"] == {"target_fps": 2, "caption_interval_s": 5}
    assert packet["counts"]["frames"] == packet["counts"]["captions"] == 1
    caption = json.loads((state.directory / "captions.jsonl").read_text())
    assert caption["frame_timestamp_us"] == frame.timestamp_us
    assert caption["frame_id"] == 1
    assert (state.directory / caption["frame_path"]).read_bytes() == image_bytes.getvalue()
    assert packet["hierarchy"][0]["name"] == "Assembly"


@pytest.mark.asyncio
async def test_sop_commands_finalize_shared_capture_bundles(demo, monkeypatch, tmp_path):
    from device_io_hub.capture._service import CaptureService
    from device_io_hub.capture.config import CaptureConfig

    # No frames are sent here: exercise actual control and disk finalization
    # without requiring NVENC or replacing the service's recording logic.
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", SimpleNamespace())
    service = CaptureService(CaptureConfig(
        out_dir=str(tmp_path / "captures"), session_mode="explicit", max_total_bytes=0,
    ))
    demo.capture_endpoint.send_return_data.side_effect = service._on_agent_data
    try:
        await demo.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined())
        assert not list((tmp_path / "captures").iterdir())
        for _ in range(2):
            await demo.say("start recording")
            state = demo.recorder._sessions["user"]
            await demo.say("Attach a wheel")
            await demo.capture_endpoint.send_return_data(DataMessage(
                "user", CAPTURE_TTS_TOPIC, time.time_ns() // 1000, b"Agent speech is not narration",
            ))
            await demo.say("stop recording")
            packet = json.loads((state.directory / "packet.json").read_text())
            media_directory = Path(packet["media_capture"]["directory"])
            manifest = json.loads(next(media_directory.glob("*/manifest.json")).read_text())
            assert manifest["complete"] is True
            assert manifest["participant_id"] == "user"
            assert manifest["metadata"]["sop_session_id"] == packet["session_id"]
            assert Path(manifest["metadata"]["sop_packet"]).is_file()
            assert packet["media_capture"]["control_status"] == "complete"
            assert packet["narration_status"] == "complete"
            rows = [json.loads(line) for line in (state.directory / "transcript.jsonl").read_text().splitlines()]
            assert [row["text"] for row in rows] == ["Attach a wheel"]
            assert packet["counts"]["transcripts"] == 1
    finally:
        await service.stop()


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
    await demo.say("stop recording")
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
    await demo.say("stop recording", "other")
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
    await demo.say("stop recording")
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
    await demo.say("stop recording")


@pytest.mark.parametrize(
    "text", [
        "start workflow lego", "end recording", "then stop recording", "stop recording after this",
        "start workflow", "finish workflow", "finish recording",
    ],
)
def test_recording_commands_do_not_match_guide_selectors_or_narration(text):
    assert RECORDING_COMMAND.fullmatch(text) is None
