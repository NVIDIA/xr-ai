# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composed control, permission and replacement-lifecycle regressions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from speaker_stt._selection import _Selection
from xr_ai_voice._frames import (
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
    _SpeakerEnrollmentFrame,
    _UnrecognizedSpeechFrame,
)
from xr_ai_voice._processors.speaker_stt import _SpeakerSttProcessor
from xr_ai_voice._processors.voice_gate import VoiceGateProcessor
from xr_ai_voice._speaker_client import _SpeakerClient
from xr_ai_voicegate import VoiceGateConfig, load_voice_gate_config

D = FrameDirection.DOWNSTREAM


def load(tmp_path, raw):
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_voice_gate_config(path)


def hybrid(tmp_path):
    return load(tmp_path, {"mode": "hybrid", "magic_phrases": ["hey agent"],
                           "speaker": {"enabled": True}, "listening_chime": True})


def speech(text):
    frame = TranscriptionFrame(text=text, user_id="a", timestamp="now")
    frame.transport_source = "a"
    return frame


def audio(track="first", size=320):
    frame = InputAudioRawFrame(audio=bytes(size), sample_rate=16000, num_channels=1)
    frame.transport_source, frame.track_id = "a", track
    return frame


@pytest.mark.parametrize("mode", ["conversation", "hybrid", "wake_only", "always_on"])
@pytest.mark.parametrize("key,value", [("start_phrase", "computer begin"), ("stop_phrase", "computer end"),
                                      ("require_wake_phrase", True), ("phrase_window_s", 9)])
def test_explicit_mode_rejects_misplaced_controls_instead_of_replacing(tmp_path, mode, key, value):
    raw = {"mode": mode, "magic_phrases": ["hey agent"], "speaker": {"enabled": True, key: value}}
    with pytest.raises(ValueError, match="move conversation controls from speaker to conversation"):
        load(tmp_path, raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "conversation", "hybrid"])
async def test_custom_controls_preserved_in_supported_location_and_wire(tmp_path, monkeypatch, mode):
    controls = {"start_phrase": "computer begin", "stop_phrase": "computer end",
                "require_wake_phrase": True, "phrase_window_s": 9}
    raw = {"magic_phrases": ["computer"], "speaker": {"enabled": True}}
    if mode is None:
        raw["speaker"].update(controls)
    else:
        raw.update(mode=mode, conversation=controls)
    cfg = load(tmp_path, raw)
    stream = SimpleNamespace(_open=AsyncMock(), _feed=AsyncMock(return_value=[]), _close=AsyncMock())
    monkeypatch.setattr("xr_ai_voice._speaker_client._make_speaker_stream", lambda *_a: stream)
    client = _SpeakerClient(cfg._speaker)
    await client._feed("a", b"\0\0", 100)
    payload = stream._open.call_args.args[0]
    assert {key: payload[key] for key in controls} == controls
    assert cfg._conversation.start_phrase == "computer begin"
    await client._close()


@pytest.mark.parametrize("raw,mode", [({}, "always_on"), ({"magic_phrases": ["hey agent"]}, "wake_only"),
                                     ({"conversation": {}}, "conversation")])
def test_mode_less_shapes_assert_effective_mode(tmp_path, raw, mode):
    assert load(tmp_path, raw)._mode == mode
    assert load_voice_gate_config(tmp_path / "missing.yaml")._mode == "always_on"


def test_mode_is_part_of_semantic_equality_and_direct_phrase_inference(tmp_path):
    assert VoiceGateConfig()._mode == "always_on"
    assert VoiceGateConfig(magic_phrases=("hey agent",))._mode == "wake_only"
    assert load(tmp_path, {"mode": "hybrid", "magic_phrases": ["hey agent"]}) != load(
        tmp_path, {"mode": "conversation", "magic_phrases": ["hey agent"]})


@pytest.mark.asyncio
@pytest.mark.parametrize("speaker", [False, True])
@pytest.mark.parametrize("middle", ["hey agent let's", "Hey agent, let's stop talking", "unrecognized"])
async def test_intervening_control_consumes_permission_but_not_fragment_matcher(tmp_path, speaker, middle):
    cfg = hybrid(tmp_path)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._use_speaker_asr = speaker
    gate.push_frame, gate._emit_chime = AsyncMock(), AsyncMock()
    selection = _Selection(cfg._speaker)
    processor = _SpeakerSttProcessor(cfg=cfg._speaker)
    async def forward(frame, *_args):
        await gate.process_frame(frame, D)
    processor.push_frame = forward
    async def say(text, step):
        if speaker:
            events = selection._enroll_completed(
                [(7, text, step*100_000, False)], completed_at_us=step*100_000+1000,
            )
            for event in events:
                await processor._event("a", event)
        else:
            started = UserStartedSpeakingFrame()
            started.transport_source = "a"
            await gate.process_frame(started, D)
            if text:
                await gate.process_frame(speech(text), D)
            else:
                await gate.process_frame(_UnrecognizedSpeechFrame(participant_id="a"), D)
    await say("hey agent", 0)
    await say("" if middle == "unrecognized" else middle, 1)
    await say("ambient speech", 2)
    assert not [c for c in gate.push_frame.call_args_list if isinstance(c.args[0], GatedQueryFrame)]
    # A genuinely new bare wake grants its own permission and can also join
    # a start control; consuming query permission must not erase fragments.
    await say("hey agent", 3)
    await say("let's start talking", 4)
    assert "a" in gate._conversation_active
    await processor.cleanup()
    await gate.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("partials", [False, True])
@pytest.mark.parametrize("chime", [False, True])
async def test_batch_hybrid_acknowledges_complete_turn_once(tmp_path, partials, chime):
    cfg = load(tmp_path, {"mode": "hybrid", "magic_phrases": ["hey agent"], "listening_chime": chime})
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate.push_frame = AsyncMock()
    # Exercise the real acknowledgement guard and configured gate, not a mock
    # of the very method whose once-only policy this regression checks.
    gate._gate._chime_wav = b"synthetic audio"
    gate.play_wav = AsyncMock()
    if partials:
        await gate.handle_partial_transcript("a", "hey agent describe")
        await gate.handle_partial_transcript("a", "hey agent describe this")
    await gate.process_frame(speech("hey agent describe this"), D)
    assert gate.play_wav.await_count == int(chime)
    assert len([c for c in gate.push_frame.call_args_list if isinstance(c.args[0], GatedQueryFrame)]) == 1
    await gate.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["rejoin", "track", "overload"])
@pytest.mark.parametrize("late", ["one_shot", "enrolled", "transcript", "error"])
async def test_replacement_revokes_old_response_before_drain_and_resets(tmp_path, replacement, late):
    cfg = hybrid(tmp_path)
    processor = _SpeakerSttProcessor(cfg=cfg._speaker)
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    frames = []
    requests = 0
    async def feed(*_args):
        nonlocal requests
        requests += 1
        if requests > 1:
            return []  # A replacement's audio is not the old response.
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        if late == "error":
            raise RuntimeError("old request failed during cancellation")
        return [{"kind": late, "speaker_id": 7, "text": "stale", "pts_us": 100, "fresh_match": True}]
    processor._client = SimpleNamespace(_feed=feed, _forget=AsyncMock(), _close=AsyncMock())
    processor.push_frame = AsyncMock(side_effect=lambda frame, *_args: frames.append(frame))
    await processor.process_frame(ParticipantJoinedFrame("a"), D)
    if late == "transcript":
        processor._enrolled.add("a")
    await processor.process_frame(audio(), D)
    await entered.wait()
    old = processor._tasks["a"]
    if replacement == "overload":
        await processor.process_frame(audio(size=32000), D)
        await processor.process_frame(audio(size=32000), D)
    trigger = ParticipantJoinedFrame("a") if replacement == "rejoin" else audio(
        "second" if replacement == "track" else "first",
    )
    replacing = asyncio.create_task(processor.process_frame(trigger, D))
    await cancelled.wait()
    assert "a" not in processor._participants and "a" not in processor._enrolled
    await processor.process_frame(audio(), D)  # No implicit readmission during drain.
    assert processor._tasks["a"] is old
    release.set()
    await replacing
    assert old.done() and "a" in processor._participants and "a" not in processor._enrolled
    assert not [f for f in frames if isinstance(f, _SpeakerEnrollmentFrame) and f.state == "enrolled"]
    assert not [f for f in frames if getattr(f, "text", None) == "stale"]
    assert len([f for f in frames if isinstance(f, _SpeakerEnrollmentFrame) and f.state == "reset"]) == 1
    await processor.cleanup()
    assert not processor._tasks and not processor._queues and not processor._lifecycles


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["rejoin", "track"])
async def test_departure_during_replacement_drain_cannot_readmit(tmp_path, replacement):
    processor = _SpeakerSttProcessor(cfg=hybrid(tmp_path)._speaker)
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def feed(*_args):
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return [{"kind": "enrolled"}]
    processor._client = SimpleNamespace(_feed=feed, _forget=AsyncMock(), _close=AsyncMock())
    processor.push_frame = AsyncMock()
    await processor.process_frame(ParticipantJoinedFrame("a"), D)
    await processor.process_frame(audio(), D)
    await entered.wait()
    replacing = asyncio.create_task(processor.process_frame(
        ParticipantJoinedFrame("a") if replacement == "rejoin" else audio("second"), D))
    await cancelled.wait()
    # Left removes the join identity synchronously before reaching its drain.
    leaving = asyncio.create_task(processor.process_frame(ParticipantLeftFrame("a"), D))
    while "a" in processor._lifecycles:
        await asyncio.sleep(0)
    release.set()
    await asyncio.gather(replacing, leaving)
    assert "a" not in processor._participants and "a" not in processor._enrolled
    await processor.process_frame(audio("second"), D)
    assert not processor._tasks and not processor._queues
    await processor.cleanup()
