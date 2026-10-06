# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation controls using ordinary STT doubles, without recorded audio."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from pipecat.frames.frames import InputAudioRawFrame, InterruptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from xr_ai_voice._frames import ParticipantLeftFrame
from xr_ai_voice._pipeline import _build_voice_pipeline
from xr_ai_voice._processors.io import _VoiceIOProcessor
from xr_ai_voice._processors.vad_stt import VadConfig
from xr_ai_voice._transport import HubVoiceTransport
from xr_ai_voicegate import load_voice_gate_config
from xr_ai_voicegate._conversation import _ControlMatcher, _ConversationConfig


def _config(tmp_path, *, require_wake=False, enabled=True):
    path = tmp_path / "voice.yaml"
    path.write_text(yaml.safe_dump({
        "magic_phrases": ["hey agent"], "listening_chime": False,
        "conversation": {"enabled": enabled, "require_wake_phrase": require_wake,
                         "start_phrase": "Hey agent, let's start talking", "phrase_window_s": 6},
    }))
    return load_voice_gate_config(path)


@pytest.mark.parametrize("parts", [
    ["Hey agent.", "Let's start talking."],
    ["Hey agent", "let us", "start talking"],
])
def test_controls_join_only_complete_prefixes(parts):
    matcher = _ControlMatcher(_ConversationConfig())
    actions = [matcher._feed(text, key="a", at_s=i) for i, text in enumerate(parts)]
    assert actions == ["pending"] * (len(parts) - 1) + ["start"]
    assert matcher._feed("hey agent let's stop talking about that", key="a", at_s=4) is None
    assert matcher._feed("someone said hey agent let's start talking", key="a", at_s=5) is None


@pytest.mark.parametrize("key,seconds", [("b", 1), ("a", 6.1), ("a", -1)])
def test_control_fragments_do_not_cross_source_or_time_window(key, seconds):
    matcher = _ControlMatcher(_ConversationConfig())
    assert matcher._feed("hey agent", key="a", at_s=0) == "pending"
    assert matcher._feed("let's start talking", key=key, at_s=seconds) is None
    assert matcher._feed("hey agent let's start talking", key=key, at_s=10) == "start"


def test_unrelated_and_empty_transcripts_clear_fragments():
    matcher = _ControlMatcher(_ConversationConfig())
    for text in ("what time is it", "", "   "):
        assert matcher._feed("hey agent", key="a", at_s=0) == "pending"
        assert matcher._feed(text, key="a", at_s=1) is None
        assert matcher._feed("let's start talking", key="a", at_s=2) is None


@pytest.fixture
async def harness(tmp_path, monkeypatch, request):
    cfg = _config(tmp_path, require_wake=getattr(request, "param", False))
    stt, captions, accepted = SimpleNamespace(transcribe=AsyncMock()), AsyncMock(), []

    class Detector:
        def __init__(self, on_utterance, on_speech_start, **_kwargs):
            self.start, self.utterance = on_speech_start, on_utterance
        async def feed(self, pcm, rate):
            await self.start()
            await self.utterance(pcm, rate)
    monkeypatch.setattr("xr_ai_voice._processors.vad_stt.VadDetector", Detector)

    async def accept(query):
        accepted.append(query)
    transport = HubVoiceTransport()
    output = _VoiceIOProcessor(accept)
    output._enable_direct_mode = True
    output.push_frame = AsyncMock()
    pipeline, _worker = _build_voice_pipeline(
        transport=transport, stt=stt, tts=SimpleNamespace(), io_processor=output,
        vad_cfg=VadConfig(stop_probe_after_s=0), voice_gate_cfg=cfg,
        on_final_transcript=captions,
    )
    audio, gate = pipeline.processors[2:4]
    emitted = []

    async def to_gate(frame, *_args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
    async def to_output(frame, *_args):
        emitted.append(frame)
        await output.process_frame(frame, FrameDirection.DOWNSTREAM)
    audio.push_frame, gate.push_frame = to_gate, to_output
    now = 0

    async def speak(text, pid="a"):
        nonlocal now
        now += 1_000_000
        stt.transcribe.side_effect = text if isinstance(text, Exception) else None
        stt.transcribe.return_value = text if isinstance(text, str) else ""
        frame = InputAudioRawFrame(audio=bytes(640), sample_rate=16000, num_channels=1)
        frame.transport_source, frame.pts = pid, now * 1_000
        await audio.process_frame(frame, FrameDirection.DOWNSTREAM)
        if output._input_tasks:
            await asyncio.gather(*output._input_tasks)

    yield SimpleNamespace(speak=speak, gate=gate, accepted=accepted, captions=captions,
                          emitted=emitted, output=output)
    await audio.cleanup()
    await gate.cleanup()
    await output.cleanup()
    transport.shutdown()


@pytest.mark.parametrize("harness", [False, True], indirect=True)
async def test_conversation_workflow_with_optional_wake_gate(harness):
    h = harness
    await h.speak("ambient conversation")
    assert not h.accepted and h.captions.await_count == 0
    h.output.push_frame.assert_not_awaited()
    assert not await h.gate.handle_partial_transcript("a", "stop")
    for part in ("hey agent", "let us", "start talking"):
        await h.speak(part)
    assert "a" in h.gate._conversation_active
    assert not h.accepted and h.captions.await_count == 0
    assert not await h.gate.handle_partial_transcript("a", "hey agent let's stop")
    assert await h.gate.handle_partial_transcript("a", "stop")

    await h.speak("what is that")
    assert len(h.accepted) == (0 if h.gate._conversation_cfg.require_wake_phrase else 1)
    await h.speak("hey agent")
    await h.speak("describe this")
    assert h.accepted[-1].text == "describe this"
    await h.speak("stop")
    assert "a" in h.gate._conversation_active
    count = len(h.accepted)
    await h.speak("hey agent")
    await h.speak("let's stop talking")
    assert "a" not in h.gate._conversation_active and len(h.accepted) == count
    await h.speak("do not send this")
    assert len(h.accepted) == count
    assert [call.args[1] for call in h.captions.await_args_list] == ["what is that", "describe this", "stop"]


@pytest.mark.parametrize("failure", ["", RuntimeError("STT failed")])
@pytest.mark.parametrize("active", [False, True])
async def test_failed_or_empty_final_stt_invalidates_pending_control(harness, failure, active):
    h = harness
    if active:
        await h.speak("hey agent let's start talking")
    await h.speak("hey agent")
    await h.speak(failure)
    suffix = "let's stop talking" if active else "let's start talking"
    await h.speak(suffix)
    assert ("a" in h.gate._conversation_active) is active
    await h.speak(f"hey agent {suffix}")
    assert ("a" in h.gate._conversation_active) is not active


async def test_failed_stt_and_disconnect_reset_only_affected_participant(harness):
    h = harness
    await h.speak("hey agent", "a")
    await h.speak("hey agent", "b")
    await h.speak(RuntimeError("STT failed"), "a")
    await h.speak("let's start talking", "b")
    await h.speak("let's start talking", "a")
    assert h.gate._conversation_active == {"b"}
    await h.speak("hey agent", "a")
    await h.gate.process_frame(ParticipantLeftFrame(participant_id="a"), FrameDirection.DOWNSTREAM)
    await h.speak("let's start talking", "a")
    assert h.gate._conversation_active == {"b"}


async def test_typed_input_retains_wake_rules_without_opening_conversation(harness):
    h = harness
    for text in ("describe this", "hey agent describe this", "hey agent let's start talking"):
        frame = TranscriptionFrame(text=text, user_id="a", timestamp="")
        await h.gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        if h.output._input_tasks:
            await asyncio.gather(*h.output._input_tasks)
    assert [query.text for query in h.accepted] == ["describe this", "let's start talking"]
    assert not h.gate._conversation_active
    h.captions.assert_not_awaited()


async def test_closed_participant_cannot_interrupt_another_conversation(harness):
    h = harness
    await h.speak("hey agent let's start talking", "a")
    h.emitted.clear()
    await h.speak("stop", "b")
    assert not h.accepted and not any(isinstance(frame, InterruptionFrame) for frame in h.emitted)
    assert not await h.gate.handle_partial_transcript("b", "stop")
    assert h.gate._conversation_active == {"a"}


async def test_natural_speech_preserves_complete_queries(harness):
    h = harness
    await h.speak("hey agent let's start talking")

    utterances = [
        "Hey agent, explain the current reading",
        "Compare the current and previous readings. Hey agent, explain the difference.",
        "What is the current pressure? Hey agent.",
    ]
    for utterance in utterances:
        await h.speak(utterance)

    assert [query.text for query in h.accepted] == utterances


async def test_natural_speech_partial_stop_uses_the_complete_transcript(harness):
    h = harness
    await h.speak("hey agent let's start talking")

    assert await h.gate.handle_partial_transcript("a", "stop")
    assert not await h.gate.handle_partial_transcript("a", "Keep explaining. Hey agent, stop")


def test_disabled_conversation_preserves_legacy_gate_config(tmp_path):
    cfg = _config(tmp_path, enabled=False)
    assert cfg._conversation is None
    assert cfg.magic_phrases == ("hey agent",)
    path = tmp_path / "voice.yaml"
    path.write_text("magic_phrases: []\n")
    assert load_voice_gate_config(path)._conversation is None


@pytest.mark.parametrize("raw", [
    {"enabled": "true"}, {"phrase_window_s": 0}, {"phrase_window_s": float("inf")},
    {"require_wake_phrase": "false"}, {"unknown": True},
    {"start_phrase": "!"}, {"stop_phrase": "Hey agent, let's start talking"},
])
def test_invalid_conversation_settings_fail_early(raw):
    with pytest.raises(ValueError):
        _ConversationConfig._from_yaml(raw)
