# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation controls using ordinary STT doubles, without recorded audio."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from pipecat.frames.frames import InputAudioRawFrame, InterruptionFrame
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
        "magic_phrases": ["hey assistant"], "listening_chime": False,
        "conversation": {"enabled": enabled, "require_wake_phrase": require_wake,
                         "start_phrase": "Start talking to me", "phrase_window_s": 6},
    }))
    return load_voice_gate_config(path)


@pytest.mark.parametrize("parts", [
    ["Start talking.", "To me."],
    ["Start", "talking", "to me"],
    # An utterance boundary can split a word.
    ["Start tal", "king to me."],
    ["Start talking to m", "e."],
    # Word boundaries inside the phrase do not matter.
    ["Sta rt talk ing to me"],
])
def test_controls_join_phrase_prefixes(parts):
    matcher = _ControlMatcher(_ConversationConfig())
    actions = [matcher._feed(text, key="a", at_s=i) for i, text in enumerate(parts)]
    assert actions == ["pending"] * (len(parts) - 1) + ["start"]
    assert matcher._feed("stop talking to me about that", key="a", at_s=4) is None
    assert matcher._feed("someone said start talking to me", key="a", at_s=5) is None


@pytest.mark.parametrize("text,could_be", [
    ("stop", False), ("Stop talking.", False), ("stop talking please", False),  # a global STOP, not a control
    ("stop talking to", True), ("start", True), ("Start talking to me", True), ("what time is it", False),
])
def test_global_stop_words_are_not_control_prefixes(text, could_be):
    assert _ConversationConfig()._could_be_control(text) is could_be


def test_global_stop_is_never_withheld_and_releases_held_fragments():
    matcher = _ControlMatcher(_ConversationConfig())
    assert matcher._feed("stop", key="a", at_s=0) is None and matcher._released == ""
    assert matcher._feed("Star.", key="a", at_s=1) == "pending"
    assert matcher._feed("stop talking", key="a", at_s=2) is None
    assert matcher._released == "Star."
    assert matcher._feed("stop talking to me", key="a", at_s=3) == "stop"


@pytest.mark.parametrize("key,seconds", [("b", 1), ("a", 6.1), ("a", -1)])
def test_control_fragments_do_not_cross_source_or_time_window(key, seconds):
    matcher = _ControlMatcher(_ConversationConfig())
    assert matcher._feed("start talking", key="a", at_s=0) == "pending"
    assert matcher._feed("to me", key=key, at_s=seconds) is None
    assert matcher._feed("start talking to me", key=key, at_s=10) == "start"


@pytest.mark.parametrize("key,seconds,released", [("a", 1, "Star."), ("b", 1, ""), ("a", 6.1, "")])
def test_withheld_fragments_that_are_not_a_control_are_released(key, seconds, released):
    matcher = _ControlMatcher(_ConversationConfig())
    assert matcher._feed("Star.", key="a", at_s=0) == "pending"
    assert matcher._feed("How are you?", key=key, at_s=seconds) is None
    assert matcher._released == released
    assert matcher._feed("what time is it", key="a", at_s=10) is None
    assert matcher._released == ""


@pytest.mark.parametrize("parts,released", [
    (["Sta."], "Sta."),
    (["Sta.", "Star."], "Sta. Star."),
])
def test_partial_word_fragment_is_withheld_then_released(parts, released):
    matcher = _ControlMatcher(_ConversationConfig())
    for at_s, text in enumerate(parts):
        assert matcher._feed(text, key="a", at_s=at_s) == "pending"
    assert matcher._feed("went home", key="a", at_s=len(parts)) is None
    assert matcher._released == released


def test_unrelated_and_empty_transcripts_clear_fragments():
    matcher = _ControlMatcher(_ConversationConfig())
    for text in ("what time is it", "", "   "):
        assert matcher._feed("start talking", key="a", at_s=0) == "pending"
        assert matcher._feed(text, key="a", at_s=1) is None
        assert matcher._feed("to me", key="a", at_s=2) is None


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
    for part in ("start", "talking", "to me"):
        await h.speak(part)
    assert "a" in h.gate._conversation_active
    assert not h.accepted and h.captions.await_count == 0
    assert not await h.gate.handle_partial_transcript("a", "stop talking to")
    assert await h.gate.handle_partial_transcript("a", "stop")

    await h.speak("what is that")
    assert len(h.accepted) == (0 if h.gate._conversation_cfg.require_wake_phrase else 1)
    # A withheld control fragment leads the next transcript, as if spoken unsplit,
    # unless wake phrases are required and a bare wake phrase opened its follow-up window.
    require_wake = h.gate._conversation_cfg.require_wake_phrase
    await h.speak("hey assistant" if require_wake else "start")
    await h.speak("describe this")
    query = "describe this" if require_wake else "start describe this"
    assert h.accepted[-1].text == query
    await h.speak("stop")
    assert "a" in h.gate._conversation_active
    count = len(h.accepted)
    await h.speak("stop talking to")
    await h.speak("me")
    assert "a" not in h.gate._conversation_active and len(h.accepted) == count
    await h.speak("do not send this")
    assert len(h.accepted) == count
    # A bare wake phrase is its own final transcript; a withheld control fragment is not.
    expected = ["what is that", *(["hey assistant"] if require_wake else []), query, "stop"]
    assert [call.args[1] for call in h.captions.await_args_list] == expected


@pytest.mark.parametrize("failure", ["", RuntimeError("STT failed")])
@pytest.mark.parametrize("active", [False, True])
async def test_failed_or_empty_final_stt_invalidates_pending_control(harness, failure, active):
    h = harness
    if active:
        await h.speak("start talking to me")
    prefix, suffix = ("stop talking to", "me") if active else ("start talking", "to me")
    await h.speak(prefix)
    await h.speak(failure)
    await h.speak(suffix)
    assert ("a" in h.gate._conversation_active) is active
    await h.speak(f"{prefix} {suffix}")
    assert ("a" in h.gate._conversation_active) is not active


async def test_failed_stt_and_disconnect_reset_only_affected_participant(harness):
    h = harness
    await h.speak("start talking", "a")
    await h.speak("start talking", "b")
    await h.speak(RuntimeError("STT failed"), "a")
    await h.speak("to me", "b")
    await h.speak("to me", "a")
    assert h.gate._conversation_active == {"b"}
    await h.speak("start talking", "a")
    await h.gate.process_frame(ParticipantLeftFrame(participant_id="a"), FrameDirection.DOWNSTREAM)
    await h.speak("to me", "a")
    assert h.gate._conversation_active == {"b"}


async def test_closed_participant_cannot_interrupt_another_conversation(harness):
    h = harness
    await h.speak("start talking to me", "a")
    h.emitted.clear()
    await h.speak("stop", "b")
    assert not h.accepted and not any(isinstance(frame, InterruptionFrame) for frame in h.emitted)
    assert not await h.gate.handle_partial_transcript("b", "stop")
    assert h.gate._conversation_active == {"a"}


async def test_natural_speech_preserves_complete_queries(harness):
    h = harness
    await h.speak("start talking to me")

    utterances = [
        "Hey assistant, explain the current reading",
        "Compare the current and previous readings. Hey assistant, explain the difference.",
        "What is the current pressure? Hey assistant.",
    ]
    for utterance in utterances:
        await h.speak(utterance)

    assert [query.text for query in h.accepted] == utterances


async def test_natural_speech_partial_stop_uses_the_complete_transcript(harness):
    h = harness
    await h.speak("start talking to me")

    assert await h.gate.handle_partial_transcript("a", "stop")
    assert not await h.gate.handle_partial_transcript("a", "Keep explaining. Hey assistant, stop")


def test_disabled_conversation_preserves_legacy_gate_config(tmp_path):
    cfg = _config(tmp_path, enabled=False)
    assert cfg._conversation is None
    assert cfg.magic_phrases == ("hey assistant",)
    path = tmp_path / "voice.yaml"
    path.write_text("magic_phrases: []\n")
    assert load_voice_gate_config(path)._conversation is None


@pytest.mark.parametrize("raw", [
    {"enabled": "true"}, {"phrase_window_s": 0}, {"phrase_window_s": float("inf")},
    {"require_wake_phrase": "false"}, {"unknown": True},
    {"start_phrase": "!"}, {"stop_phrase": "start talking to me"},
])
def test_invalid_conversation_settings_fail_early(raw):
    with pytest.raises(ValueError):
        _ConversationConfig._from_yaml(raw)
