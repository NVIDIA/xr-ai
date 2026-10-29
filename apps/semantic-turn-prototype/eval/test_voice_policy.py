"""The live adapter handles cumulative ASR without repeating admitted prefixes."""

import asyncio

import pytest
from semantic_turn_prototype._turns import _Policy
from semantic_turn_prototype._voice_policy import _ClefVoicePolicy
from test_turns import _Decisions
from xr_ai_voice import TranscriptUpdate, VoiceOutputState


def _update(text, revision=0, segment=0, epoch="source"):
    return TranscriptUpdate("user", epoch, segment, revision, text, 123_000)


@pytest.mark.asyncio
async def test_cumulative_asr_suffix_is_a_followup_not_duplicate_original_query():
    d = _Decisions()
    p = _ClefVoicePolicy(d, _Policy(stable_s=0), allow_unspoken_merge=True)
    p.update(_update("Show the panel"))
    first = await p.decide("user")
    assert first[0].text == "Show the panel"
    p.update(_update("Show the panel", revision=1))
    assert await p.decide("user") == []
    p.update(_update("Show the panel.", revision=2))
    assert await p.decide("user") == []
    d.relation = "continuation"
    p.update(_update("Show the panel on the right", revision=3))
    actions = await p.decide("user")
    assert [a.kind for a in actions] == ["interrupt", "submit"]
    assert actions[1].text == "Show the panel\non the right"
    assert actions[0].request_id == first[0].request_id


@pytest.mark.asyncio
async def test_playing_followup_uses_context_and_queue_drains_on_matching_done():
    d = _Decisions()
    p = _ClefVoicePolicy(d, _Policy(stable_s=0), allow_unspoken_merge=True)
    p.update(_update("Describe the scene"))
    first = (await p.decide("user"))[0]
    p.output(VoiceOutputState("user", first.request_id, "committed"))
    d.relation = "continuation"
    p.update(_update("focus on the corner", segment=1))
    actions = await p.decide("user")
    assert "context only" in actions[-1].text
    assert actions[-1].text.startswith("User follow-up (new intent): focus on the corner")
    d.relation = "separate"
    p.update(_update("What is the weather?", segment=2))
    assert await p.decide("user") == []
    p.output(VoiceOutputState("user", actions[-1].request_id, "done"))
    assert (await p.decide("user"))[0].text == "What is the weather?"


@pytest.mark.asyncio
async def test_post_admission_revision_is_not_executed_as_a_second_command():
    d = _Decisions()
    p = _ClefVoicePolicy(d, _Policy(stable_s=0))
    p.update(_update("Move the panel left"))
    await p.decide("user")
    p.update(_update("Move the panel right", revision=1))
    assert await p.decide("user") == []
    assert len(d.calls) == 1


@pytest.mark.asyncio
async def test_effectful_sample_defaults_to_contextual_followup_even_before_tts():
    d = _Decisions()
    p = _ClefVoicePolicy(d, _Policy(stable_s=0))
    p.update(_update("Send the report"))
    await p.decide("user")
    d.relation = "continuation"
    p.update(_update("and include a chart", segment=1))
    result = await p.decide("user")
    assert "context only" in result[-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["", "."])
async def test_retracted_candidate_invalidates_pending_classification(replacement):
    d = _Decisions()
    d.release = asyncio.Event()
    p = _ClefVoicePolicy(d, _Policy(stable_s=0))
    p.update(_update("Move the panel left"))
    pending = asyncio.create_task(p.decide("user"))
    await d.entered.wait()
    p.update(_update(replacement, revision=1))
    d.release.set()
    assert await pending == []
    assert await p.decide("user") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["Move the panel right", "Move the panel left"])
async def test_changed_admitted_snapshot_invalidates_pending_followup(replacement):
    d = _Decisions()
    p = _ClefVoicePolicy(d, _Policy(stable_s=0), allow_unspoken_merge=True)
    p.update(_update("Move the panel left"))
    await p.decide("user")
    d.entered.clear()
    d.release = asyncio.Event()
    d.relation = "continuation"
    p.update(_update("Move the panel left and make it red", revision=1))
    pending = asyncio.create_task(p.decide("user"))
    await d.entered.wait()
    p.update(_update(replacement, revision=2))
    d.release.set()
    assert await pending == []
    assert await p.decide("user") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["", "hey agent"])
async def test_speaker_gate_retraction_invalidates_blocked_clef_decision(replacement):
    from pathlib import Path
    from types import SimpleNamespace

    from pipecat.processors.frame_processor import FrameDirection
    from xr_ai_voice._frames import _TurnTranscriptFrame
    from xr_ai_voice._processors.speaker_stt import _SpeakerSttProcessor
    from xr_ai_voice._processors.voice_gate import VoiceGateProcessor
    from xr_ai_voicegate import load_voice_gate_config

    root = Path(__file__).resolve().parents[3]
    cfg = load_voice_gate_config(root / "agent-samples/conversation-example/yaml/voice_gate.yaml")
    d = _Decisions()
    d.release = asyncio.Event()
    p = _ClefVoicePolicy(d, _Policy(stable_s=0))
    speaker = _SpeakerSttProcessor(cfg=cfg._speaker, transcript_updates=True)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._semantic_turns = True

    async def capture(frame, *args):
        if isinstance(frame, _TurnTranscriptFrame):
            p.update(frame.update)

    async def to_gate(frame, *args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    speaker.push_frame = to_gate
    gate.push_frame = capture
    pending = None
    try:
        await speaker._event("user", {"kind": "enrolled"})
        await speaker._event("user", {"kind": "speech_start"})
        await speaker._event("user", {"kind": "partial", "text": "Move the panel"})
        pending = asyncio.create_task(p.decide("user"))
        await d.entered.wait()
        await speaker._event("user", {"kind": "partial", "text": replacement})
        d.release.set()
        assert await pending == []
        assert await p.decide("user") == []
        assert len(d.calls) == 1
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await speaker.cleanup()
        await gate.cleanup()


@pytest.mark.asyncio
async def test_empty_new_segment_preserves_earlier_incomplete_input():
    d = _Decisions()
    d.choice = "hold"
    p = _ClefVoicePolicy(d, _Policy(stable_s=0))
    p.update(_update("Move the panel"))
    assert await p.decide("user") == []
    p.update(_update("", segment=1))
    assert await p.decide("user") == []
    d.choice = "ready"
    p.update(_update("to the left", segment=1, revision=1))
    assert (await p.decide("user"))[0].text == "Move the panel to the left"
