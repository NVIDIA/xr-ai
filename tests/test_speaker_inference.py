# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker service tests with synthetic input; no recorded audio or model loading."""

from __future__ import annotations

import pytest
from xr_ai_voicegate._speaker import _SpeakerConfig


def test_inference_runs_one_target_stream_and_bounds_diarization_history(_streaming_models):
    from speaker_stt._inference import _Session

    models, activity, asr_calls, masks, cache_allocations, text = _streaming_models
    cfg = _SpeakerConfig(silence_duration=0.3)
    session = _Session(models, cfg, audio_origin_us=1_000_000)

    def feed(speakers, transcript):
        activity[:] = speakers
        text[0] = transcript
        events = session._feed(bytes(5120))
        assert session.adapter.instance_manager.diar_states.diar_pred_out_stream.shape == (1, 2, 8)
        return events

    feed([7], cfg.start_phrase)
    feed([], cfg.start_phrase)
    assert feed([], cfg.start_phrase)[-1] == {"kind": "enrolled"}
    count = len(asr_calls)
    assert feed([1], "bystander command") == []
    assert len(asr_calls) == count
    events = feed([7, 1], "what is that?")
    assert events[0] == {"kind": "speech_start"}
    assert masks[-1][0].shape == (1, 2)
    assert masks[-1][0].all() and masks[-1][1].all()
    feed([1], "what is that?")
    assert not masks[-1][0].any() and masks[-1][1].all()
    assert feed([1], "what is that?")[-1]["kind"] == "transcript"
    assert len(asr_calls) == count + 3
    assert all(call["drop_extra_pre_encoded"] == 2 for call in asr_calls)
    assert all(size == 1 for size in cache_allocations)

    # Startup warming covers both decoding paths without touching live sessions.
    from speaker_stt._inference import _Models

    activity[:] = []
    text[0] = ""
    warmed = []

    def warm_session(cfg, *, audio_origin_us):
        isolated = _Session(models, cfg, audio_origin_us=audio_origin_us)
        warmed.append(isolated)
        return isolated

    models._session = warm_session
    count = len(asr_calls)
    _Models._warmup(models)
    assert {call["keep_all_outputs"] for call in asr_calls[count:]} == {False, True}
    assert session.selection.owner == 7
    assert warmed[0] is not session


@pytest.mark.parametrize("audio", [bytes(1), bytes(32002)])
def test_invalid_pcm_fails_before_model_inference(_streaming_models, audio):
    from speaker_stt._inference import _Session

    models, _activity, calls, _masks, _allocations, _text = _streaming_models
    session = _Session(models, _SpeakerConfig(), audio_origin_us=0)
    with pytest.raises(ValueError):
        session._feed(audio)
    assert not calls


def test_audio_timeline_uses_origin_and_accepted_sample_count(_streaming_models):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, _text = _streaming_models
    activity[:] = [7]
    session = _Session(models, _SpeakerConfig(), audio_origin_us=1_234_567)
    session.selection.owner = 7
    observed_pts = []
    activity_step = session.selection._activity

    def record_pts(active, seconds, pts_us):
        observed_pts.append(pts_us)
        return activity_step(active, seconds, pts_us)

    session.selection._activity = record_pts
    # Request boundaries are transport details. Two differently partitioned
    # 2,560-sample hops still advance the media clock by exactly 160 ms each.
    for size in (1000, 4120, 2000, 3120):
        session._feed(bytes(size))

    assert observed_pts == [1_234_567, 1_394_567]


def test_diagnostics_isolate_speaker_history_and_share_encoding(_streaming_models):
    from speaker_stt._inference import _Session

    models, activity, calls, masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(diagnostics=True, silence_duration=0.3)
    session = _Session(models, cfg, audio_origin_us=0)
    def feed(speakers, transcript):
        activity[:] = speakers
        text[0] = transcript
        result = session._feed(bytes(5120))
        assert session.adapter.instance_manager.diar_states.diar_pred_out_stream.shape == (1, 2, 8)
        return result

    # Diagnostics must not turn competing start phrases into enrollment.
    feed([7, 1], cfg.start_phrase)
    feed([], cfg.start_phrase)
    events = feed([], cfg.start_phrase)
    assert session.selection.owner is None
    assert {e["speaker_id"] for e in events if e["kind"] == "diagnostic"} == {1, 7}
    assert not session._diagnostic_streams

    feed([7], cfg.start_phrase)
    feed([], cfg.start_phrase)
    events = feed([], cfg.start_phrase)
    assert {"kind": "enrolled"} in events
    assert any(e.get("speaker_id") == 7 and e.get("status") == "enrolled" for e in events)

    count, encodes = len(calls), len(models.encodes)
    events = feed([7, 1], "what is that?")
    assert len(models.encodes) == encodes + 1
    assert any(e.get("speaker_id") == 1 and e.get("status") == "ignored" for e in events)
    owner_call, other_call = calls[count:]
    assert owner_call["cache_last_channel"] is not other_call["cache_last_channel"]
    assert owner_call["previous_hypotheses"] is other_call["previous_hypotheses"] is None
    assert all(target.all() and interference.all() for target, interference in masks[-2:])
    feed([], "what is that?")
    owner_call, other_call = calls[-2:]
    assert owner_call["previous_hypotheses"] is not other_call["previous_hypotheses"]
    events = feed([], "what is that?")
    assert [e["speaker_id"] for e in events if e["kind"] == "transcript"] == [7]
    assert [e["speaker_id"] for e in events if e["kind"] == "diagnostic" and e["text"]] == [1]
    assert not session._diagnostic_streams

    feed([1], cfg.stop_phrase)
    feed([], cfg.stop_phrase)
    events = feed([], cfg.stop_phrase)
    assert not any(e["kind"] in {"released", "transcript", "partial", "speech_start"} for e in events)
    assert events[-1]["text"] == cfg.stop_phrase
    assert session.selection.owner == 7


@pytest.mark.parametrize("finish", ["silence", "truncation"])
def test_all_eight_diagnostic_caches_are_released_at_utterance_boundary(_streaming_models, finish):
    from speaker_stt._inference import _Session

    models, activity, calls, _masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(diagnostics=True, silence_duration=0.16, max_utterance_s=0.32)
    session = _Session(models, cfg, audio_origin_us=0)
    activity[:] = range(8)
    text[0] = cfg.start_phrase
    session._feed(bytes(5120))
    assert len(session._candidate_streams) == 8
    assert not session._diagnostic_streams
    assert len(calls) == 8 and len(models.encodes) == 1
    if finish == "silence":
        activity[:] = []
    events = session._feed(bytes(5120))
    assert not session._diagnostic_streams
    assert len(calls) == 16 and len(models.encodes) == 2
    assert all(call["keep_all_outputs"] for call in calls[-8:])
    assert session.selection.owner is None
    assert not any(e["kind"] in {"enrolled", "transcript"} for e in events)
    assert len([e for e in events if e["kind"] == "diagnostic" and e["text"]]) == 8


@pytest.mark.parametrize("failure", ["decode", "allocation"])
def test_ignored_diagnostic_failure_preserves_owner_transcription(_streaming_models, monkeypatch, failure):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, text = _streaming_models
    session = _Session(models, _SpeakerConfig(diagnostics=True, silence_duration=0.3), audio_origin_us=0)
    session.selection.owner = 7
    if failure == "decode":
        original = session._decode

        def decode(*args):
            if args[3] == 1:
                raise RuntimeError("background decoder failed")
            return original(*args)

        monkeypatch.setattr(session, "_decode", decode)
    else:
        original = session._new_asr
        allocations = 0

        def allocate():
            nonlocal allocations
            allocations += 1
            if allocations > 1:
                raise RuntimeError("background cache allocation failed")
            return original()

        monkeypatch.setattr(session, "_new_asr", allocate)
    activity[:] = [1, 7]
    text[0] = "look around"
    session._feed(bytes(5120))
    assert session.selection.owner == 7 and 7 in session._diagnostic_streams
    assert 1 not in session._diagnostic_streams
    activity[:] = []
    session._feed(bytes(5120))
    events = session._feed(bytes(5120))
    assert session.selection.owner == 7
    assert [e["text"] for e in events if e["kind"] == "transcript"] == ["look around"]


def test_diagnostic_encoding_failure_is_isolated_only_without_a_selected_speaker(_streaming_models, monkeypatch):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, _text = _streaming_models
    session = _Session(models, _SpeakerConfig(diagnostics=True), audio_origin_us=0)
    session.selection.owner = 7

    def fail(*_args):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(session.adapter, "forward_pre_encoded", fail)
    activity[:] = [1]
    session._feed(bytes(5120))
    assert session.selection.owner == 7 and not session._diagnostic_streams
    activity[:] = [7]
    with pytest.raises(RuntimeError, match="encoder failed"):
        session._feed(bytes(5120))


def test_selected_diagnostic_decoder_failure_propagates_for_fail_closed_reset(_streaming_models, monkeypatch):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, _text = _streaming_models
    session = _Session(models, _SpeakerConfig(diagnostics=True), audio_origin_us=0)
    session.selection.owner = 7

    def fail(*_args):
        raise RuntimeError("owner decoder failed")

    monkeypatch.setattr(session, "_decode", fail)
    activity[:] = [7]
    with pytest.raises(RuntimeError, match="owner decoder failed"):
        session._feed(bytes(5120))


@pytest.mark.parametrize("diagnostics", [False, True])
def test_forced_truncation_cannot_enroll_an_unowned_speaker(_streaming_models, diagnostics):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(diagnostics=diagnostics, silence_duration=0.16, max_utterance_s=0.32)
    session = _Session(models, cfg, audio_origin_us=0)
    activity[:] = [7]
    text[0] = cfg.start_phrase
    events = session._feed(bytes(5120))
    events += session._feed(bytes(5120))
    assert session.selection.owner is None
    assert not any(event["kind"] in {"enrolled", "released", "transcript"} for event in events)
    assert session.selection.candidate is None
    assert session.selection.duration == 0
    assert session.selection._controls._words == ""


@pytest.mark.parametrize("diagnostics", [False, True])
def test_forced_truncation_emits_owner_transcript_without_releasing(_streaming_models, diagnostics):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(diagnostics=diagnostics, silence_duration=0.16, max_utterance_s=0.32)
    session = _Session(models, cfg, audio_origin_us=1_000_000)
    session.selection.owner = 7
    activity[:] = [7]
    text[0] = cfg.stop_phrase
    events = session._feed(bytes(5120))
    events += session._feed(bytes(5120))

    assert [event for event in events if event["kind"] != "diagnostic"][-2:] == [
        {"kind": "speech_stop"},
        {
            "kind": "transcript",
            "text": cfg.stop_phrase,
            "pts_us": 1_000_000,
            "speaker_id": 7,
        },
    ]
    assert session.selection.owner == 7
    assert session.selection.candidate is None
    assert session.selection.duration == 0
    assert session.selection._controls._words == ""
    assert session.last_partial == ""


def _attributed_session(streaming_models, cfg):
    from speaker_stt._inference import _Session

    models, activity, calls, masks, allocations, text = streaming_models
    session = _Session(models, cfg, audio_origin_us=1_000_000)
    decoded = []
    original = session._decode
    phrases = {}

    def decode(encoded, lengths, predictions, selected, drop, stream, finished):
        decoded.append(selected)
        text[0] = phrases.get(selected, "")
        return original(encoded, lengths, predictions, selected, drop, stream, finished)

    session._decode = decode

    def feed(speakers, transcripts, *, audio=None):
        activity[:] = speakers
        phrases.update(transcripts)
        return session._feed(bytes(5120) if audio is None else audio)

    return session, feed, decoded


def test_phrase_speaker_enrolls_over_an_already_talking_nonowner(_streaming_models):
    cfg = _SpeakerConfig(silence_duration=0.3)
    session, feed, decoded = _attributed_session(_streaming_models, cfg)
    assert feed([1], {1: "ordinary conversation"}) == []
    assert feed([1, 2], {2: cfg.start_phrase}) == []
    assert feed([1], {}) == []
    assert feed([1], {}) == [{"kind": "enrolled"}]
    assert session.selection.owner == 2 and 2 in decoded
    assert not session._candidate_streams
    assert session._asr is None  # candidates were released before owner decoding starts
    feed([1, 2], {1: cfg.start_phrase, 2: "owner query"})
    assert session.selection.owner == 2
    assert decoded[-1] == 2


def test_simultaneous_step_matches_are_rejected_without_global_noise_silence(_streaming_models):
    cfg = _SpeakerConfig(silence_duration=0.3)
    session, feed, decoded = _attributed_session(_streaming_models, cfg)
    feed([1, 2], {1: cfg.start_phrase, 2: cfg.start_phrase})
    feed([], {})
    assert feed([], {}, audio=b"\xff\x3f" * 2560) == []
    assert session.selection.owner is None and {1, 2} <= set(decoded)
    assert not session._candidate_streams
    feed([2], {2: cfg.start_phrase})
    feed([], {})
    assert feed([], {}) == [{"kind": "enrolled"}]
    assert session.selection.owner == 2


@pytest.mark.parametrize("diagnostics", [False, True])
def test_candidate_cache_budget_is_diarizer_bound_and_encoding_is_shared(_streaming_models, diagnostics):
    cfg = _SpeakerConfig(diagnostics=diagnostics)
    session, feed, decoded = _attributed_session(_streaming_models, cfg)
    encoded = []
    allocated = []
    original = session.adapter.forward_pre_encoded
    new_asr = session._new_asr

    def encode(*args):
        encoded.append(True)
        return original(*args)

    session.adapter.forward_pre_encoded = encode
    def allocate():
        allocated.append(True)
        return new_asr()

    session._new_asr = allocate
    feed(list(range(8)), {})
    assert len(session._candidate_streams) == 8
    assert len(allocated) == 8
    assert not session._diagnostic_streams
    assert decoded == list(range(8)) and encoded == [True]


@pytest.mark.parametrize("diagnostics", [False, True])
def test_candidate_failure_is_not_silently_dropped_from_arbitration(_streaming_models, diagnostics):
    cfg = _SpeakerConfig(diagnostics=diagnostics, silence_duration=0.3)
    session, feed, _decoded = _attributed_session(_streaming_models, cfg)
    original = session._decode

    def fail_one(encoded, lengths, predictions, selected, *args):
        if selected == 2:
            raise RuntimeError("candidate failed")
        return original(encoded, lengths, predictions, selected, *args)

    session._decode = fail_one
    with pytest.raises(RuntimeError, match="candidate failed"):
        feed([1, 2], {1: cfg.start_phrase, 2: cfg.start_phrase})
    assert session.selection.owner is None


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("tie", [False, True])
def test_candidate_display_reuses_overlap_decode_without_changing_nomination(_streaming_models, diagnostics, tie):
    models, _activity, calls, _masks, _allocations, _text = _streaming_models
    cfg = _SpeakerConfig(diagnostics=diagnostics, silence_duration=0.16)
    session, feed, decoded = _attributed_session(_streaming_models, cfg)
    feed([1], {1: cfg.start_phrase if tie else "ordinary conversation"})
    feed([1, 2], {2: cfg.start_phrase})
    events = feed([] if tie else [1], {})
    assert len(calls) == len(decoded) == 5
    assert len(models.encodes) == 3
    assert not session._diagnostic_streams
    assert session.selection.owner == (None if tie else 2)
    assert [e for e in events if e["kind"] == "enrolled"] == ([] if tie else [{"kind": "enrolled"}])
    display = [e for e in events if e["kind"] == "diagnostic"]
    assert bool(display) is diagnostics
    if diagnostics and tie:
        assert {e["speaker_id"] for e in display if e["text"]} == {1, 2}
        assert {e["pts_us"] for e in display if e["text"]} == {1_000_000, 1_160_000}


def test_truncated_candidate_cannot_restart_from_same_speech_tail(_streaming_models):
    cfg = _SpeakerConfig(silence_duration=0.16, max_utterance_s=0.48)
    session, feed, decoded = _attributed_session(_streaming_models, cfg)
    for _ in range(3):
        assert feed([7], {7: cfg.start_phrase}) == []
    count = len(decoded)
    for _ in range(3):
        assert feed([7], {}) == []
    assert len(decoded) == count and session.selection.owner is None
    feed([], {})  # this speaker's existing inactivity interval ends its episode
    feed([7], {7: cfg.start_phrase})
    assert feed([], {}) == [{"kind": "enrolled"}]


def test_selected_speaker_allocates_and_decodes_before_optional_caches(_streaming_models, monkeypatch):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, text = _streaming_models
    session = _Session(models, _SpeakerConfig(diagnostics=True, silence_duration=0.3), audio_origin_us=0)
    session.selection.owner = 7
    trace = []
    original_allocate, original_decode = session._new_asr, session._decode

    def allocate():
        trace.append("allocate")
        return original_allocate()

    def decode(*args):
        trace.append(("decode", args[3]))
        return original_decode(*args)

    monkeypatch.setattr(session, "_new_asr", allocate)
    monkeypatch.setattr(session, "_decode", decode)
    activity[:] = [1, 7]
    text[0] = "look around"
    session._feed(bytes(5120))
    assert trace == ["allocate", ("decode", 7), "allocate", ("decode", 1)]
    trace.clear()
    session._feed(bytes(5120))
    assert trace == [("decode", 7), ("decode", 1)]



def test_new_candidate_cannot_enroll_from_its_pre_release_diagnostic_history(_streaming_models, monkeypatch):
    from speaker_stt._inference import _Session

    models, activity, calls, _masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(diagnostics=True, silence_duration=0.3)
    session = _Session(models, cfg, audio_origin_us=0)
    session.selection.owner = 7
    original = session._decode
    released = False

    def decode(*args):
        speaker, stream = args[3], args[5]
        if speaker == 7:
            text[0] = cfg.stop_phrase
        elif not released:
            text[0] = cfg.start_phrase
        else:
            # Silence preserves a cumulative hypothesis; a fresh cache starts empty.
            text[0] = stream.previous_hypotheses[0].text if stream.previous_hypotheses is not None else ""
        result = original(*args)
        stream.previous_hypotheses[0].text = result
        return result

    monkeypatch.setattr(session, "_decode", decode)
    activity[:] = [7, 1]
    session._feed(bytes(5120))
    activity[:] = [1]
    session._feed(bytes(5120))
    events = session._feed(bytes(5120))
    assert {"kind": "released"} in events
    assert session.selection.owner is None
    assert not session._diagnostic_streams
    released = True
    count = len(calls)
    events = session._feed(bytes(5120))
    assert 1 in session._candidate_streams and not session._diagnostic_streams
    assert calls[count]["previous_hypotheses"] is None
    activity[:] = []
    events += session._feed(bytes(5120))
    events += session._feed(bytes(5120))
    assert session.selection.owner is None
    assert not any(e["kind"] == "enrolled" for e in events)
