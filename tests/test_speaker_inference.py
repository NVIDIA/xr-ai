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
    assert len(asr_calls) == count + 1
    events = feed([7, 1], "what is that?")
    assert events[0] == {"kind": "speech_start"}
    assert masks[-1][0].shape == (1, 2)
    assert masks[-1][0].all() and masks[-1][1].all()
    feed([1], "what is that?")
    assert not masks[-1][0].any() and masks[-1][1].all()
    assert feed([1], "what is that?")[-1]["kind"] == "transcript"
    assert len(asr_calls) == count + 6
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
    assert owner_call["previous_hypotheses"] is not other_call["previous_hypotheses"]
    assert owner_call["previous_hypotheses"][0].y_sequence == []
    assert other_call["previous_hypotheses"][0].y_sequence == []
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
def test_all_eight_continuations_stay_bounded_across_utterance_boundary(_streaming_models, finish):
    from speaker_stt._inference import _Session

    models, activity, calls, _masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(diagnostics=True, silence_duration=0.16,
                         max_utterance_s=0.32 if finish == "truncation" else 0.64)
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
    assert len(calls) == (16 if finish == "truncation" else 32)
    assert len(models.encodes) == (2 if finish == "truncation" else 18)
    assert len(session._speaker_states) == 8
    assert len(session._history) <= models.history_hops
    assert all(state.previous_hypotheses[0].y_sequence == [] for state in session._speaker_states.values())
    assert session.selection.owner is None
    assert not any(e["kind"] in {"enrolled", "transcript"} for e in events)
    assert len([e for e in events if e["kind"] == "diagnostic" and e["text"]]) == 8
    assert {e["status"] for e in events if e["kind"] == "diagnostic" and e["text"]} == {
        "ignored; waiting for enrollment" + ("; truncated" if finish == "truncation" else ""),
    }


@pytest.mark.parametrize("failure", ["decode", "allocation", "continuation"])
def test_ignored_diagnostic_failure_preserves_owner_transcription(_streaming_models, monkeypatch, failure):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, text = _streaming_models
    session = _Session(models, _SpeakerConfig(diagnostics=True, silence_duration=0.3), audio_origin_us=0)
    session.selection.owner = 7
    if failure == "decode":
        original = session._decode

        def decode(*args):
            if args[1] == 1:
                raise RuntimeError("background decoder failed")
            return original(*args)

        monkeypatch.setattr(session, "_decode", decode)
    elif failure == "continuation":
        original = session._step_native

        def step(*args):
            if args[3] == 1:
                raise RuntimeError("background continuation failed")
            return original(*args)

        monkeypatch.setattr(session, "_step_native", step)
    else:
        original = session._new_asr
        allocations = 0

        def allocate():
            nonlocal allocations
            allocations += 1
            if allocations == 2:
                raise RuntimeError("background cache allocation failed")
            return original()

        monkeypatch.setattr(session, "_new_asr", allocate)
    activity[:] = [1, 7]
    text[0] = "look around"
    session._feed(bytes(5120))
    assert session.selection.owner == 7 and session._asr is not None
    assert 7 not in session._diagnostic_streams
    assert 1 not in session._diagnostic_streams
    activity[:] = []
    session._feed(bytes(5120))
    events = session._feed(bytes(5120))
    assert session.selection.owner == 7
    assert [e["text"] for e in events if e["kind"] == "transcript"] == ["look around"]


def test_continuous_owner_encoding_failure_propagates_during_quiet_or_speech(_streaming_models, monkeypatch):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, _text = _streaming_models
    session = _Session(models, _SpeakerConfig(diagnostics=True), audio_origin_us=0)
    session.selection.owner = 7

    def fail(*_args):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(session.adapter, "forward_pre_encoded", fail)
    for speakers in ([1], [7]):
        activity[:] = speakers
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
    original_step = session._step_native
    phrases = {}

    def decode(predictions, selected, drop, stream, finished):
        decoded.append(selected)
        return original(predictions, selected, drop, stream, finished)

    def step(encoded, lengths, predictions, selected, *args):
        text[0] = phrases.get(selected, "")
        return original_step(encoded, lengths, predictions, selected, *args)

    session._decode = decode
    session._step_native = step

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

    def fail_one(predictions, selected, *args):
        if selected == 2:
            raise RuntimeError("candidate failed")
        return original(predictions, selected, *args)

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
    assert len(decoded) == 5
    assert len(calls) == (10 if tie else 8)
    assert len(models.encodes) == (7 if tie else 5)
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
    original_allocate, original_decode = session._new_asr, session._step_native

    def allocate():
        trace.append("allocate")
        return original_allocate()

    def decode(*args):
        trace.append(("decode", args[3]))
        return original_decode(*args)

    monkeypatch.setattr(session, "_new_asr", allocate)
    monkeypatch.setattr(session, "_step_native", decode)
    activity[:] = [1, 7]
    text[0] = "look around"
    session._feed(bytes(5120))
    assert trace == ["allocate", ("decode", 7), "allocate", ("decode", 1)]
    trace.clear()
    session._feed(bytes(5120))
    assert trace == [("decode", 7), ("decode", 1)]


@pytest.mark.parametrize("finish", ["silence", "truncation"])
def test_owner_decoder_history_and_finalization_are_independent_of_diagnostics(
    _streaming_models, monkeypatch, finish,
):
    from types import SimpleNamespace

    import torch
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from speaker_stt._inference import _Session

    models, activity, _calls, masks, _allocations, _text = _streaming_models

    def trace(diagnostics):
        cfg = _SpeakerConfig(diagnostics=diagnostics, silence_duration=0.32,
                             max_utterance_s=0.64 if finish == "truncation" else 30.0)
        session = _Session(models, cfg, audio_origin_us=0)
        session.selection.owner = 7
        selected = None
        owner_calls, events = [], []
        original = session._step_native

        def step(*args):
            nonlocal selected
            selected = args[3]
            return original(*args)

        def model_decode(**kwargs):
            previous = kwargs["previous_hypotheses"]
            count = previous[0].dec_state.count if previous else 0
            assert int(kwargs["cache_last_channel_len"][0]) == count
            assert not previous or previous[0].y_sequence == []
            assert kwargs["previous_pred_out"] is None
            final = kwargs["keep_all_outputs"]
            if selected == 7:
                owner_calls.append((count + 1, final))
            channel, time, length = models.asr.encoder.get_initial_cache_state(batch_size=1)
            length[0] = count + 1
            tokens = [65 + selected] if masks[-1][0].any() and not final else []
            hyp = Hypothesis(y_sequence=torch.tensor(tokens), dec_state=SimpleNamespace(count=count + 1))
            return None, [hyp], channel, time, length, [hyp]

        monkeypatch.setattr(session, "_step_native", step)
        monkeypatch.setattr(models.asr, "conformer_stream_step", model_decode)
        for _utterance in range(2):
            for speakers in ([1, 7],) * 4 if finish == "truncation" else ([1, 7], [1, 7], [1], [1]):
                activity[:] = speakers
                events.extend(session._feed(bytes(5120)))
            assert session.selection.owner == 7
        expected = "HHHH" if finish == "truncation" else "HH"
        assert [e["text"] for e in events if e["kind"] == "transcript"] == [expected, expected]
        assert [count for count, final in owner_calls if not final] == list(range(1, 9))
        assert sum(final for _, final in owner_calls) == (0 if finish == "truncation" else 4)
        return owner_calls, [event for event in events if event["kind"] != "diagnostic"]

    assert trace(False) == trace(True)


def test_new_candidate_cannot_enroll_from_its_pre_release_diagnostic_history(_streaming_models, monkeypatch):
    from speaker_stt._inference import _Session

    models, activity, calls, _masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(diagnostics=True, silence_duration=0.3)
    session = _Session(models, cfg, audio_origin_us=0)
    session.selection.owner = 7
    original = session._step_native
    released = False

    def decode(*args):
        speaker, state = args[3], args[5]
        if speaker == 7:
            text[0] = cfg.stop_phrase
        elif not released:
            text[0] = cfg.start_phrase
        else:
            previous = state.previous_hypotheses
            text[0] = (previous[0].dec_state["text"] if previous else "") or cfg.start_phrase
        return original(*args)

    monkeypatch.setattr(session, "_step_native", decode)
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
    assert all(not call["previous_hypotheses"] or call["previous_hypotheses"][0].y_sequence == []
               for call in calls[count:])
    activity[:] = []
    events += session._feed(bytes(5120))
    events += session._feed(bytes(5120))
    assert session.selection.owner is None
    assert session._speaker_states[1].previous_hypotheses[0].dec_state["text"] == cfg.start_phrase
    assert not any(e["kind"] == "enrolled" for e in events)


def test_repeated_turns_keep_native_continuation_and_bounded_output_history(_streaming_models):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, text = _streaming_models
    session = _Session(models, _SpeakerConfig(mode="always_on", silence_duration=0.16), audio_origin_us=0)
    state = None
    for index in range(40):
        expected = f"parcel number {index}"
        activity[:] = [3]
        text[0] = expected
        session._feed(bytes(5120))
        state = session._speaker_states[3] if state is None else state
        activity[:] = []
        events = session._feed(bytes(5120))
        assert [e["text"] for e in events if e["kind"] == "one_shot"] == [expected]
        assert session._speaker_states[3] is state
        assert state.previous_hypotheses[0].y_sequence == []
        assert len(state.tokens) == len(expected)
        assert not state.pending
        assert len(session._history) <= models.history_hops
    assert len(session._speaker_states) == 1


def test_next_turn_does_not_inherit_late_finalized_tokens(_streaming_models, monkeypatch):
    import torch
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, _text = _streaming_models

    def decode(**kwargs):
        previous = kwargs["previous_hypotheses"]
        step = previous[0].dec_state + 1 if previous else 1
        # The previous utterance's last word is emitted during the next
        # utterance's first frame; virtual quiet must not consume its new word.
        value = "alpha" if step == 1 else ""
        if step == 3:
            value = " beta" + ("" if kwargs["keep_all_outputs"] else " gamma")
        hyp = Hypothesis(y_sequence=torch.tensor([ord(c) for c in value]), dec_state=step)
        return None, [hyp], *models.asr.encoder.get_initial_cache_state(batch_size=1), [hyp]

    monkeypatch.setattr(models.asr, "conformer_stream_step", decode)
    session = _Session(models, _SpeakerConfig(mode="always_on", silence_duration=0.16), audio_origin_us=0)
    outputs = []
    for speakers in ([3], [], [3], []):
        activity[:] = speakers
        outputs.extend(e["text"] for e in session._feed(bytes(5120)) if e["kind"] == "one_shot")
    assert [value.strip() for value in outputs] == ["alpha beta", "gamma"]
    assert not session._speaker_states[3].pending


@pytest.mark.parametrize("mode", ["conversation", "always_on", "wake_only", "hybrid"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_finalized_candidate_reconciliation_remains_required_during_silence(
    _streaming_models, monkeypatch, mode, diagnostics,
):
    import torch
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, _text = _streaming_models

    def decode(**kwargs):
        previous = kwargs["previous_hypotheses"]
        step = previous[0].dec_state + 1 if previous else 1
        value = "alpha" if step == 1 else ""
        if step == 3:
            value = " beta" if kwargs["keep_all_outputs"] else "delta"
        hyp = Hypothesis(y_sequence=torch.tensor([ord(c) for c in value]), dec_state=step)
        return None, [hyp], *models.asr.encoder.get_initial_cache_state(batch_size=1), [hyp]

    monkeypatch.setattr(models.asr, "conformer_stream_step", decode)
    session = _Session(
        models, _SpeakerConfig(mode=mode, diagnostics=diagnostics, silence_duration=0.16), audio_origin_us=0,
    )
    activity[:] = [3]
    session._feed(bytes(5120))
    activity[:] = []
    events = session._feed(bytes(5120))
    if mode == "always_on":
        assert [e["text"] for e in events if e["kind"] == "one_shot"] == ["alpha beta"]
    assert session.selection.owner is None
    assert session._speaker_states[3].pending
    # No active candidate remains, but real quiet must still reconcile the
    # suffix already consumed by the finalized utterance. Real audio that
    # decodes differently is kept as new words, not a session failure.
    session._feed(bytes(5120))
    assert not session._speaker_states[3].pending
    assert session._speaker_states[3].emitted == [ord(c) for c in "delta"]


@pytest.mark.parametrize("owner", [None, 3])
def test_utterance_keeps_words_decoded_just_before_its_speaker_is_active(_streaming_models, monkeypatch, owner):
    import torch
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, _text = _streaming_models
    words = {1: "earlier", 2: "what", 3: " time"}

    def decode(**kwargs):
        previous = kwargs["previous_hypotheses"]
        step = previous[0].dec_state + 1 if previous else 1
        value = "" if kwargs["keep_all_outputs"] else words.get(step, "")
        hyp = Hypothesis(y_sequence=torch.tensor([ord(c) for c in value]), dec_state=step)
        return None, [hyp], *models.asr.encoder.get_initial_cache_state(batch_size=1), [hyp]

    monkeypatch.setattr(models.asr, "conformer_stream_step", decode)
    session = _Session(models, _SpeakerConfig(mode="always_on", silence_duration=0.16), audio_origin_us=0)
    session.selection.owner = owner
    events = []
    # Diarization marks the speaker active a step after their first word. A
    # word older than one silence_duration belongs to no utterance.
    for speakers in ([], [], [3], [], []):
        activity[:] = speakers
        events.extend(session._feed(bytes(5120)))
    kind = "one_shot" if owner is None else "transcript"
    assert [e["text"] for e in events if e["kind"] == kind] == ["what time"]


def test_final_padding_stays_attributed_to_the_finishing_speaker(_streaming_models):
    from speaker_stt._inference import _Session

    models, activity, calls, masks, _allocations, text = _streaming_models
    session = _Session(models, _SpeakerConfig(mode="always_on", silence_duration=0.16), audio_origin_us=0)
    text[0] = "talking"
    for speakers in ([3, 5], []):
        activity[:] = speakers
        session._feed(bytes(5120))
    padding = [mask for call, mask in zip(calls, masks, strict=True) if call["keep_all_outputs"]]
    assert len(padding) == 2 * models.final_padding_hops
    assert all(target.all() and not others.any() for target, others in padding)


def test_released_speaker_does_not_carry_its_last_word_into_reenrollment(_streaming_models, monkeypatch):
    import torch
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, _text = _streaming_models
    # The stop phrase's last word is decoded in the quiet step that ends it.
    words = {1: "hey agent, let's stop", 2: " talking", 3: "hey agent, let's start talking"}

    def decode(**kwargs):
        previous = kwargs["previous_hypotheses"]
        step = previous[0].dec_state + 1 if previous else 1
        value = "" if kwargs["keep_all_outputs"] else words.get(step, "")
        hyp = Hypothesis(y_sequence=torch.tensor([ord(c) for c in value]), dec_state=step)
        return None, [hyp], *models.asr.encoder.get_initial_cache_state(batch_size=1), [hyp]

    monkeypatch.setattr(models.asr, "conformer_stream_step", decode)
    session = _Session(models, _SpeakerConfig(silence_duration=0.16), audio_origin_us=0)
    session.selection.owner = 3
    events = []
    for speakers in ([3], [], [3], []):
        activity[:] = speakers
        events.extend(session._feed(bytes(5120)))
    assert [e["kind"] for e in events if e["kind"] in {"released", "enrolled"}] == ["released", "enrolled"]
