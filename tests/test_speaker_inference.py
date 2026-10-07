# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker service tests with synthetic input; no recorded audio or model loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from xr_ai_voicegate._speaker import _SpeakerConfig


@pytest.fixture
def _streaming_models(monkeypatch):
    """Drive the live adapter with generated features and stubbed NeMo models."""
    import sys
    from types import ModuleType

    import torch

    activity = []
    asr_calls = []
    masks = []
    cache_allocations = []
    text = [""]

    class Adapter:
        def __init__(self, cfg, asr, diar):
            self.instance_manager = SimpleNamespace()

        def _forward_diarization_streaming_step(self, features, lengths, drop):
            predictions = torch.zeros(1, 2, 8)
            predictions[:, :, activity] = 1
            previous = self.instance_manager.diar_states.diar_pred_out_stream
            return object(), torch.cat((previous, predictions), dim=1)

        def forward_pre_encoded(self, features, lengths, drop):
            assert drop == 2
            return torch.zeros(1, 2, 128), torch.tensor([2])

    class Buffer:
        def __init__(self, **kwargs):
            pass

        def preprocess_audio(self, audio, device):
            return torch.zeros(1, 128, 16, device=device), torch.tensor([16], device=device)

    def initial_cache(*, batch_size):
        cache_allocations.append(batch_size)
        return torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.tensor([0])

    def decode(**kwargs):
        asr_calls.append(kwargs)
        return (None, [SimpleNamespace(text=text[0])], *initial_cache(batch_size=1), [SimpleNamespace()])

    for name, symbol, value in [
        ("nemo.collections.asr.parts.utils.multispk_transcribe_utils", "SpeakerTaggedASR", Adapter),
        ("nemo.collections.asr.parts.utils.streaming_utils", "CacheAwareStreamingAudioBuffer", Buffer),
    ]:
        module = ModuleType(name)
        setattr(module, symbol, value)
        monkeypatch.setitem(sys.modules, name, module)

    models = SimpleNamespace(
        cfg={},
        device=torch.device("cpu"),
        precision="float32",
        cache_samples=0,
        hop_samples=2560,
        nframes=2,
        asr=SimpleNamespace(
            encoder=SimpleNamespace(
                get_initial_cache_state=initial_cache,
                streaming_cfg=SimpleNamespace(valid_out_len=2, drop_extra_pre_encoded=2),
            ),
            set_speaker_targets=lambda target, others: masks.append((target.clone(), others.clone())),
            conformer_stream_step=decode,
        ),
        diar=SimpleNamespace(sortformer_modules=SimpleNamespace(init_streaming_state=lambda **kwargs: object())),
    )
    return models, activity, asr_calls, masks, cache_allocations, text


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


def test_forced_truncation_cannot_enroll_an_unowned_speaker(_streaming_models):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(silence_duration=0.16, max_utterance_s=0.32)
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


def test_forced_truncation_emits_owner_transcript_without_releasing(_streaming_models):
    from speaker_stt._inference import _Session

    models, activity, _calls, _masks, _allocations, text = _streaming_models
    cfg = _SpeakerConfig(silence_duration=0.16, max_utterance_s=0.32)
    session = _Session(models, cfg, audio_origin_us=1_000_000)
    session.selection.owner = 7
    activity[:] = [7]
    text[0] = cfg.stop_phrase
    events = session._feed(bytes(5120))
    events += session._feed(bytes(5120))

    assert events[-2:] == [
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
