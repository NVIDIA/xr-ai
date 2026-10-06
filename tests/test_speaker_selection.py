# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker service tests with synthetic input; no recorded audio or model loading."""

from __future__ import annotations

import pytest
from speaker_stt._selection import _Selection
from xr_ai_voicegate._speaker import _SpeakerConfig


def _enroll(selection, speaker=5):
    selection._activity({speaker}, 0.2, 123)
    events = selection._finish(selection.cfg.start_phrase)
    assert events == [{"kind": "enrolled"}]


def test_enrollment_needs_exact_phrase_and_unambiguous_speaker():
    cfg = _SpeakerConfig()
    selection = _Selection(cfg)
    selection._activity({2}, 0.2, 10)
    assert selection._finish("someone said hey agent let's start talking") == []
    assert selection.owner is None
    selection._activity({2}, 0.2, 10)
    selection._activity({2, 4}, 0.2, 20)
    assert selection._finish(cfg.start_phrase) == []
    assert selection.owner is None
    _enroll(selection, 7)
    assert selection.owner == 7


def test_ambiguous_onset_cannot_be_enrolled_from_later_single_speaker():
    selection = _Selection(_SpeakerConfig())
    assert selection._activity({1, 2}, 0.2, 10)[0] is None
    assert selection._activity({1}, 0.2, 20)[0] is None
    assert selection._finish(selection.cfg.start_phrase) == []
    _enroll(selection, 1)


def test_only_owner_reaches_asr_and_release_requires_owner():
    selection = _Selection(_SpeakerConfig())
    _enroll(selection, 6)
    assert selection._activity({0, 4}, 0.2, 200) == (None, [], False)
    assert selection._finish(selection.cfg.stop_phrase) == [{"kind": "speech_stop"}]
    assert selection.owner == 6
    selected, events, _ = selection._activity({6, 4}, 0.2, 300)
    assert selected == 6
    assert events == [{"kind": "speech_start"}]
    assert selection._finish("what is that?")[-1] == {
        "kind": "transcript",
        "text": "what is that?",
        "pts_us": 300,
        "speaker_id": 6,
    }
    selection._activity({6}, 0.2, 400)
    assert selection._finish(selection.cfg.stop_phrase)[-1] == {"kind": "released"}
    assert selection.owner is None
    _enroll(selection, 3)


def test_control_phrase_matching_normalizes_case_punctuation_and_apostrophes():
    cfg = _SpeakerConfig()
    assert cfg._matches_start("HEY AGENT! Lets start talking.")
    assert cfg._matches_stop("Hey agent, let’s stop talking!")
    assert not cfg._matches_stop("Hey agent, let's stop talking about this")
    assert cfg._could_be_control("Hey agent let's stop")


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": "true"},
        {"start_phrase": ""},
        {"start_phrase": "stop", "stop_phrase": "stop"},
        {"require_wake_phrase": "false"},
        {"activity_threshold": float("nan")},
        {"silence_duration": -1},
        {"unknown": True},
        {"endpoint": "tcp://*:5000"},
        {"endpoint": "ipc:///"},
        {"phrase_window_s": float("inf")},
        {"max_utterance_s": 0.5},
        {"activity_threshold": 1},
    ],
)
def test_invalid_speaker_settings_fail_early(settings):
    with pytest.raises(ValueError):
        _SpeakerConfig._from_yaml({"enabled": True, **settings})


@pytest.mark.parametrize("interference", ["other-speaker", "overlap", "unrelated", "timeout"])
def test_enrollment_fragments_require_same_speaker_without_interference(interference):
    selection = _Selection(_SpeakerConfig())
    selection._activity({7}, 0.3, 0)
    assert selection._finish("hey agent") == []
    if interference == "other-speaker":
        selection._activity({1}, 0.3, 1_000_000)
    elif interference == "overlap":
        selection._activity({7, 1}, 0.3, 1_000_000)
    elif interference == "unrelated":
        selection._activity({7}, 0.3, 1_000_000)
        selection._finish("what time is it")
        selection._activity({7}, 0.3, 2_000_000)
    else:
        selection._activity({7}, 0.3, 6_100_000)
    assert selection._finish("let's start talking") == []
    assert selection.owner is None


def test_start_and_stop_controls_can_span_utterances():
    selection = _Selection(_SpeakerConfig())
    selection._activity({7}, 0.3, 0)
    assert selection._finish("hey agent") == []
    selection._activity({7}, 0.3, 1_000_000)
    assert selection._finish("let us start talking") == [{"kind": "enrolled"}]
    selection._activity({7}, 0.3, 2_000_000)
    assert selection._finish("hey agent")[-1]["kind"] == "control_pending"
    selection._activity({7}, 0.3, 3_000_000)
    assert selection._finish("let's stop talking")[-1] == {"kind": "released"}
