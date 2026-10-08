# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker service tests with synthetic input; no recorded audio or model loading."""

from __future__ import annotations

import pytest
from speaker_stt._selection import _Selection
from xr_ai_voicegate._speaker import _SpeakerConfig


def _enroll(selection, speaker=5):
    events = selection._enroll_completed([(speaker, selection.cfg.start_phrase, 123, False)])
    assert events == [{"kind": "enrolled"}]


def test_enrollment_needs_one_attributed_exact_phrase():
    cfg = _SpeakerConfig()
    selection = _Selection(cfg)
    assert selection._enroll_completed([(2, "someone said hey agent let's start talking", 10, False)]) == []
    assert selection.owner is None
    assert selection._activity({2, 4}, 0.2, 20) == (None, [], False)
    assert selection.owner is None
    _enroll(selection, 7)
    assert selection.owner == 7


@pytest.mark.parametrize("speakers", [(1, 2), (2, 1)])
def test_same_step_exact_phrase_tie_cannot_choose_by_label_order(speakers):
    selection = _Selection(_SpeakerConfig())
    assert selection._enroll_completed([
        (speaker, selection.cfg.start_phrase, 10, False) for speaker in speakers
    ]) == []
    assert selection.owner is None
    _enroll(selection, 1)


def test_unique_phrase_completion_wins_and_later_start_cannot_take_over():
    selection = _Selection(_SpeakerConfig())
    assert selection._enroll_completed([
        (1, "random ongoing conversation", 10, False),
        (2, selection.cfg.start_phrase, 20, False),
    ]) == [{"kind": "enrolled"}]
    assert selection.owner == 2
    assert selection._enroll_completed([(1, selection.cfg.start_phrase, 30, False)]) == []
    assert selection.owner == 2


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


@pytest.mark.parametrize("interference", ["other-speaker", "unrelated", "timeout", "truncated"])
def test_enrollment_fragments_remain_attributed_and_bounded(interference):
    selection = _Selection(_SpeakerConfig())
    assert selection._enroll_completed([(7, "hey agent", 0, False)]) == []
    if interference == "other-speaker":
        assert selection._enroll_completed([(1, "let's start talking", 1_000_000, False)]) == []
        return
    elif interference == "unrelated":
        selection._enroll_completed([(7, "what time is it", 1_000_000, False)])
        at_us = 2_000_000
    elif interference == "truncated":
        selection._enroll_completed([(7, "uncompleted", 1_000_000, True)])
        at_us = 2_000_000
    else:
        at_us = 6_100_000
    assert selection._enroll_completed([(7, "let's start talking", at_us, False)]) == []
    assert selection.owner is None


def test_start_and_stop_controls_can_span_utterances():
    selection = _Selection(_SpeakerConfig())
    assert selection._enroll_completed([(7, "hey agent", 0, False)]) == []
    assert selection._enroll_completed([(7, "let us start talking", 1_000_000, False)]) == [{"kind": "enrolled"}]
    selection._activity({7}, 0.3, 2_000_000)
    assert selection._finish("hey agent")[-1]["kind"] == "control_pending"
    selection._activity({7}, 0.3, 3_000_000)
    assert selection._finish("let's stop talking")[-1] == {"kind": "released"}
