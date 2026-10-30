# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict filler filtering must never remove meaningful participant speech."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-samples/conversation-example/worker"))
from conversation_example_worker._semantic_turns import _is_clear_filler, _Policy  # noqa: E402


@pytest.mark.parametrize("text", ["um", "uh", "Um… uh!", "umm uhh erm", "  uh  "])
def test_only_clear_filler_vocalizations_match(text):
    assert _is_clear_filler(text)


@pytest.mark.parametrize(
    "text",
    [
        "yes",
        "no",
        "okay",
        "hmm",
        "ah",
        "um, tell me the temperature",
        "What does um mean?",
        "like",
        "uh 42",
        "",
        "...",
        "uh α",
        "um@example.com",
    ],
)
def test_meaningful_words_and_ambiguous_acknowledgments_are_preserved(text):
    assert not _is_clear_filler(text)


@pytest.mark.parametrize(
    "kwargs", [{"stable_s": -1}, {"min_confidence": 1.1}, {"decision_timeout_s": 0}, {"max_sessions": True}]
)
def test_invalid_advisory_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _Policy(**kwargs)
