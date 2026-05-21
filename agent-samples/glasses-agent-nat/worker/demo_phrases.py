# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure phrase-matching helpers for demo / guidance intent detection.

All inputs are pre-lowercased, punctuation-stripped utterance text.
No state, no I/O — easy to unit-test in isolation.
"""
from __future__ import annotations

import string
import time

_DEMO_START_PHRASES = (
    # Explicit start commands
    "start recording",
    "start record",
    "star recording",
    "star record",
    "begin recording",
    "start demo",
    "start a demo",
    "begin demo",
    # Natural "record X" phrasing
    "record demo",
    "record a demo",
    "record steps",
    "record how",
    "record me",
    # Capture / save variants
    "capture a demo",
    "capture how",
    "save these steps",
    "save this",
    # Show / demonstrate
    "let me show you",
    "watch what i",
    "watch me",
    "watch how i",
    "i'll demonstrate",
    "demonstrate how",
    # Remember variants
    "remember this",
    "remember how",
    "remember these",
    "remember steps",
    "can you remember",
)

_DEMO_NAME_PREFIXES = (
    "for task where ",
    "for the task where ",
    "for task ",
    "for the task ",
    "for task ",
    "for the task ",
    "called ",
    "named ",
    "where ",
    "the task ",
    "task ",
    "a ",
    "the ",
)

DEMO_END_PHRASES = (
    "that's it",
    "stop recording",
    "end demonstration",
    "finished",
    "end recording",
    "finish recording",
    # "done" is short — match only if it's a standalone word to avoid false positives
)

_GUIDANCE_PHRASES = (
    "how do i",
    "walk me through",
    "show me how",
    "teach me how to",
    "guide me through",
    "step by step",
)

_GUIDANCE_ADVANCE_PHRASES = (
    "next",
    "continue",
    "got it",
    "okay next",
    "ok next",
    "next step",
    "go on",
)

_GUIDANCE_DONE_PHRASES = (
    "done",
    "finished",
    "all done",
    "that's it",
    "complete",
    "stop guiding",
    "stop guide",
    "stop guidance",
    "exit guidance",
    "exit",
    "stop",
    "quit",
    "cancel",
)


def is_demo_end(lower: str) -> bool:
    for phrase in DEMO_END_PHRASES:
        if phrase in lower:
            return True
    # "done" alone (not part of a longer phrase)
    if lower.strip() == "done" or lower.startswith("done ") or lower.endswith(" done"):
        return True
    return False


def extract_demo_name(lower: str) -> str | None:
    """Return the demo name if a start phrase is detected, else None.

    Heuristic: the demo name is whatever comes after the start phrase.
    If nothing follows (bare trigger), use a timestamp-based name.
    """
    for phrase in _DEMO_START_PHRASES:
        if phrase in lower:
            after = lower.split(phrase, 1)[1].strip().rstrip(string.punctuation).strip()
            for prefix in _DEMO_NAME_PREFIXES:
                if after.startswith(prefix):
                    after = after[len(prefix):].strip().rstrip(string.punctuation).strip()
                    break
            if after and len(after) > 2:
                return after
            ts = time.strftime("%H%M%S")
            return f"demo-{ts}"
    return None


def match_guidance_request(lower: str) -> str | None:
    """Return the user query if a guidance phrase is detected, else None."""
    for phrase in _GUIDANCE_PHRASES:
        if phrase in lower:
            return lower
    return None


def is_guidance_advance(lower: str) -> bool:
    for phrase in _GUIDANCE_ADVANCE_PHRASES:
        if lower.strip() == phrase or lower.strip().startswith(phrase):
            return True
    return False


def is_guidance_done(lower: str) -> bool:
    # Exact match for multi-word phrases; word-boundary match for single words
    # so "stop, stop" / "stop recording" / "okay stop" all exit guidance.
    words = set(lower.replace(",", " ").replace(".", " ").split())
    for phrase in _GUIDANCE_DONE_PHRASES:
        if lower.strip() == phrase:
            return True
        if " " not in phrase and phrase in words:
            return True
    return False
