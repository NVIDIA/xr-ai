# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magic-phrase and STOP pattern matching for the voice gate."""
from __future__ import annotations

import re
from typing import Sequence


# Transcripts matching this pattern bypass the magic-phrase gate so the
# user can interrupt a response mid-flight without having to start with
# the configured phrase.
STOP_RE: re.Pattern = re.compile(
    r'^\s*(?:\S+\s+){0,2}'               # 0–2 filler words before stop ("uh, stop"); 3+ = ordinary speech
    r'(?:stop(?:\s+\w+){0,2}|be\s+quiet|quiet|shut\s+up)'
    r'\s*[.!?]?\s*$',
    re.IGNORECASE,
)


def build_magic_pattern(phrases: Sequence[str]) -> re.Pattern | None:
    """Compile one strict-prefix regex covering every configured phrase.

    Longest-first ordering picks the most specific match when one phrase
    is a prefix of another (e.g. "agent" vs "agent buddy"). Inside each
    phrase, the literal space between words is treated as "whitespace OR
    punctuation" so STT transcripts like "Hey, agent." still match the
    configured "hey agent". Returns ``None`` when ``phrases`` is empty
    so the gate degrades to always-on.
    """
    cleaned = tuple(p.strip().lower() for p in phrases if p and p.strip())
    if not cleaned:
        return None
    sep = r'[\s,.:;!?-]+'
    alts = "|".join(
        sep.join(re.escape(w) for w in p.split())
        for p in sorted(cleaned, key=len, reverse=True)
    )
    return re.compile(rf'^\s*(?:{alts})\b[\s,.:;!?-]*', re.IGNORECASE)


def strip_magic(pattern: re.Pattern | None, text: str) -> str | None:
    """Return the transcript with the matched phrase stripped, or ``None``
    when no phrase is the strict prefix. With ``pattern is None`` the gate
    is disabled and ``text`` is returned unchanged."""
    if pattern is None:
        return text
    m = pattern.match(text)
    return None if m is None else text[m.end():]
