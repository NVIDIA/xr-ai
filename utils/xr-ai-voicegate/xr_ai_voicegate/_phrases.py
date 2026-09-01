# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magic-phrase and STOP pattern matching for the voice gate."""
from __future__ import annotations

import re
from typing import Sequence


# Speech-interruption phrases matching this pattern bypass the magic-phrase
# gate. Classification tests raw text and, when a phrase is configured, its
# wake-stripped tail. Scoped application commands such as "stop recording"
# must not match: they need to reach the agent and its lifecycle tools.
STOP_RE: re.Pattern = re.compile(
    r"""
    ^\s*
    (?:
        (?:please|hey|okay|ok|uh|um|wait|no|just|alright|sorry|whoa)[,\s]+
        |hang\s+on[,\s]+
        |i\s+said[,\s]+
        |(?:can|could|would|will)\s+you[,\s]+
    ){0,2}
    (?:
        stop
        (?:
            (?:[,\s]+stop)+
            |[,\s]+(?:it|that|this)
            |[,\s]+doing[,\s]+(?:it|that|this)
            |[,\s]+(?:talking|speaking)
        )?
        (?:[,\s]+(?:please|already|now|right\s+now|for\s+now)){0,2}
        |be\s+quiet(?:\s+please)?
        |quiet(?:\s+please)?
        |shut\s+up(?:\s+please)?
    )
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def build_magic_pattern(phrases: Sequence[str]) -> re.Pattern | None:
    """Compile one sentence-boundary regex covering every configured phrase.

    Longest-first ordering picks the most specific match when one phrase
    is a prefix of another (e.g. "agent" vs "agent buddy"). Inside each
    phrase, the literal space between words is treated as "whitespace OR
    punctuation" so STT transcripts like "Hey, agent." still match the
    configured "hey agent". A phrase may begin the transcript or follow
    sentence-final punctuation separated by whitespace or a closing delimiter;
    commas and other mid-sentence punctuation do not open the gate. Returns
    ``None`` when ``phrases`` is empty so the gate degrades to always-on.
    """
    cleaned = tuple(p.strip().lower() for p in phrases if p and p.strip())
    if not cleaned:
        return None
    sep = r'[\s,.:;!?-]+'
    alts = "|".join(
        sep.join(re.escape(w) for w in p.split())
        for p in sorted(cleaned, key=len, reverse=True)
    )
    return re.compile(
        rf'(?:^[\s"\'\u2019\u201d)\]]*|'
        rf'[.!?][\s"\'\u2019\u201d)\]]+)'
        rf'(?:{alts})\b[\s,.:;!?-]*',
        re.IGNORECASE,
    )


def strip_magic(pattern: re.Pattern | None, text: str) -> str | None:
    """Return text after a sentence-boundary phrase, or ``None`` if absent.

    Text before a boundary match is discarded with the phrase so background
    speech never becomes part of the agent query. With ``pattern is None`` the
    gate is disabled and ``text`` is returned unchanged.
    """
    if pattern is None:
        return text
    m = pattern.search(text)
    return None if m is None else text[m.end():]
