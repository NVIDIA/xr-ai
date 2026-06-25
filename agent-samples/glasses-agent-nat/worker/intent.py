# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-layer noise / intent gate for STT transcripts.

Layer 1 — ``is_shape_noise``
    Synchronous, ~free. Rejects transcripts that are obviously not a
    user request (too short, all filler, no letters).

Layer 2 — ``is_real_assistant_request``
    Single ~150 ms call to a small LLM. Used only when the worker has no
    other context to disambiguate (not recording, not guiding, no fresh
    demo on the table). Fail-open on transport / decode error: the worker
    is built to handle weird transcripts further down, and we'd rather
    answer a junk request than drop a real one.

Voice path only — data-channel text bypasses both layers (the client
explicitly sent it).
"""
from __future__ import annotations

import logging
import re

import httpx

log = logging.getLogger("glasses_agent_nat.intent")

# Pure filler tokens. If every word in the transcript is one of these, the
# user almost certainly said nothing useful (background "uh", phatic
# acknowledgements, etc.).
_FILLER: frozenset[str] = frozenset({
    "uh", "uhh", "um", "umm", "huh", "hmm", "mhm", "mm",
    "ah", "oh", "eh", "er", "err",
    "yeah", "yep", "yup", "no", "nope", "ok", "okay",
    "thanks", "thank", "you", "please",
    "the", "a", "an",
})

_WORD = re.compile(r"[A-Za-z']+")
_HAS_LETTER = re.compile(r"[A-Za-z]")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def is_shape_noise(text: str) -> bool:
    """Return True if *text* is obviously not a user request.

    Drops:
      * Strings shorter than 3 characters (whitespace-stripped).
      * Strings with no alphabetic characters (digits / punctuation only).
      * Strings whose every token is in ``_FILLER`` ("uh", "yeah", "the", ...).
    """
    s = text.strip()
    if len(s) < 3:
        return True
    if not _HAS_LETTER.search(s):
        return True
    toks = _tokens(s)
    if not toks:
        return True
    if all(t in _FILLER for t in toks):
        return True
    return False


_CLASSIFIER_SYSTEM = (
    "You are a noise-gate classifier for a wearable AI assistant. You see "
    "one STT transcript at a time. Answer with exactly one token: "
    "REQUEST if the transcript is the wearer asking the assistant for "
    "something (a question, a command, a stop, naming a demo, asking for "
    "guidance, asking what the camera sees, anything addressed to the "
    "assistant), or NOISE if it is background speech, mic crosstalk, "
    "filler, or an STT hallucination on silence. When in doubt answer "
    "REQUEST. Do not explain."
)


async def is_real_assistant_request(
    http: httpx.AsyncClient,
    llm_url: str,
    text: str,
    *,
    timeout: float = 2.0,
) -> bool:
    """Classify *text* as a real assistant request vs. background noise.

    Fail-open: any transport / decode error returns True so we err toward
    answering rather than dropping. Returns False only when the classifier
    explicitly says NOISE.
    """
    if not text.strip():
        return False
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user",   "content": text.strip()},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    try:
        resp = await http.post(
            llm_url.rstrip("/") + "/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        if resp.is_error:
            log.debug("intent classifier %s: %s", resp.status_code, resp.text[:120])
            return True
        body = resp.json()
        verdict = (
            body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .upper()
        )
    except Exception as exc:
        log.debug("intent classifier failed (fail-open): %s", exc)
        return True

    if verdict.startswith("NOISE"):
        return False
    return True
