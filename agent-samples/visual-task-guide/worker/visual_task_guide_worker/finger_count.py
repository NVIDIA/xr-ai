# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse and present the task's compact VLM finger-count contract."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_COUNT = re.compile(r"\bCOUNT\s*=\s*(10|[0-9])\b", re.IGNORECASE)
_HANDS = re.compile(r"\bHANDS\s*=\s*([0-2])\b", re.IGNORECASE)
_CONFIDENCE = re.compile(r"\bCONFIDENCE\s*=\s*(high|medium|low)\b", re.IGNORECASE)
_NOTE = re.compile(r"\bNOTE\s*=\s*(.*?)(?:\s*$)", re.IGNORECASE)


class FingerCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0, le=10)
    hands: int = Field(ge=0, le=2)
    confidence: Literal["high", "medium", "low"]
    note: str = ""


def parse_finger_count(text: str) -> FingerCount | None:
    count = _COUNT.search(text)
    hands = _HANDS.search(text)
    confidence = _CONFIDENCE.search(text)
    if count is None or hands is None or confidence is None:
        return None
    note = _NOTE.search(text)
    return FingerCount(
        count=int(count.group(1)),
        hands=int(hands.group(1)),
        confidence=confidence.group(1).casefold(),
        note=note.group(1).strip(" .") if note else "",
    )


def format_finger_count(result: FingerCount) -> str:
    fingers = "finger" if result.count == 1 else "fingers"
    text = f"{result.count} extended {fingers} ({result.confidence} confidence)"
    return f"{text}. {result.note}." if result.note else f"{text}."


__all__ = ["FingerCount", "format_finger_count", "parse_finger_count"]
