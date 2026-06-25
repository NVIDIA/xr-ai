# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared text utilities for capability modules (lenient JSON extraction)."""
from __future__ import annotations

import json


def extract_json(text: str) -> str | None:
    clean = text.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) >= 2:
            clean = parts[1].lstrip("json").strip()
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(clean):
        if ch != "{":
            continue
        try:
            _obj, end = decoder.raw_decode(clean[idx:])
            return clean[idx:idx + end]
        except json.JSONDecodeError:
            pass

    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(clean):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return clean[start:i + 1]
    if start >= 0 and depth > 0 and not in_string and depth <= 3:
        return clean[start:] + ("}" * depth)
    return None
