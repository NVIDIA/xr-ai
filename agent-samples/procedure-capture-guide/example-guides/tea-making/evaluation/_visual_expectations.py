# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share recognition expectations between this sample's visual and guide evaluators.

Expected answers come from the case manifest, not model responses. Recognition
ignores whitespace layout; CLEF independently judges the original description against the guide condition.
"""

import re


def matches_answer(pattern: str, text: str) -> bool:
    return re.fullmatch(pattern, " ".join(text.split()), re.IGNORECASE) is not None


def visual_cases(manifest: dict) -> dict[str, dict]:
    cases = {}
    for case in manifest["cases"]:
        name = case["scene"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name) or name in cases:
            raise ValueError("visual cases require unique, safe scene names")
        if not isinstance(case["accept"], bool) or not isinstance(case["answer"], str):
            raise ValueError("cases require a boolean accept and a string answer regex")
        re.compile(case["answer"])
        cases[name] = case
    if not cases:
        raise ValueError("visual cases must not be empty")
    return cases
