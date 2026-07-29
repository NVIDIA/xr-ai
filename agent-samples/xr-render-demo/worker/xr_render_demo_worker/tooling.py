# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Module-level tool-call helpers for the xr-render-demo brain.

Pure functions shared between ``processors.py`` (the agentic loop) and
``agent.py`` (the XR-lifecycle render-mcp calls): unwrapping FastMCP tool
results, detecting leaked tool-call JSON in model text, extracting the first
balanced JSON object from a string, and flattening nested position/color args.
"""
from __future__ import annotations

import json


def tool_payload(result) -> dict | list | None:
    if hasattr(result, "data") and result.data is not None:
        return result.data
    return getattr(result, "structured_content", None)


_TOOL_CALL_KEY_SHAPES: tuple[frozenset[str], ...] = (
    frozenset({"name", "arguments"}),
    frozenset({"tool", "args"}),
    frozenset({"function", "arguments"}),
)


def looks_like_leaked_tool_call(text: str) -> bool:
    """True if *text* is a JSON object/array whose top level matches an
    OpenAI-style tool-call envelope (name+arguments, tool+args, or
    function+arguments). Plain prose that happens to start with "{" returns
    False; only parseable JSON with the right keys is sanitized.
    """
    obj_text = extract_json(text)
    if obj_text is None:
        return False
    try:
        obj = json.loads(obj_text)
    except json.JSONDecodeError:
        return False
    candidates: list[dict] = []
    if isinstance(obj, dict):
        candidates.append(obj)
    elif isinstance(obj, list):
        candidates.extend(c for c in obj if isinstance(c, dict))
    else:
        return False
    for c in candidates:
        keys = set(c.keys())
        if any(shape <= keys for shape in _TOOL_CALL_KEY_SHAPES):
            return True
    return False


def extract_json(text: str) -> str | None:
    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:        escape = False
            elif ch == "\\": escape = True
            elif ch == '"':  in_string = False
            continue
        if ch == '"':   in_string = True; continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth == 0: continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


def normalize_tool_args(args: dict) -> dict:
    """Flatten nested position/color dicts that the LLM sometimes generates.

    The LLM may produce {"position": {"x":0,"y":1.6,"z":-1.5}} because it
    pattern-matches the get_scene_state output format. Flatten to scalar kwargs
    so FastMCP validation passes.
    """
    args = dict(args)

    if "position" in args and isinstance(args["position"], dict):
        pos = args.pop("position")
        for k in ("x", "y", "z"):
            if k in pos and k not in args:
                args[k] = float(pos[k])

    if "color" in args and isinstance(args["color"], dict):
        col = args.pop("color")
        for k in ("r", "g", "b"):
            if k in col and k not in args:
                args[k] = float(col[k])

    # Strip None and empty-string values — the model sometimes emits r=''
    # when thinking is enabled and the value wasn't filled in.
    return {k: v for k, v in args.items() if v is not None and v != ""}
