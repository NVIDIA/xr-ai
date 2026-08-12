# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the named tool-call recovery in the shared LangChain adapter.

Recovery converts model text into executed tool calls, so its rejection
branches matter as much as its acceptance branches: a lone argument echo
must never become a mutation.
"""

from xr_ai_models import ToolDef
from xr_ai_nat.llm._langchain import _content_tool_call


def _tool(name, required, optional=()):
    properties = {key: {"type": "string"} for key in (*required, *optional)}
    return ToolDef(
        name=name,
        description="",
        parameters={"type": "object", "properties": properties, "required": list(required)},
    )


OFFERED = {
    tool.name: tool
    for tool in (
        _tool("object_ops__remove_object", ["object_words"]),
        _tool("object_ops__resize_object", ["object_words", "factor"]),
        _tool("placement_ops__nudge", ["object_words"], ["forward", "right", "up"]),
    )
}


def test_named_call_recovers():
    recovered = _content_tool_call(
        '{"name": "object_ops__resize_object", "arguments": {"object_words": "the cone", "factor": 2}}',
        OFFERED,
    )
    assert recovered and recovered["name"] == "object_ops__resize_object"
    assert recovered["args"] == {"object_words": "the cone", "factor": 2}


def test_wrapper_keys_unwrap():
    for wrapper in ("command", "function", "tool_call"):
        recovered = _content_tool_call(
            '{"%s": {"tool": "object_ops__resize_object", "args": {"object_words": "x", "factor": 2}}}' % wrapper,
            OFFERED,
        )
        assert recovered and recovered["name"] == "object_ops__resize_object"


def test_bare_name_suffix_match_requires_uniqueness():
    recovered = _content_tool_call('{"name": "nudge", "arguments": {"object_words": "x"}}', OFFERED)
    assert recovered and recovered["name"] == "placement_ops__nudge"
    two = dict(OFFERED)
    two["other_ops__nudge"] = _tool("other_ops__nudge", ["object_words"])
    assert _content_tool_call('{"name": "nudge", "arguments": {"object_words": "x"}}', two) is None


def test_string_encoded_arguments_parse():
    recovered = _content_tool_call(
        '{"name": "object_ops__remove_object", "arguments": "{\\"object_words\\": \\"the cone\\"}"}',
        OFFERED,
    )
    assert recovered and recovered["args"] == {"object_words": "the cone"}


def test_mixed_prose_is_not_recovered():
    text = 'I will not do that. {"name": "object_ops__remove_object", "arguments": {"object_words": "x"}}'
    assert _content_tool_call(text, OFFERED) is None


def test_unnamed_json_never_recovers():
    assert _content_tool_call('{"object_words": "the blue sphere"}', OFFERED) is None
    assert _content_tool_call('{"object_words": "the cone", "factor": 0.5}', OFFERED) is None


def test_recovery_ids_are_unique():
    first = _content_tool_call('{"name": "nudge", "arguments": {"object_words": "x"}}', OFFERED)
    second = _content_tool_call('{"name": "nudge", "arguments": {"object_words": "x"}}', OFFERED)
    assert first["id"] != second["id"]
