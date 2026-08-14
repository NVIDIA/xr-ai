# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the named tool-call recovery in the direct LLM loop.

Recovery converts model text into executed tool calls, so its rejection
branches matter as much as its acceptance branches: a lone argument echo
must never become a mutation.
"""

import json

from xr_ai_models import ToolDef
from xr_render_demo_worker._loop import _recover


def _tool(name, required, optional=()):
    properties = {key: {"type": "string"} for key in (*required, *optional)}
    return ToolDef(
        name=name,
        description="",
        parameters={"type": "object", "properties": properties, "required": list(required)},
    )


OFFERED = tuple(
    _tool(name, req, opt)
    for name, req, opt in (
        ("object_ops__remove_object", ("object_words",), ()),
        ("object_ops__resize_object", ("object_words", "factor"), ()),
        ("placement_ops__nudge", ("object_words",), ("forward", "right", "up")),
    )
)


def test_named_call_recovers():
    recovered = _recover(
        '{"name": "object_ops__resize_object", "arguments": {"object_words": "the cone", "factor": 2}}',
        OFFERED,
    )
    assert recovered and recovered.name == "object_ops__resize_object"
    assert json.loads(recovered.arguments) == {"object_words": "the cone", "factor": 2}


def test_wrapper_keys_unwrap():
    for wrapper in ("command", "function", "tool_call"):
        recovered = _recover(
            '{"%s": {"tool": "object_ops__resize_object", "args": {"object_words": "x", "factor": 2}}}' % wrapper,
            OFFERED,
        )
        assert recovered and recovered.name == "object_ops__resize_object"


def test_bare_name_suffix_match_requires_uniqueness():
    recovered = _recover('{"name": "nudge", "arguments": {"object_words": "x"}}', OFFERED)
    assert recovered and recovered.name == "placement_ops__nudge"
    extra = (*OFFERED, _tool("other_ops__nudge", ("object_words",)))
    assert _recover('{"name": "nudge", "arguments": {"object_words": "x"}}', extra) is None


def test_string_encoded_arguments_parse():
    recovered = _recover(
        '{"name": "object_ops__remove_object", "arguments": "{\\"object_words\\": \\"the cone\\"}"}',
        OFFERED,
    )
    assert recovered and json.loads(recovered.arguments) == {"object_words": "the cone"}


def test_mixed_prose_is_not_recovered():
    text = 'I will not do that. {"name": "object_ops__remove_object", "arguments": {"object_words": "x"}}'
    assert _recover(text, OFFERED) is None


def test_unnamed_json_never_recovers():
    assert _recover('{"object_words": "the blue sphere"}', OFFERED) is None
    assert _recover('{"object_words": "the cone", "factor": 0.5}', OFFERED) is None


def test_recovery_id_is_set():
    recovered = _recover('{"name": "nudge", "arguments": {"object_words": "x"}}', OFFERED)
    assert recovered and recovered.id
