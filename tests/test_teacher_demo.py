# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_capabilities.teacher_demo's per-step key-info derivation.

Regression coverage for the "step 2 passed though the student did it wrong"
failure: for a relational placement step ("put a mouse next to the AirPod case")
the LLM — told to prefer the teacher frame caption, which described the frame
showing only the already-placed AirPod case — derived key objects naming the
ANCHOR (the AirPod case) instead of the object being PLACED (the mouse). Live
monitoring then verified the anchor that was already in view and auto-completed.
The fix forces the verb's direct object to be the primary key object and demotes
the anchor to position.
"""
from __future__ import annotations

import pytest
from xr_ai_capabilities.teacher_demo import derive_step_key_info
from xr_ai_models import ChatResponse

pytestmark = pytest.mark.asyncio


class _FakeLLM:
    """LLMService double: chat returns a scripted JSON string."""

    def __init__(self, json_response: str) -> None:
        self._json = json_response

    async def chat(self, messages, **_kw) -> ChatResponse:
        return ChatResponse(content=self._json, reasoning=None, tool_calls=None,
                            finish_reason="stop", raw={})


async def test_placement_step_keys_on_placed_object_not_anchor():
    """The LLM drifts onto the anchor (AirPod case) for "put a mouse next to the
    AirPod case". Derivation must force the PLACED object (mouse) to be primary
    so monitoring grounds on the mouse, not the carried-over anchor."""
    info = await derive_step_key_info(
        "put a mouse next to the AirPod case",
        teacher_caption="A bright blue AirPod case with a red strap on the desk.",
        llm=_FakeLLM(
            '{"objects": ["AirPod case (bright blue with red strap)"],'
            ' "action": "place next to", "position": "centered on the desk",'
            ' "target_state": "AirPod case centered on the desk", "ignore": []}'
        ),
    )
    assert info.objects, info
    assert "mouse" in info.objects[0].lower(), info
    # The anchor must not be the object the monitor verifies present.
    assert not any("airpod" in o.lower() for o in info.objects), info


async def test_placement_step_keeps_correct_llm_object():
    """When the LLM already names the placed object, derivation leaves it as the
    primary key object (no regression, no spurious duplication)."""
    info = await derive_step_key_info(
        "put a black wireless mouse next to the AirPod case",
        teacher_caption="A black wireless mouse beside a blue case.",
        llm=_FakeLLM(
            '{"objects": ["black wireless mouse"], "action": "place next to",'
            ' "position": "next to the AirPod case", "target_state": "mouse on desk",'
            ' "ignore": []}'
        ),
    )
    assert info.objects[0].lower() == "black wireless mouse", info


async def test_non_placement_step_unchanged():
    """A non-relational step is passed through untouched — the backstop only
    fires for "put/place X <prep> Y"."""
    info = await derive_step_key_info(
        "put on the headset",
        teacher_caption="A person wearing a black headset.",
        llm=_FakeLLM(
            '{"objects": ["black headset"], "action": "place on head",'
            ' "position": "strap around back of head", "target_state": "headset on head",'
            ' "ignore": []}'
        ),
    )
    assert info.objects == ["black headset"], info
