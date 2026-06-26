# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_capabilities.agent_monitor's completion grounding.

Regression coverage for the "student skipped a step but the agent passed it"
failure: a multi-object step ("put the glasses case next to the AirPod case")
must NOT complete on a frame that shows only one of the two required objects —
or an off-task frame — even when the VLM returns all-requirements-visible with
(fabricated) evidence. The fix grounds completion in the VLM's honest
``observation`` naming every key object, not just its per-requirement flags.
"""
from __future__ import annotations

import pytest
from xr_ai_capabilities.agent_monitor import FrameRef, check_guidance_step_complete
from xr_ai_models import ChatResponse

pytestmark = pytest.mark.asyncio


class _FakeVLM:
    """VLMService double: ask_image returns a scripted JSON string."""

    def __init__(self, json_response: str) -> None:
        self._json = json_response

    async def ask_image(self, image, question, **_kw) -> ChatResponse:
        return ChatResponse(content=self._json, reasoning=None, tool_calls=None,
                            finish_reason="stop", raw={})

    async def ask_images(self, images, question, **_kw) -> ChatResponse:
        return ChatResponse(content=self._json, reasoning=None, tool_calls=None,
                            finish_reason="stop", raw={})


async def _fake_frame(_pid: str) -> FrameRef:
    return FrameRef(image_path="/tmp/live.png", timestamp_us=1)


_STEP2_KEY_OBJECTS = [
    "blue silicone case with red zipper",
    "brown leather case with subtle logo embossed on lid",
]
_STEP2_REQUIREMENTS = [
    "blue silicone case with red zipper",
    "brown leather case with subtle logo embossed on lid",
]


async def _check(json_response: str):
    """Run a live-only (no teacher image) completion check for the 2-object
    "next to" step against a scripted VLM response."""
    return await check_guidance_step_complete(
        participant_id="p1",
        instruction="Put the glasses case next to the AirPod case.",
        expected_requirements=_STEP2_REQUIREMENTS,
        key_objects=_STEP2_KEY_OBJECTS,
        key_action="place next to",
        vlm=_FakeVLM(json_response),
        get_latest_frame=_fake_frame,
    )


async def test_missing_second_object_does_not_complete():
    """Frame shows only the AirPod (blue) case; VLM still marks every
    requirement visible with evidence. The brown leather case is absent from the
    observation, so the step must NOT complete."""
    out = await _check(
        '{"observation": "A blue AirPod case with a red zipper sits on the desk.",'
        ' "requirements": {'
        '   "blue silicone case with red zipper": {"visible": true, "evidence": "blue case with red zipper"},'
        '   "brown leather case with subtle logo embossed on lid": '
        '{"visible": true, "evidence": "a case is on the desk"}'
        ' }, "issue": ""}'
    )
    assert out.completed is False, out


async def test_off_task_frame_does_not_complete():
    """An off-task frame (a comb) with fabricated all-visible requirements must
    NOT complete — no key object appears in the observation."""
    out = await _check(
        '{"observation": "A wooden comb with a dark handle rests on the desk.",'
        ' "requirements": {'
        '   "blue silicone case with red zipper": {"visible": true, "evidence": "case visible"},'
        '   "brown leather case with subtle logo embossed on lid": {"visible": true, "evidence": "case visible"}'
        ' }, "issue": ""}'
    )
    assert out.completed is False, out


async def test_both_objects_present_completes():
    """Both required objects are named in the observation with grounded
    evidence → the step completes."""
    out = await _check(
        '{"observation": "A blue silicone case and a brown leather case sit side by side on the desk.",'
        ' "requirements": {'
        '   "blue silicone case with red zipper": {"visible": true, "evidence": "blue case with red zipper, left"},'
        '   "brown leather case with subtle logo embossed on lid": '
        '{"visible": true, "evidence": "brown leather case, right"}'
        ' }, "issue": ""}'
    )
    assert out.completed is True, out
