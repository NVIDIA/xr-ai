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


# ── distinguishing-object guard (stale/drifted key objects) ───────────────────
# Regression for the "step 2 passed though the student did it wrong" trace: a
# step's derived key objects drifted onto the PREVIOUS step's object (the teacher
# frame chosen for "put a mouse next to the AirPod case" still showed the case),
# so the still-in-view case satisfied the check without the student doing step 2.


async def _check_step2(json_response: str, *, prev_key_objects, key_objects):
    return await check_guidance_step_complete(
        participant_id="p1",
        instruction="put a mouse next to the AirPod case",
        expected_requirements=["AirPod case on desk", "AirPod case has red strap"],
        key_objects=key_objects,
        prev_key_objects=prev_key_objects,
        key_action="place next to",
        vlm=_FakeVLM(json_response),
        get_latest_frame=_fake_frame,
    )


async def test_drifted_key_objects_do_not_complete_on_carryover_object():
    """Step 2's key objects drifted to the prior step's AirPod case; the live
    frame shows a brown pouch + the carried-over AirPod case. Because the step
    adds no NEW object beyond the prior step's, completion must be blocked even
    though the (stale) AirPod-case requirements are grounded."""
    out = await _check_step2(
        '{"observation": "A bright blue AirPod case with a red strap sits on the desk '
        'next to a brown leather pouch.",'
        ' "requirements": {'
        '   "AirPod case on desk": {"visible": true, "evidence": "blue AirPod case on desk"},'
        '   "AirPod case has red strap": {"visible": true, "evidence": "red strap visible"}'
        ' }, "issue": ""}',
        prev_key_objects=["blue case"],
        key_objects=["AirPod case (bright blue with red strap)"],
    )
    assert out.completed is False, out


async def test_new_object_present_still_completes():
    """When the step's key objects DO introduce a new object vs. the previous
    step and that object is named in the observation, completion still works —
    the guard does not regress the happy path."""
    out = await _check_step2(
        '{"observation": "A black wireless mouse sits on the desk beside the blue case.",'
        ' "requirements": {'
        '   "wireless mouse on desk": {"visible": true, "evidence": "black wireless mouse on desk"}'
        ' }, "issue": ""}',
        prev_key_objects=["blue case"],
        key_objects=["black wireless mouse"],
    )
    assert out.completed is True, out
