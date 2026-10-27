# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separated finger framing, conservative layout priority and actual crop pixels."""

import io
import math

import pytest
from gesture_capture_worker.agent import GestureCaptureAgent
from gesture_capture_worker.gesture import RegionSelector, camera_box
from PIL import Image, ImageDraw
from test_gesture_capture import _box, _hand, _pinch, _settings
from xr_ai_tools import Tool
from xr_ai_tools.current_frame import CurrentFrameRequest, ImageFrame
from xr_ai_tools.hand_tracking import HandTrackingRequest, HandTrackingResult
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.image_crop import ImageCropTool
from xr_ai_tools.vision import ImageQueryTool


def _tip_box():
    return HandTrackingResult(width=400, height=200, hands=[
        _hand((.1, .3), (.25, .75), (.25, .25)),
        _hand((.9, .3), (.75, .75), (.75, .25)),
    ])


def _ray_box(gap):
    return HandTrackingResult(width=400, height=200, hands=[
        _hand((.2, .2), ((320 - gap) / 400, .2), (.2, (160 - gap * .4) / 200)),
        _hand((.8, .8), ((80 + gap * .7) / 400, .8), (.8, (40 + gap * .3) / 200)),
    ])


def _region(observation):
    return camera_box(observation, corner_tolerance=.35, angle_cosine=.45)


def _transform(observation, angle=0, mirror=False):
    radians = math.radians(angle)
    for hand in observation.hands:
        for point in hand.landmarks:
            x, y = point.x * 400 - 200, point.y * 200 - 100
            x, y = math.cos(radians) * x - math.sin(radians) * y, math.sin(radians) * x + math.cos(radians) * y
            point.x, point.y = (200 + (-x if mirror else x)) / 400, (100 + y) / 200
    return observation


@pytest.mark.parametrize('gap', [0, 30, 120, 180])
def test_opposed_rays_do_not_require_cross_hand_tip_contact(gap):
    region = _region(_ray_box(gap))
    assert region is not None
    assert (region.left, region.top, region.right, region.bottom) == pytest.approx((.215, .23, .785, .77), abs=.006)


def test_four_real_tip_corners_define_the_rectangle_with_palms_outside():
    region = _region(_tip_box())
    assert region is not None
    assert .25 < region.left < .3 and .7 < region.right < .75
    assert .25 < region.top < .3 and .7 < region.bottom < .75


def test_opposed_l_layout_wins_when_four_tips_also_form_a_rectangle():
    observation = HandTrackingResult(width=400, height=200, hands=[
        _hand((.1, .5), (.25, .75), (.25, .25)),
        _hand((.9, .5), (.75, .75), (.75, .25)),
    ])
    region = _region(observation)
    assert region is not None
    assert region.left > .3 and region.right < .7
    assert region.top < .25 and region.bottom > .75  # L-ray interior, not the shallower tip rectangle


def test_noisy_near_touch_clusters_keep_the_original_large_frame():
    observation = _box()
    observation.hands[1].landmarks[4].x += .005
    observation.hands[1].landmarks[8].x -= .005
    region = _region(observation)
    assert region is not None
    assert region.left < .25 and region.right > .75 and region.top < .3 and region.bottom > .7


@pytest.mark.parametrize('observation', [
    HandTrackingResult(width=400, height=200, hands=[
        _hand((.2, .2), (.4, .2), (.2, .4)), _hand((.6, .6), (.8, .6), (.6, .8))]),
    HandTrackingResult(width=400, height=200, hands=[
        _hand((.2, .2), (.05, .2), (.2, .05)), _hand((.8, .8), (.95, .8), (.8, .95))]),
    HandTrackingResult(width=400, height=200, hands=[
        _hand((.4, .5), (.25, .75), (.25, .25)), _hand((.6, .5), (.75, .75), (.75, .25))]),
    HandTrackingResult(width=400, height=200, hands=[
        _hand((.1, .5), (.25, .75), (.25, .25)), _hand((.9, .5), (.75, .9), (.6, .3))]),
])
def test_incompatible_outward_random_or_palm_inside_poses_are_not_frames(observation):
    assert _region(observation) is None


def test_two_pinches_and_duplicate_degenerate_hands_are_not_box_gestures():
    observation = _pinch(120, 100)
    observation.hands += _pinch(280, 100).hands
    assert _region(observation) is None
    observation = _box()
    observation.hands[1] = observation.hands[0]
    assert _region(observation) is None


@pytest.mark.parametrize('layout', ['tips', 'rays'])
@pytest.mark.parametrize('angle,mirror', [(0, False), (20, False), (-20, True)])
async def test_separated_layout_vlm_crop_keeps_target_and_excludes_fingers(layout, angle, mirror):
    from gesture_capture_worker import __main__ as worker
    from test_image_tools import _Vlm

    observation = _transform(_tip_box() if layout == 'tips' else _ray_box(180), angle, mirror)
    assert _region(observation) is not None
    image = Image.new('RGB', (400, 200), 'white')
    draw = ImageDraw.Draw(image)
    finger, target = (210, 160, 130), (0, 100, 220)
    for hand in observation.hands:
        for base_id, tip_id in ((2, 4), (5, 8)):
            base, tip = hand.landmarks[base_id], hand.landmarks[tip_id]
            draw.line([(base.x * 400, base.y * 200), (tip.x * 400, tip.y * 200)], fill=finger, width=10)
    draw.rectangle((190, 90, 209, 109), fill=target)
    output = io.BytesIO()
    image.save(output, 'PNG')
    images = ImageRegistry()
    timestamp = 100_000
    async def frame(_):
        return ImageFrame(image=images.put(output.getvalue()), width=400, height=200,
                          sequence=timestamp, timestamp_us=timestamp, participant_id='test')
    async def hands(_):
        return observation
    vlm = _Vlm('Target')
    agent = GestureCaptureAgent(
        frames=Tool('frame', 'Frame', CurrentFrameRequest, ImageFrame, frame),
        hands=Tool('hands', 'Hands', HandTrackingRequest, HandTrackingResult, hands), capture=None,
        crop=ImageCropTool(images=images), query=ImageQueryTool(images=images, vlm=vlm,
                                                              system_prompt=worker._REGION_SYSTEM_PROMPT),
        selector=RegionSelector(_settings(hold_s=.2)), settings=_settings(),
    )
    assert await agent.step('test') is None
    timestamp = 300_000
    assert await agent.step('test') == 'Target'
    sent_images, *_ = vlm.calls[0]
    with Image.open(io.BytesIO(sent_images[0])) as cropped:
        pixels = list(cropped.get_flattened_data())
        assert pixels.count(target) == 400 and finger not in pixels
