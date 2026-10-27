# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in aggregate diagnosis without media/identity capture or altered decisions."""

import io
import json
import logging
from types import SimpleNamespace

from gesture_capture_worker.agent import GestureCaptureAgent
from gesture_capture_worker.circle import PinchCircleTrigger
from gesture_capture_worker.config import CircleSettings
from gesture_capture_worker.diagnostics import GestureDiagnostics
from gesture_capture_worker.gesture import RegionSelector
from gesture_capture_worker.selection_wait import CircleCaptureWait
from PIL import Image
from test_circle_selection_wait import _frame, _path
from test_gesture_capture import _pinch, _settings
from xr_ai_tools.hand_tracking import HandTrackingResult
from xr_ai_tools.image import ImageRegistry


def _clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('gesture_capture_worker.diagnostics.time.monotonic', lambda: clock[0])
    return clock


def _payload(caplog):
    return json.loads(caplog.records[-1].getMessage())


def test_disabled_diagnostics_collect_and_emit_nothing(monkeypatch, caplog):
    clock = _clock(monkeypatch)
    diagnostics = GestureDiagnostics()
    diagnostics.count('no_hands_returned')
    diagnostics.number('pinch_ratio', 0.3)
    diagnostics.frame(100_000, 1, 23)
    diagnostics.grip('pinched')
    clock[0] = 3
    diagnostics.emit()
    assert not caplog.records and not diagnostics._counts and not diagnostics._ranges


def test_rate_limited_structured_summary_contains_numeric_aggregates_only(monkeypatch, caplog):
    clock = _clock(monkeypatch)
    caplog.set_level(logging.INFO, logger='gesture_capture_worker.diagnostics')
    diagnostics = GestureDiagnostics(True)
    for i in range(20):
        diagnostics.frame(i * 80_000, i % 2, 12 + i)
        diagnostics.number('pinch_ratio', i / 100)
        diagnostics.emit()
    assert not caplog.records
    clock[0] = 1
    diagnostics.grip('pinched')
    diagnostics.emit()
    payload = _payload(caplog)
    assert payload['counts']['frames'] == 20
    assert payload['counts']['hands_0'] == payload['counts']['hands_1'] == 10
    assert payload['ranges'] == {'backend_ms': [12, 31], 'frame_interval_ms': [80.0, 80.0],
                                 'pinch_ratio': [0, 0.19]}
    assert payload['grip'] == 'pinched'
    for _ in range(100):
        diagnostics.count('duplicate_frame')
        diagnostics.emit()
    assert len(caplog.records) == 1
    clock[0] = 2
    diagnostics.emit()
    assert len(caplog.records) == 2
    assert _payload(caplog)['counts'] == {'duplicate_frame': 100}


def test_real_selector_reasons_distinguish_missing_association_grip_and_geometry(monkeypatch, caplog):
    clock = _clock(monkeypatch)
    caplog.set_level(logging.INFO, logger='gesture_capture_worker.diagnostics')
    diagnostics = GestureDiagnostics(True)
    selector = RegionSelector(_settings(), diagnostics=diagnostics)
    selector.observe(HandTrackingResult(width=400, height=200, hands=[]), 80_000)
    ambiguous = _pinch(200, 100)
    ambiguous.hands *= 2
    selector.observe(ambiguous, 160_000)
    selector.observe(_pinch(200, 100), 240_000)
    selector.observe(_pinch(201, 100), 320_000)
    selector.observe(_pinch(0, 0), 400_000)
    selector.observe(_pinch(201, 100), 960_000)
    clock[0] = 1
    diagnostics.emit()
    counts = _payload(caplog)['counts']
    assert counts['no_hands_returned'] == 1
    assert counts['association_ambiguous'] == 1
    assert counts['associated'] >= 2 and counts['association_no_usable_nearby_hand'] == 1
    assert counts['grip_to_pinched'] == 1
    assert 'grip_to_open' not in counts
    assert counts['circle_reset'] >= 1
    assert counts['circle_min_points'] >= 1 and counts['selector_dimensions_or_frame_gap'] == 1


def test_enabled_diagnostics_do_not_change_completion(monkeypatch):
    _clock(monkeypatch)
    triggers = [PinchCircleTrigger(CircleSettings(), diagnostics=GestureDiagnostics(enabled))
                for enabled in (False, True)]
    results = [[trigger.observe(_pinch(*p), (i + 1) * 80_000)
                for i, p in enumerate(_path(310, 14))] for trigger in triggers]
    assert results[0] == results[1] and results[0][-1] is not None


async def test_agent_measures_real_boundary_without_logging_frame_or_identity(monkeypatch, caplog):
    clock = _clock(monkeypatch)
    caplog.set_level(logging.INFO, logger='gesture_capture_worker.diagnostics')
    diagnostics = GestureDiagnostics(True)
    images = ImageRegistry()
    output = io.BytesIO()
    Image.new('RGB', (400, 200), 'red').save(output, 'PNG')
    reference = images.put(output.getvalue(), owner='private-participant')
    async def frames(_):
        return SimpleNamespace(track_id='private-track', sequence=7, timestamp_us=123456789,
                               image=reference)
    async def hands(_):
        clock[0] = 1
        return HandTrackingResult(width=400, height=200, hands=[])
    agent = GestureCaptureAgent(frames=SimpleNamespace(execute=frames), hands=SimpleNamespace(execute=hands),
        capture=None, crop=None, query=None, selector=RegionSelector(_settings(), diagnostics=diagnostics),
        settings=_settings(), diagnostics=diagnostics)
    assert await agent.step('private-participant') is None
    text = caplog.records[-1].getMessage()
    payload = json.loads(text)
    assert payload['counts']['frames'] == payload['counts']['hands_0'] == 1
    assert payload['ranges']['backend_ms'] == [1000, 1000]
    for secret in ('private-participant', 'private-track', reference.uri, '123456789', 'PNG', 'landmarks'):
        assert secret not in text


def test_pending_timeout_outcome_is_reported(monkeypatch, caplog):
    from gesture_capture_worker.hand_motion import match_hand
    from xr_ai_tools.image_crop import ImageRegion

    clock = _clock(monkeypatch)
    caplog.set_level(logging.INFO, logger='gesture_capture_worker.diagnostics')
    diagnostics = GestureDiagnostics(True)
    observation = _pinch(220, 100)
    wait = CircleCaptureWait(ImageRegion(left=.35, top=.2, right=.65, bottom=.8),
                             match_hand(observation, None, 2), _frame(100_000), _settings(), diagnostics)
    for timestamp in range(200_000, 2_100_000, 100_000):
        assert not wait.observe(observation, _frame(timestamp))
    assert wait.observe(observation, _frame(2_100_000))
    clock[0] = 1
    diagnostics.emit()
    assert _payload(caplog)['counts']['pending_timeout'] == 1


def test_partial_circle_reports_geometry_stage_without_changing_decision(monkeypatch, caplog):
    clock = _clock(monkeypatch)
    caplog.set_level(logging.INFO, logger='gesture_capture_worker.diagnostics')
    diagnostics = GestureDiagnostics(True)
    trigger = PinchCircleTrigger(CircleSettings(), diagnostics=diagnostics)
    for i, point in enumerate(_path(240, 14)):
        assert trigger.observe(_pinch(*point), (i + 1) * 80_000) is None
    clock[0] = 1
    diagnostics.emit()
    counts = _payload(caplog)['counts']
    assert counts['circle_endpoint_gap'] > 0
    assert 'circle_recognized' not in counts


def test_decomposed_numeric_ranges_explain_ratio_and_trace_without_coordinates(monkeypatch, caplog):
    clock = _clock(monkeypatch)
    caplog.set_level(logging.INFO, logger='gesture_capture_worker.diagnostics')
    diagnostics = GestureDiagnostics(True)
    trigger = PinchCircleTrigger(CircleSettings(), diagnostics=diagnostics)
    for i, point in enumerate(_path(240, 14)):
        assert trigger.observe(_pinch(*point), (i + 1) * 80_000) is None
    clock[0] = 1
    diagnostics.emit()
    payload = _payload(caplog)
    assert payload['ranges']['tip_distance_px'] == [4, 4]
    assert payload['ranges']['palm_width_px'] == [40, 40]
    assert payload['ranges']['trace_width_px'][1] > 100
    assert payload['ranges']['trace_height_px'][1] > 100
    assert payload['ranges']['endpoint_gap_radius'][1] > 1
    assert all('pointer_x' not in key and 'pointer_y' not in key for key in payload['ranges'])


def test_rearm_diagnostics_distinguish_loss_from_completed_turn(monkeypatch, caplog):
    clock = _clock(monkeypatch)
    caplog.set_level(logging.INFO, logger='gesture_capture_worker.diagnostics')
    diagnostics = GestureDiagnostics(True)
    lost = PinchCircleTrigger(CircleSettings(), diagnostics=diagnostics)
    lost.observe(_pinch(200, 100), 80_000)
    lost.observe(_pinch(200, 100), 160_000)
    assert lost.observe(_pinch(200, 100), 2_800_000) is None
    completed = PinchCircleTrigger(CircleSettings(), diagnostics=diagnostics)
    for i, point in enumerate(_path(310, 14)):
        completed.observe(_pinch(*point), (i + 1) * 80_000)
    # Even a later continuity reset belongs to completed-turn rearm until open.
    completed.reset()
    assert completed.observe(_pinch(200, 100), 1_500_000) is None
    clock[0] = 1
    diagnostics.emit()
    counts = _payload(caplog)['counts']
    assert counts['fresh_attempt_after_loss'] >= 1
    assert counts['rearm_wait_after_completion'] >= 1


def test_peak_ratio_retains_its_own_numerator_and_denominator(monkeypatch, caplog):
    clock = _clock(monkeypatch)
    caplog.set_level(logging.INFO, logger='gesture_capture_worker.diagnostics')
    diagnostics = GestureDiagnostics(True)
    diagnostics.count('associated')
    diagnostics.pinch(1.0, 10, 10)
    diagnostics.pinch(.5, 20, 40)
    clock[0] = 1
    diagnostics.emit()
    assert _payload(caplog)['pinch_peak'] == {'ratio': 1.0, 'tip_distance_px': 10, 'palm_width_px': 10}
    diagnostics.count('no_hands_returned')
    clock[0] = 2
    diagnostics.emit()
    assert 'pinch_peak' not in _payload(caplog)
