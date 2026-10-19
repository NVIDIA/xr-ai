# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""openxr-service's headless OpenXR session, against a recording fake of pyopenxr.

The real bindings need a CloudXR runtime with a connected headset, so these
tests pin the OpenXR call sequence and handle cleanup instead.
"""
from __future__ import annotations

import ctypes
import enum
import importlib
import sys
import types
from types import SimpleNamespace

import pytest


class _Flags(enum.IntFlag):
    ORIENTATION_VALID_BIT = 1
    POSITION_VALID_BIT = 2


class _FormFactorUnavailableError(Exception):
    pass


class _Vector3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class _Pose(ctypes.Structure):
    _fields_ = [("position", _Vector3)]


class _JointLocation(ctypes.Structure):
    _fields_ = [("location_flags", ctypes.c_uint64), ("pose", _Pose)]


def _fake_xr(
    calls: list,
    *,
    system_error: Exception | None = None,
    location_flags: int = 3,
    hand_tracking: bool = False,
):
    xr = types.ModuleType("xr")
    xr.MND_HEADLESS_EXTENSION_NAME = "XR_MND_headless"
    xr.EXTX_OVERLAY_EXTENSION_NAME = "XR_EXTX_overlay"
    xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME = "XR_KHR_convert_timespec_time"
    xr.SpaceLocationFlags = _Flags
    xr.XrException = Exception
    xr.FormFactor = SimpleNamespace(HEAD_MOUNTED_DISPLAY="hmd")
    xr.ReferenceSpaceType = SimpleNamespace(STAGE="stage", VIEW="view")
    xr.ViewConfigurationType = SimpleNamespace(PRIMARY_MONO="mono", PRIMARY_STEREO="stereo")
    for name in ("InstanceCreateInfo", "ApplicationInfo", "SystemGetInfo", "SessionCreateInfo",
                 "ReferenceSpaceCreateInfo", "SessionBeginInfo", "SessionCreateInfoOverlayEXTX", "timespec"):
        setattr(xr, name, lambda name=name, **fields: SimpleNamespace(kind=name, **fields))
    xr.Version = lambda *parts: parts
    xr.Posef = lambda: "identity"
    xr.EXT_HAND_TRACKING_EXTENSION_NAME = "XR_EXT_hand_tracking"
    xr.HandEXT = SimpleNamespace(LEFT="left-hand", RIGHT="right-hand")
    xr.HandJointSetEXT = SimpleNamespace(DEFAULT="default")
    xr.HandTrackerCreateInfoEXT = lambda **fields: SimpleNamespace(**fields)
    xr.HandJointsLocateInfoEXT = lambda **fields: SimpleNamespace(**fields)
    xr.HandJointLocationEXT = _JointLocation
    xr.HandJointLocationsEXT = lambda **fields: SimpleNamespace(is_active=False, **fields)
    xr.HAND_JOINT_COUNT_EXT = 26
    offered = ["XR_MND_headless", "XR_EXT_hand_tracking"] if hand_tracking else ["XR_MND_headless"]
    xr.enumerate_instance_extension_properties = lambda: [
        SimpleNamespace(extension_name=name.encode()) for name in offered
    ]
    xr.create_hand_tracker_ext = lambda _session, info: f"{info.hand}-tracker"

    def create_instance(info):
        calls.append(("create_instance", tuple(info.enabled_extension_names)))
        return "instance"

    def get_system(_instance, _info):
        if system_error is not None:
            raise system_error
        return "system"

    def create_session(_instance, info):
        calls.append(("create_session", info.next.kind))
        return "session"

    def create_reference_space(_session, info):
        return f"{info.reference_space_type}-space"

    xr.create_instance = create_instance
    xr.get_system = get_system
    xr.create_session = create_session
    xr.create_reference_space = create_reference_space
    xr.enumerate_view_configurations = lambda *_: ["mono", "stereo"]
    xr.begin_session = lambda _s, info: calls.append(("begin_session", info.primary_view_configuration_type))
    xr.convert_timespec_time_to_time_khr = lambda _i, _t: 42
    xr.locate_space = lambda space, base, time: SimpleNamespace(
        location_flags=location_flags, pose=f"{space} in {base} at {time}"
    )
    for name in ("destroy_instance", "destroy_session", "destroy_space", "destroy_hand_tracker_ext"):
        setattr(xr, name, lambda handle, name=name: calls.append((name, handle)))
    return xr


@pytest.fixture
def load_session(monkeypatch):
    def load(xr):
        monkeypatch.setitem(sys.modules, "xr", xr)
        monkeypatch.delitem(sys.modules, "openxr_service.session", raising=False)
        return importlib.import_module("openxr_service.session")

    yield load
    sys.modules.pop("openxr_service.session", None)


def test_session_is_a_headless_overlay_begun_in_stereo_and_closed_newest_first(load_session):
    calls: list = []
    module = load_session(_fake_xr(calls))

    session = module.HeadlessSession("test")
    session.close()

    assert calls == [
        ("create_instance", ("XR_MND_headless", "XR_EXTX_overlay", "XR_KHR_convert_timespec_time")),
        ("create_session", "SessionCreateInfoOverlayEXTX"),
        ("begin_session", "stereo"),
        ("destroy_space", "view-space"),
        ("destroy_space", "stage-space"),
        ("destroy_session", "session"),
        ("destroy_instance", "instance"),
    ]


def test_head_is_the_view_located_in_stage_space_now(load_session):
    module = load_session(_fake_xr([]))

    head = module.HeadlessSession("test").head()

    assert head.is_valid
    assert head.pose == "view-space in stage-space at 42"


@pytest.mark.parametrize("flags", [0, 1, 2])
def test_head_without_both_valid_bits_is_invalid(load_session, flags):
    module = load_session(_fake_xr([], location_flags=flags))

    head = module.HeadlessSession("test").head()

    assert not head.is_valid
    assert head.pose == "identity"


def test_failed_open_releases_the_instance(load_session):
    calls: list = []
    module = load_session(_fake_xr(calls, system_error=_FormFactorUnavailableError("no headset")))

    with pytest.raises(_FormFactorUnavailableError):
        module.HeadlessSession("test")

    assert calls[-1] == ("destroy_instance", "instance")


def test_pose_source_reports_unavailable_until_a_headset_connects(load_session):
    module = load_session(_fake_xr([], system_error=_FormFactorUnavailableError("no headset")))
    source = module.HardwarePoseSource()

    first, second = source.get_pose(), source.get_pose()

    assert not first["is_valid"] and not second["is_valid"]
    assert "no headset" in first["error"]
    assert source.health() == {
        "status": "ok",
        "session_open": False,
        "open_attempts": 2,
        "last_open_error": "_FormFactorUnavailableError: no headset",
    }


def test_hand_trackers_are_created_when_offered_and_destroyed_first(load_session):
    calls: list = []
    module = load_session(_fake_xr(calls, hand_tracking=True))

    module.HeadlessSession("test").close()

    assert calls[0] == (
        "create_instance",
        ("XR_MND_headless", "XR_EXTX_overlay", "XR_KHR_convert_timespec_time", "XR_EXT_hand_tracking"),
    )
    assert calls[3:5] == [
        ("destroy_hand_tracker_ext", "right-hand-tracker"),
        ("destroy_hand_tracker_ext", "left-hand-tracker"),
    ]


def test_hands_report_joint_positions_and_validity_per_side(load_session, monkeypatch):
    module = load_session(_fake_xr([], hand_tracking=True))

    def locate(tracker, info, locations):
        assert (info.base_space, info.time) == ("stage-space", 42)
        if tracker == "right-hand-tracker":
            return  # inactive
        locations.is_active = True
        joints = locations.joint_locations
        joints[0].pose.position.x = 0.5
        joints[0].location_flags = 2  # position valid
        joints[1].location_flags = 1  # orientation only

    monkeypatch.setattr(module, "_locate_hand_joints", locate)

    hands = module.HeadlessSession("test").hands()

    assert hands["right"] is None
    left = hands["left"]
    assert len(left) == 26
    assert left[0][0].x == pytest.approx(0.5) and left[0][1] is True
    assert left[1][1] is False


def test_runtime_without_hand_tracking_still_tracks_the_head(load_session):
    calls: list = []
    module = load_session(_fake_xr(calls))

    session = module.HeadlessSession("test")

    assert "XR_EXT_hand_tracking" not in calls[0][1]
    assert session.hands() == {}
    assert session.head().is_valid
