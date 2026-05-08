# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for mono-slam-example: pose.py and pixels.py.

These tests run without any hub infrastructure.

Synthetic view strategy
-----------------------
We use cv2.warpAffine / cv2.warpPerspective to produce views with a known
pixel-space displacement from a rich structured base image.  ORB reliably
finds matches between views with small-to-moderate shifts on structured
texture, whereas random noise or sparse blob images do not survive ORB's
descriptor computation well.

Monocular ambiguity: recoverPose returns unit-norm translation.  We compare
only the direction (|dot| >= threshold), allowing for chirality sign flip.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from pose import (
    PoseResult,
    build_camera_matrix,
    compute_pose,
    rotation_to_euler_deg,
)


# ── texture helpers ─────────────────────────────────────────────────────────────

def _base_image(w: int = 640, h: int = 480, seed: int = 7) -> np.ndarray:
    """Structured checkerboard + noise — gives ORB rich, repeatable keypoints."""
    img = np.zeros((h, w), dtype=np.uint8)
    block = 20
    for i in range(0, h, block):
        for j in range(0, w, block):
            if (i // block + j // block) % 2 == 0:
                img[i : i + block, j : j + block] = 180
    rng = np.random.default_rng(seed)
    noise = rng.integers(-25, 25, (h, w), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _shifted_view(img: np.ndarray, tx_px: float, ty_px: float) -> np.ndarray:
    """Return img shifted by (tx_px, ty_px) pixels via affine warp."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx_px], [0, 1, ty_px]])
    return cv2.warpAffine(img, M, (w, h))


# ── rotation helpers ─────────────────────────────────────────────────────────────

def _rz(deg: float) -> np.ndarray:
    """Rotation matrix for a pure Z-axis rotation (yaw)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _ry(deg: float) -> np.ndarray:
    """Rotation matrix for a pure Y-axis rotation (pitch)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


# ── build_camera_matrix ────────────────────────────────────────────────────────

class TestBuildCameraMatrix:

    def test_shape(self):
        K = build_camera_matrix(640, 480)
        assert K.shape == (3, 3)

    def test_principal_point(self):
        K = build_camera_matrix(640, 480)
        assert K[0, 2] == pytest.approx(320.0)
        assert K[1, 2] == pytest.approx(240.0)

    def test_symmetric_focal_length_from_fov(self):
        K = build_camera_matrix(640, 480, fov_deg=60.0)
        assert K[0, 0] == pytest.approx(K[1, 1], rel=1e-9)
        # fx = w / (2 * tan(30°)) ≈ 554.3
        expected_f = 640 / (2.0 * math.tan(math.radians(30.0)))
        assert K[0, 0] == pytest.approx(expected_f, rel=1e-9)

    def test_override_focal_length(self):
        K = build_camera_matrix(640, 480, focal_length_px=600.0)
        assert K[0, 0] == pytest.approx(600.0)
        assert K[1, 1] == pytest.approx(600.0)


# ── rotation_to_euler_deg ──────────────────────────────────────────────────────

class TestRotationToEulerDeg:

    def test_identity_gives_zero_angles(self):
        roll, pitch, yaw = rotation_to_euler_deg(np.eye(3))
        assert roll  == pytest.approx(0.0, abs=1e-10)
        assert pitch == pytest.approx(0.0, abs=1e-10)
        assert yaw   == pytest.approx(0.0, abs=1e-10)

    def test_pure_yaw_90(self):
        R = _rz(90.0)
        roll, pitch, yaw = rotation_to_euler_deg(R)
        assert roll  == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert yaw   == pytest.approx(90.0, abs=1e-9)

    def test_pure_yaw_neg45(self):
        R = _rz(-45.0)
        _, _, yaw = rotation_to_euler_deg(R)
        assert yaw == pytest.approx(-45.0, abs=1e-9)

    def test_pure_pitch(self):
        R = _ry(30.0)
        roll, pitch, yaw = rotation_to_euler_deg(R)
        assert roll  == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(30.0, abs=1e-9)
        assert yaw   == pytest.approx(0.0, abs=1e-9)

    def test_round_trip_composed(self):
        """Composed rotation → euler → check individual component sign/magnitude."""
        R = _rz(20.0) @ _ry(10.0)
        roll, pitch, yaw = rotation_to_euler_deg(R)
        # With pure yaw+pitch, roll should be ~0.
        assert abs(roll) < 1.0
        assert pitch == pytest.approx(10.0, abs=1.0)
        assert yaw   == pytest.approx(20.0, abs=1.0)


# ── compute_pose ───────────────────────────────────────────────────────────────

class TestComputePoseDegenerate:

    def test_empty_frames_returns_not_ok(self):
        K = build_camera_matrix(64, 48)
        blank = np.zeros((48, 64), dtype=np.uint8)
        result = compute_pose(blank, blank, K)
        assert not result.ok
        assert result.num_inliers == 0

    def test_result_type(self):
        K = build_camera_matrix(64, 48)
        blank = np.zeros((48, 64), dtype=np.uint8)
        result = compute_pose(blank, blank, K)
        assert isinstance(result, PoseResult)
        assert result.R.shape == (3, 3)
        assert result.t.shape == (3,)

    def test_identity_fallback_on_failure(self):
        K = build_camera_matrix(64, 48)
        blank = np.zeros((48, 64), dtype=np.uint8)
        result = compute_pose(blank, blank, K)
        # When ok=False, R must be identity and t must be zero (safe accumulation).
        np.testing.assert_allclose(result.R, np.eye(3))
        np.testing.assert_allclose(result.t, np.zeros(3))

    def test_identical_frames_handled_gracefully(self):
        """Identical frames produce a degenerate Essential matrix — must not raise."""
        K = build_camera_matrix(640, 480)
        gray = _base_image(640, 480)
        # Degenerate: all correspondences have zero displacement; recoverPose
        # returns 0 inliers. The function must return a valid (ok=False) result.
        result = compute_pose(gray, gray, K)
        assert isinstance(result, PoseResult)
        # ok may be True or False depending on OpenCV internals, but no exception.
        assert result.R.shape == (3, 3)
        assert result.t.shape == (3,)
        assert np.isfinite(result.R).all()


class TestComputePoseTranslation:
    """Verify recoverPose recovers the correct translation direction.

    Synthetic views are produced by pixel-shifting a rich structured image.
    A lateral shift of tx_px pixels at focal length f, depth Z corresponds to
    3D translation t proportional to (tx_px/f, 0, 0) — i.e., pure X direction.
    recoverPose should return t ≈ ±[1, 0, 0] for a pure horizontal shift.
    """

    @pytest.mark.parametrize("tx_px,ty_px,expected_dir", [
        (25,  0, np.array([1.0, 0.0, 0.0])),   # pure rightward shift → +X
        ( 0, 20, np.array([0.0, 1.0, 0.0])),   # pure downward shift → +Y
    ])
    def test_translation_direction(self, tx_px, ty_px, expected_dir):
        K = build_camera_matrix(640, 480, fov_deg=60.0)
        base = _base_image(640, 480, seed=7)
        shifted = _shifted_view(base, tx_px, ty_px)

        result = compute_pose(base, shifted, K)
        assert result.ok, (
            f"compute_pose returned ok=False for pixel shift "
            f"(tx={tx_px}, ty={ty_px}), inliers={result.num_inliers}"
        )

        t_hat = result.t / np.linalg.norm(result.t)
        # Allow chirality sign flip.
        dot = abs(float(np.dot(t_hat, expected_dir)))
        assert dot > 0.9, (
            f"Translation direction mismatch: |dot|={dot:.3f}  "
            f"expected~{expected_dir}  got~{t_hat}"
        )


# ── pose accumulation math ─────────────────────────────────────────────────────

class TestPoseAccumulationMath:
    """Unit-test the T_curr_from_world accumulation formula in isolation.

    These tests verify the sign and direction of the accumulation without
    needing the ORB pipeline.  They use exact rotation matrices.
    """

    def _accumulate(self, steps: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        """Mirror the accumulation logic in agent.py _TrackState.

        Returns (R_world, t_world_cam) where:
          R_world     = R_curr_from_world
          t_world_cam = translation part of T_curr_from_world
          camera position in world = -R_world.T @ t_world_cam
        """
        R_world     = np.eye(3)
        t_world_cam = np.zeros(3)
        for R_step, t_step in steps:
            t_world_cam = R_step @ t_world_cam + t_step
            R_world     = R_step @ R_world
        return R_world, t_world_cam

    def test_single_identity_step(self):
        R, t_cam = self._accumulate([(np.eye(3), np.zeros(3))])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        # Camera position in world = -R.T @ t_cam = 0.
        np.testing.assert_allclose(-R.T @ t_cam, np.zeros(3), atol=1e-12)

    def test_two_forward_steps(self):
        """Two forward translations accumulate correctly in T_curr_from_world."""
        t_step = np.array([0.0, 0.0, 1.0])  # forward in camera frame
        R, t_cam = self._accumulate([(np.eye(3), t_step), (np.eye(3), t_step)])
        # T_curr_from_world translation: first step gives t=[0,0,1],
        # second step gives t = I@[0,0,1] + [0,0,1] = [0,0,2].
        np.testing.assert_allclose(t_cam, [0.0, 0.0, 2.0], atol=1e-12)
        # Camera position in world = -I.T @ [0,0,2] = [0,0,-2].
        np.testing.assert_allclose(-R.T @ t_cam, [0.0, 0.0, -2.0], atol=1e-12)

    def test_rotation_then_translation(self):
        """Rotate 90° around Y then translate forward: verify world-frame position."""
        R_step = _ry(90.0)
        t_step = np.array([0.0, 0.0, 1.0])  # forward in rotated camera frame

        R, t_cam = self._accumulate([(R_step, t_step)])
        pos = -R.T @ t_cam
        # Camera moved forward in rotated frame (world +X after 90° Y rotation).
        expected_pos = R_step.T @ (-t_step)
        np.testing.assert_allclose(pos, expected_pos, atol=1e-12)
        assert pos[0] > 0.9, f"Expected world-X>0.9 after 90deg Y rotation, got {pos}"

    def test_round_trip_inverse(self):
        """Applying a pose then its inverse should return to identity."""
        R_step = _rz(45.0)
        t_step = np.array([0.1, 0.0, 0.5])

        # Forward then backward: inverse is (R.T, -R.T @ t).
        R_inv   = R_step.T
        t_inv   = -R_step.T @ t_step

        R, t_cam = self._accumulate([(R_step, t_step), (R_inv, t_inv)])
        np.testing.assert_allclose(R,     np.eye(3), atol=1e-12)
        np.testing.assert_allclose(t_cam, np.zeros(3), atol=1e-12)


# ── pixels.py ─────────────────────────────────────────────────────────────────

class TestFrameToGray:
    """Verify pixel-format conversion produces the right grayscale output."""

    def _make_frame(self, fmt, data, w=4, h=4):
        from xr_ai_agent import FrameData
        return FrameData(
            seq=1, pts_us=0, width=w, height=h, fmt=fmt,
            data=bytes(data),
        )

    def test_rgb24_known_pixel(self):
        from pixels import frame_to_gray
        from xr_ai_agent import PixelFormat
        # Pure red (255,0,0) → BT.601 luma ≈ 76
        data = bytes([255, 0, 0] * 16)
        frame = self._make_frame(PixelFormat.RGB24, data)
        gray = frame_to_gray(frame)
        assert gray.shape == (4, 4)
        assert gray.dtype.kind == "u"
        # Red channel dominant — luma should be well above zero but not max.
        assert gray.mean() > 20

    def test_nv12_luma_extraction(self):
        from pixels import frame_to_gray
        from xr_ai_agent import PixelFormat
        # NV12: luma plane is first w*h bytes, set to 128.
        w, h = 4, 4
        luma = bytes([128] * (w * h))
        chroma = bytes([128] * (w * h // 2))  # UV interleaved
        frame = self._make_frame(PixelFormat.NV12, luma + chroma, w=w, h=h)
        gray = frame_to_gray(frame)
        assert gray.shape == (h, w)
        np.testing.assert_array_equal(gray, np.full((h, w), 128, dtype=np.uint8))

    def test_i420_luma_extraction(self):
        from pixels import frame_to_gray
        from xr_ai_agent import PixelFormat
        w, h = 4, 4
        luma = bytes([200] * (w * h))
        u = bytes([128] * (w * h // 4))
        v = bytes([128] * (w * h // 4))
        frame = self._make_frame(PixelFormat.I420, luma + u + v, w=w, h=h)
        gray = frame_to_gray(frame)
        assert gray.shape == (h, w)
        np.testing.assert_array_equal(gray, np.full((h, w), 200, dtype=np.uint8))

    def test_bgra_conversion(self):
        from pixels import frame_to_gray
        from xr_ai_agent import PixelFormat
        # Pure blue in BGRA: (255, 0, 0, 255) → luma ≈ 29
        data = bytes([255, 0, 0, 255] * 16)
        frame = self._make_frame(PixelFormat.BGRA, data, w=4, h=4)
        gray = frame_to_gray(frame)
        assert gray.shape == (4, 4)
        # Blue has low luma weight; should be low but not zero.
        assert gray.mean() < 60

    def test_unsupported_format_raises(self):
        from pixels import frame_to_gray
        from unittest.mock import MagicMock
        frame = MagicMock()
        frame.fmt = 99        # not a valid PixelFormat
        frame.width = 4
        frame.height = 4
        frame.data = b"\x00" * 16
        with pytest.raises(ValueError, match="Unsupported"):
            frame_to_gray(frame)
