# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Monocular visual odometry — feature matching and pose recovery.

All functions are pure (no I/O, no global state) so they are directly
unit-testable without any hub infrastructure.

Coordinate conventions
----------------------
- Camera frame: X right, Y down, Z forward (OpenCV standard).
- R, t returned by ``compute_pose`` encode the pose of the CURRENT camera
  relative to the PREVIOUS camera: p_curr = R @ p_prev + t.
  This is the convention returned by cv2.recoverPose (pts1 = previous,
  pts2 = current).
- Translation has unit norm (monocular scale ambiguity): the pipeline
  accumulates direction only, not metric distance.
- Euler angles use ZYX intrinsic convention (yaw → pitch → roll applied in
  body frame), reported in degrees for human readability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseResult:
    """Recovered relative pose between two consecutive frames.

    R:           3x3 rotation matrix, current camera relative to previous.
                 (OpenCV camera frame: X right, Y down, Z forward)
    t:           3-element unit translation vector (monocular — scale unknown).
    num_inliers: RANSAC inlier count from recoverPose.
    ok:          False when fewer than min_inliers inliers were found or
                 when feature matching failed; R and t are identity/zero.
    """
    R:           np.ndarray   # shape (3, 3)
    t:           np.ndarray   # shape (3,)
    num_inliers: int
    ok:          bool


def build_camera_matrix(
    width: int,
    height: int,
    fov_deg: float = 60.0,
    focal_length_px: float | None = None,
) -> np.ndarray:
    """Construct a pinhole camera matrix K from frame dimensions.

    Args:
        width:           Frame width in pixels.
        height:          Frame height in pixels.
        fov_deg:         Horizontal FOV in degrees (ignored when
                         ``focal_length_px`` is provided).
        focal_length_px: If given, used directly as fx = fy.

    Returns:
        3x3 float64 camera matrix K.
    """
    if focal_length_px is not None:
        f = float(focal_length_px)
    else:
        # f = w / (2 * tan(hfov/2))
        f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [f,   0.0, cx],
        [0.0, f,   cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rotation_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Convert a rotation matrix to ZYX intrinsic Euler angles in degrees.

    ZYX intrinsic = extrinsic XYZ: yaw (Z) applied first in body frame,
    then pitch (Y), then roll (X).  Returns (roll, pitch, yaw) in degrees.

    Gimbal lock occurs when |pitch| = 90°; the returned roll is set to 0
    and yaw absorbs the combined rotation in that degenerate case.
    """
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  →  ZYX intrinsic
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6  # near gimbal lock
    if not singular:
        roll  = math.degrees(math.atan2( R[2, 1], R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = math.degrees(math.atan2( R[1, 0], R[0, 0]))
    else:
        roll  = 0.0
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
    return roll, pitch, yaw


def compute_pose(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    K: np.ndarray,
    *,
    max_features: int = 500,
    match_ratio: float = 0.75,
    ransac_prob: float = 0.999,
    ransac_threshold: float = 1.0,
    min_inliers: int = 20,
) -> PoseResult:
    """Estimate the relative pose between two grayscale frames.

    Uses ORB keypoints, BFMatcher with ratio test, Essential matrix
    estimation (RANSAC), and ``cv2.recoverPose``.

    Args:
        prev_gray:        Previous frame as a 2D uint8 grayscale array (H, W).
        curr_gray:        Current frame as a 2D uint8 grayscale array (H, W).
        K:                3x3 camera intrinsic matrix (float64).
        max_features:     ORB keypoint budget per frame.
        match_ratio:      Lowe ratio-test threshold.
        ransac_prob:      RANSAC confidence level for findEssentialMat.
        ransac_threshold: Reprojection threshold in pixels for RANSAC.
        min_inliers:      Minimum inliers to accept the pose as valid.

    Returns:
        PoseResult with ok=True when the pose is reliable.
    """
    _FAIL = PoseResult(
        R=np.eye(3), t=np.zeros(3), num_inliers=0, ok=False,
    )

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return _FAIL

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe ratio test filters ambiguous matches.
    good = [m for m, n in raw_matches if m.distance < match_ratio * n.distance]
    if len(good) < 8:
        return _FAIL

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, inlier_mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=ransac_prob,
        threshold=ransac_threshold,
    )
    if E is None or inlier_mask is None:
        return _FAIL

    # recoverPose: pts1 are from the reference (previous) camera,
    # pts2 are from the new (current) camera.
    # Returns R, t such that: p_curr = R @ p_prev + t  (OpenCV convention).
    # t is always unit-norm — monocular scale is unobservable.
    inliers, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=inlier_mask)

    if inliers < min_inliers:
        return _FAIL

    # Validate R is a proper rotation (det ≈ +1, no NaN).
    if not np.isfinite(R).all() or abs(np.linalg.det(R) - 1.0) > 0.01:
        return _FAIL

    return PoseResult(
        R=R,
        t=t.reshape(3),
        num_inliers=int(inliers),
        ok=True,
    )
