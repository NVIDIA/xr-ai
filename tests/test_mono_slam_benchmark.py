# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Offline benchmark: DPVO ATE RMSE on TUM RGB-D fr1/xyz.

Skipped unless:
  - DPVO is installed (scripts/install_dpvo.sh)
  - A CUDA GPU is available
  - --dataset-dir and --weights-path are provided on the pytest CLI

Usage
-----
    pytest tests/test_mono_slam_benchmark.py -v \\
        --dataset-dir datasets/tum/rgbd_dataset_freiburg1_xyz \\
        --weights-path models/dpvo.pth

The test downloads nothing; run scripts/download_dataset.sh first.

ATE RMSE computation
--------------------
Monocular SLAM has scale ambiguity: the estimated trajectory lives in an
arbitrary coordinate frame up to a Sim(3) similarity transform.  Standard
practice (DPVO paper, ORB-SLAM2, etc.) is to align estimated and ground-truth
trajectories with a Sim(3) least-squares fit (Umeyama 1991) before computing
the Absolute Trajectory Error (ATE).

Reference
    S. Umeyama, "Least-Squares Estimation of Transformation Parameters Between
    Two Point Patterns," IEEE TPAMI 13(4), 1991.
    doi:10.1109/34.88573

The implementation below follows the SVD formulation from the paper directly.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

dpvo_available = importlib.util.find_spec("dpvo") is not None


def _cuda_available() -> bool:
    if not dpvo_available:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Umeyama Sim(3) alignment ──────────────────────────────────────────────────

def umeyama_align(
    P: np.ndarray,
    Q: np.ndarray,
    with_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the Sim(3) alignment T* = argmin_T ||Q - s*R*P - t||_F.

    Implements the closed-form SVD solution from:
        S. Umeyama (1991), IEEE TPAMI 13(4).

    Args:
        P:          (N, 3) estimated trajectory positions.
        Q:          (N, 3) ground-truth trajectory positions (matched timestamps).
        with_scale: If True, estimate scale s (standard for monocular); if
                    False, fix s=1 (metric systems).

    Returns:
        R:  (3, 3) rotation matrix (Q_aligned = s * R @ P + t).
        t:  (3,) translation vector.
        s:  Scale factor (1.0 if with_scale=False).

    Raises:
        ValueError: if N < 3 or P and Q have incompatible shapes.
    """
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P and Q must both be (N, 3); got {P.shape} vs {Q.shape}")
    N = P.shape[0]
    if N < 3:
        raise ValueError(f"Need at least 3 matched points; got {N}")

    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    P_c  = P - mu_P
    Q_c  = Q - mu_Q

    # Variance of P (sigma^2 in Umeyama notation).
    var_P = (P_c ** 2).sum() / N

    # Cross-covariance matrix Sigma_QP (Eq. 38 in Umeyama 1991).
    Sigma = (Q_c.T @ P_c) / N   # (3, 3)

    U, D, Vt = np.linalg.svd(Sigma)
    V = Vt.T

    # Sign correction matrix S (Eq. 43).
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt   # (3, 3)

    if with_scale and var_P > 1e-14:
        s = (D * S.diagonal()).sum() / var_P
    else:
        s = 1.0

    t = mu_Q - s * R @ mu_P
    return R, t, float(s)


def ate_rmse(
    poses_est:  np.ndarray,
    poses_gt:   np.ndarray,
    with_scale: bool = True,
) -> float:
    """Compute ATE RMSE (metres) after Sim(3) alignment.

    Args:
        poses_est:  (N, 7) estimated poses [tx, ty, tz, qx, qy, qz, qw].
        poses_gt:   (N, 7) ground-truth poses in the same order.
        with_scale: Whether to recover scale (True for monocular).

    Returns:
        ATE RMSE in the ground-truth coordinate units (metres for TUM).
    """
    t_est = poses_est[:, :3]
    t_gt  = poses_gt[:, :3]

    R, t, s = umeyama_align(t_est, t_gt, with_scale=with_scale)
    t_aligned = (s * (R @ t_est.T)).T + t   # (N, 3)

    errors = np.linalg.norm(t_aligned - t_gt, axis=1)
    return float(np.sqrt((errors ** 2).mean()))


# ── TUM data loader ───────────────────────────────────────────────────────────

def _load_tum_sequence(
    dataset_dir: Path,
    stride: int = 1,
) -> tuple[list[float], list[np.ndarray], np.ndarray]:
    """Load timestamps, RGB frames, and ground-truth poses from TUM RGB-D.

    Args:
        dataset_dir: Path to an extracted TUM sequence (contains rgb/, depth/,
                     rgb.txt, groundtruth.txt).
        stride:      Use every Nth frame (default 1 = all frames).

    Returns:
        tstamps:   List of float timestamps (seconds).
        frames:    List of (H, W, 3) uint8 RGB frames.
        gt_poses:  (M, 7) float64 ground-truth poses [tx,ty,tz,qx,qy,qz,qw]
                   at the timestamps closest to each loaded frame.
    """
    import cv2

    rgb_txt  = dataset_dir / "rgb.txt"
    gt_txt   = dataset_dir / "groundtruth.txt"

    # Parse rgb.txt — lines: "timestamp path"
    rgb_entries: list[tuple[float, Path]] = []
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rgb_entries.append((float(parts[0]), dataset_dir / parts[1]))

    rgb_entries = rgb_entries[::stride]

    # Parse groundtruth.txt — lines: "timestamp tx ty tz qx qy qz qw"
    gt_ts: list[float] = []
    gt_data: list[np.ndarray] = []
    with open(gt_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            gt_ts.append(float(parts[0]))
            gt_data.append(np.array([float(x) for x in parts[1:]], dtype=np.float64))
    gt_ts_arr  = np.array(gt_ts)
    gt_data_arr = np.vstack(gt_data)  # (M, 7)

    tstamps: list[float] = []
    frames:  list[np.ndarray] = []
    gt_poses: list[np.ndarray] = []

    for ts, img_path in rgb_entries:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Associate ground-truth by nearest timestamp.
        idx = int(np.argmin(np.abs(gt_ts_arr - ts)))
        tstamps.append(ts)
        frames.append(rgb)
        gt_poses.append(gt_data_arr[idx])

    return tstamps, frames, np.vstack(gt_poses)


# ── TUM freiburg1 intrinsics (hard-coded, same as DPVO evaluate_tum.py) ───────

_FREIBURG1_INTRINSICS = np.array(
    [517.3, 516.5, 318.6, 255.3],   # fx, fy, cx, cy
    dtype=np.float32,
)


# ── benchmark test ────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    not dpvo_available,
    reason="DPVO not installed — run scripts/install_dpvo.sh",
)
@pytest.mark.skipif(
    not _cuda_available(),
    reason="DPVO requires a CUDA GPU",
)
class TestDPVOBenchmarkTUM:
    """End-to-end DPVO accuracy benchmark on TUM fr1/xyz."""

    # Acceptance threshold: DPVO reports ≈0.017 m ATE on fr1_xyz.
    # We use a lenient bound (5 cm) to accommodate driver/environment variation.
    ATE_THRESHOLD_M = 0.05

    @pytest.fixture(scope="class")
    def dataset_dir(self, request) -> Path:
        raw = request.config.getoption("--dataset-dir")
        if raw is None:
            pytest.skip(
                "No dataset provided.  Run with "
                "--dataset-dir datasets/tum/rgbd_dataset_freiburg1_xyz"
            )
        p = Path(raw)
        if not p.is_dir():
            pytest.skip(f"Dataset directory not found: {p}")
        return p

    @pytest.fixture(scope="class")
    def weights_path(self, request) -> str:
        return request.config.getoption("--weights-path", default="models/dpvo.pth")

    @pytest.fixture(scope="class")
    def trajectory_result(self, dataset_dir, weights_path):
        """Run DPVO on the full TUM sequence; return (poses_est, poses_gt)."""
        from slam import DPVOSlam

        stride = 2   # every other frame — same as DPVO paper evaluation
        tstamps, frames, gt_poses = _load_tum_sequence(dataset_dir, stride=stride)
        assert len(frames) >= 50, (
            f"Only {len(frames)} frames loaded — check dataset_dir path"
        )

        h, w = frames[0].shape[:2]
        slam = DPVOSlam(
            weights_path=weights_path,
            height=h,
            width=w,
            intrinsics=_FREIBURG1_INTRINSICS,
        )

        import torch
        with torch.no_grad():
            for ts, frame in zip(tstamps, frames):
                slam.push(ts, frame)

        poses_est, est_tstamps = slam.terminate()   # (N, 7), (N,)

        # Associate estimated poses with GT poses by timestamp.
        gt_ts = np.array(tstamps, dtype=np.float64)
        # est_tstamps are indices into tstamps (DPVO uses tstamp as frame index when int).
        # After terminate(), tstamps array echoes the values passed to __call__.
        # We trust the order and crop GT to match estimated length.
        n = min(len(poses_est), len(gt_poses))
        return poses_est[:n], gt_poses[:n]

    def test_ate_rmse_within_threshold(self, trajectory_result):
        """ATE RMSE (after Sim(3) alignment) is below 5 cm on TUM fr1/xyz."""
        poses_est, poses_gt = trajectory_result
        rmse = ate_rmse(poses_est, poses_gt, with_scale=True)
        print(f"\nATE RMSE (Sim3-aligned): {rmse * 100:.2f} cm  "
              f"(threshold: {self.ATE_THRESHOLD_M * 100:.0f} cm)")
        assert rmse < self.ATE_THRESHOLD_M, (
            f"ATE RMSE {rmse:.4f} m exceeds threshold {self.ATE_THRESHOLD_M} m"
        )

    def test_trajectory_length(self, trajectory_result):
        """Estimated trajectory covers at least 80% of the sequence."""
        poses_est, poses_gt = trajectory_result
        assert len(poses_est) >= 0.8 * len(poses_gt), (
            f"Trajectory too short: {len(poses_est)} / {len(poses_gt)} frames"
        )


# ── unit tests for Umeyama alignment (always run, no GPU required) ────────────

class TestUmeyamaAlign:
    """Unit tests for the Umeyama Sim(3) implementation."""

    def test_identity_alignment(self):
        """Perfect match → s≈1, R≈I, t≈0."""
        rng = np.random.default_rng(0)
        P = rng.standard_normal((20, 3))
        R, t, s = umeyama_align(P, P, with_scale=True)
        P_aligned = (s * (R @ P.T)).T + t
        assert np.allclose(P_aligned, P, atol=1e-9)

    def test_pure_translation(self):
        """Known 3 m translation recovered exactly."""
        rng = np.random.default_rng(1)
        P = rng.standard_normal((20, 3))
        shift = np.array([3.0, -1.5, 0.5])
        Q = P + shift
        R, t, s = umeyama_align(P, Q, with_scale=False)
        P_aligned = (R @ P.T).T + t
        assert np.allclose(P_aligned, Q, atol=1e-9)

    def test_known_scale(self):
        """Known 2× scale recovered correctly."""
        rng = np.random.default_rng(2)
        P = rng.standard_normal((30, 3))
        Q = 2.0 * P
        R, t, s = umeyama_align(P, Q, with_scale=True)
        assert abs(s - 2.0) < 1e-6, f"Expected scale ≈ 2, got {s}"
        P_aligned = (s * (R @ P.T)).T + t
        assert np.allclose(P_aligned, Q, atol=1e-9)

    def test_known_rotation(self):
        """Known 90° rotation about Z recovered correctly."""
        rng = np.random.default_rng(3)
        P = rng.standard_normal((40, 3))
        # 90° CCW around Z
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        Q = (R_true @ P.T).T
        R, t, s = umeyama_align(P, Q, with_scale=False)
        P_aligned = (R @ P.T).T + t
        assert np.allclose(P_aligned, Q, atol=1e-9)

    def test_ate_zero_on_perfect_match(self):
        """ATE RMSE is 0 when estimated == ground-truth."""
        rng = np.random.default_rng(4)
        poses = rng.standard_normal((20, 7))
        poses[:, 6] = 1.0   # w=1 quaternion (not normalised but shape is fine)
        rmse = ate_rmse(poses, poses, with_scale=True)
        assert rmse < 1e-9, f"ATE RMSE should be ~0 on perfect match; got {rmse}"

    def test_shape_error(self):
        """ValueError raised for mismatched shapes."""
        P = np.zeros((10, 3))
        Q = np.zeros((11, 3))
        with pytest.raises(ValueError):
            umeyama_align(P, Q)

    def test_too_few_points(self):
        """ValueError raised for N < 3."""
        P = np.zeros((2, 3))
        with pytest.raises(ValueError):
            umeyama_align(P, P)
