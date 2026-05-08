# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CI smoke test for the DPVO SLAM adapter (slam.py).

Skipped when DPVO is not installed (no model weights, no CUDA required for
import check).  When DPVO is installed, feeds synthetic frames to the
adapter and asserts that:

  1. DPVOSlam constructs without exception.
  2. push() accepts 15 frames without exception.
  3. current_pose() returns a valid SlamPose after initialisation (frame ≥ 8).
  4. The returned rotation matrix is a proper rotation (det ≈ 1, finite).
  5. The position vector is finite.

The synthetic frames are checkerboard-noise composites — they carry no real
visual content, so DPVO's pose estimate will be numerically arbitrary.  The
test validates the adapter contract, not DPVO accuracy.

GPU requirement: DPVO requires CUDA.  The test is further guarded by
``torch.cuda.is_available()`` inside the ``dpvo`` skip condition because
DPVO raises at construction if no GPU is present.
"""
from __future__ import annotations

import importlib.util
import numpy as np
import pytest

# Skip the entire module if DPVO is not installed.  Check importlib rather
# than importing directly so a missing dpvo dep doesn't pollute the test
# collection output with an ImportError.
dpvo_available = importlib.util.find_spec("dpvo") is not None
skip_no_dpvo = pytest.mark.skipif(
    not dpvo_available,
    reason="DPVO not installed — run scripts/install_dpvo.sh",
)

# Additional guard for the GPU (needed at construction time, before any test
# body runs).
def _cuda_and_dpvo() -> bool:
    if not dpvo_available:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


skip_no_gpu = pytest.mark.skipif(
    not _cuda_and_dpvo(),
    reason="DPVO requires a CUDA GPU",
)


def _synthetic_frames(n: int, height: int = 240, width: int = 320) -> list[np.ndarray]:
    """Return n uint8 (H, W, 3) RGB frames with a translating checkerboard.

    The checkerboard shifts 4 pixels per frame, producing coherent optical
    flow so DPVO's motion_probe() exceeds its 2.0 threshold and increments n.
    Without translation (static content + noise only), motion_probe() stays
    near zero and is_initialized never flips.
    """
    frames = []
    for i in range(n):
        # Translate checkerboard by 4*i pixels horizontally for real optical flow.
        shift = i * 4
        xs = (np.arange(width) - shift) // 16
        ys = np.arange(height) // 16
        board = ((xs[None, :] + ys[:, None]) % 2).astype(np.uint8) * 200
        rgb = np.stack([board, board, board], axis=-1)
        # Small per-frame noise (seeded per-frame for determinism).
        rng = np.random.default_rng(seed=i)
        noise = rng.integers(0, 15, size=(height, width, 3), dtype=np.uint8)
        frames.append(np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    return frames


@skip_no_dpvo
@skip_no_gpu
class TestDPVOSlamAdapter:
    """Smoke tests for slam.DPVOSlam."""

    HEIGHT = 240
    WIDTH  = 320
    # fx=fy≈277 for a 60° FOV over 320px wide image
    INTRINSICS = np.array([277.0, 277.0, 160.0, 120.0], dtype=np.float32)
    # Synthetic frames need more margin than the raw DPVO threshold of 8:
    # on a translating checkerboard, motion_probe() skips every other frame,
    # so n reaches 8 around frame 14-16 in practice. 25 gives enough headroom.
    N_FRAMES   = 25

    def _make_slam(self, weights_path: str) -> object:
        """Construct a DPVOSlam; return it."""
        from slam import DPVOSlam
        return DPVOSlam(
            weights_path=weights_path,
            height=self.HEIGHT,
            width=self.WIDTH,
            intrinsics=self.INTRINSICS,
        )

    @pytest.fixture(scope="class")
    def weights_path(self, request) -> str:
        """Resolve the DPVO weights file from the CLI option or a default path."""
        path = request.config.getoption("--weights-path", default="models/dpvo.pth")
        return str(path)

    @pytest.fixture(scope="class")
    def slam_and_frames(self, weights_path):
        """Construct slam, push all frames, return (slam, frames)."""
        from slam import DPVOSlam
        slam = DPVOSlam(
            weights_path=weights_path,
            height=self.HEIGHT,
            width=self.WIDTH,
            intrinsics=self.INTRINSICS,
        )
        frames = _synthetic_frames(self.N_FRAMES, self.HEIGHT, self.WIDTH)
        for i, frame in enumerate(frames):
            slam.push(float(i) / 30.0, frame)   # 30 fps synthetic timestamps
        return slam, frames

    def test_construction(self, weights_path):
        """DPVOSlam constructs without raising."""
        from slam import DPVOSlam
        slam = DPVOSlam(
            weights_path=weights_path,
            height=self.HEIGHT,
            width=self.WIDTH,
            intrinsics=self.INTRINSICS,
        )
        assert slam is not None

    def test_push_all_frames(self, slam_and_frames):
        """push() accepts all frames without exception."""
        slam, frames = slam_and_frames
        # If we got here, no exception was raised during push() — pass.
        assert slam._dpvo.counter == len(frames)

    def test_pose_available_after_init(self, slam_and_frames):
        """current_pose() is not None after ≥8 frames."""
        slam, _ = slam_and_frames
        pose = slam.current_pose()
        assert pose is not None, (
            "current_pose() returned None after {n} frames — "
            "DPVO may not have initialised (needs ≥8 frames with sufficient motion)"
        ).format(n=self.N_FRAMES)

    def test_rotation_is_proper(self, slam_and_frames):
        """Returned rotation matrix has det ≈ 1 and finite entries."""
        slam, _ = slam_and_frames
        pose = slam.current_pose()
        if pose is None:
            pytest.skip("DPVO not yet initialised — increase N_FRAMES")
        R = pose.R_world
        assert R.shape == (3, 3), f"R_world has unexpected shape {R.shape}"
        assert np.isfinite(R).all(), "R_world contains NaN or Inf"
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 0.01, f"R_world det = {det:.6f} (expected ≈ 1)"

    def test_position_is_finite(self, slam_and_frames):
        """Returned position vector has finite entries."""
        slam, _ = slam_and_frames
        pose = slam.current_pose()
        if pose is None:
            pytest.skip("DPVO not yet initialised — increase N_FRAMES")
        pos = pose.pos_world
        assert pos.shape == (3,), f"pos_world has unexpected shape {pos.shape}"
        assert np.isfinite(pos).all(), "pos_world contains NaN or Inf"

    def test_frame_idx_increments(self, slam_and_frames):
        """frame_idx in the final pose matches the frame count."""
        slam, frames = slam_and_frames
        pose = slam.current_pose()
        if pose is None:
            pytest.skip("DPVO not yet initialised — increase N_FRAMES")
        # frame_idx is dpvo.counter, which counts all pushed frames.
        assert pose.frame_idx == len(frames)


# ── helpers exposed for benchmark re-use ───────────────────────────────────────

def make_synthetic_frames(n: int, height: int = 240, width: int = 320) -> list[np.ndarray]:
    """Public alias for _synthetic_frames; importable from benchmark tests."""
    return _synthetic_frames(n, height, width)
