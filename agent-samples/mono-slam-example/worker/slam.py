# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DPVO adapter — thin synchronous wrapper around the DPVO SLAM system.

DPVO (Deep Patch Visual Odometry, MIT license) tracks camera pose by
maintaining a patch graph that is bundle-adjusted incrementally.  This
module hides DPVO internals from agent.py and exposes a simple push-based
interface:

    slam = DPVOSlam(weights_path, height, width, intrinsics_4)
    slam.push(tstamp, rgb_hwc_uint8)   # call per frame
    result = slam.current_pose()       # query after push; may be None

The returned pose is the *live* (unoptimised) world-frame camera pose read
directly from the patch graph after each frame.  It is updated by local BA
every frame once the system is initialised (frame ≥ 8) and therefore has
modest drift.  Call terminate() at end-of-session for the globally-refined
trajectory (12 rounds of bundle adjustment); this blocks and is intended
for offline evaluation only.

Coordinate frame
----------------
- DPVO camera frame: X right, Y down, Z forward (OpenCV convention).
- ``pg.poses_[i]`` stores the INVERSE world pose (camera-to-world is
  inverted), so we call ``.inv()`` before extracting R, t to get
  T_world_from_cam: pos_world = T_world_from_cam.t, R_world = T_world_from_cam.R.

Threading / GPU
---------------
DPVO runs entirely on the CUDA device specified at construction.  All
public methods are synchronous and must be called from a single thread (or
inside a thread-pool executor — not from asyncio directly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SlamPose:
    """World-frame camera pose from DPVO.

    pos_world:  Camera position in world (first-frame) coordinates, metres
                (up to DPVO's internal scale — not metric without additional
                scale recovery).  Shape (3,), float64.
    R_world:    R_cam_from_world (3×3 float64).  Same convention as the
                ORB-based pose that was here before:
                p_cam = R_world @ p_world + t_cam.
                Camera position = -R_world.T @ t_cam = pos_world.
    frame_idx:  Zero-based DPVO frame index (``slam.counter`` at push time).
    """
    pos_world: np.ndarray   # (3,)
    R_world:   np.ndarray   # (3, 3)
    frame_idx: int


class DPVOSlam:
    """Streaming DPVO adapter.

    Args:
        weights_path:   Path to the ``dpvo.pth`` checkpoint.
        height:         Frame height in pixels.
        width:          Frame width in pixels.
        intrinsics:     4-vector [fx, fy, cx, cy] in pixels.
        cfg_overrides:  Optional list of YACS key=value strings applied
                        after loading the default config (e.g.
                        ``["LOOP_CLOSURE=True"]``).
    """

    def __init__(
        self,
        weights_path: str,
        height: int,
        width: int,
        intrinsics: np.ndarray,
        cfg_overrides: list[str] | None = None,
    ) -> None:
        # Lazy import — DPVO is not installed in the base worker venv; the
        # CI smoke test gates on importlib.util.find_spec("dpvo").
        import torch
        from dpvo.config import cfg as dpvo_cfg
        from dpvo.dpvo import DPVO

        if cfg_overrides:
            flat = []
            for item in cfg_overrides:
                k, v = item.split("=", 1)
                flat += [k.strip(), v.strip()]
            dpvo_cfg.merge_from_list(flat)

        self._torch  = torch
        self._SE3    = _import_SE3()
        self._cfg    = dpvo_cfg
        self._ht     = height
        self._wd     = width
        self._intrinsics_np = np.asarray(intrinsics, dtype=np.float32)   # [fx,fy,cx,cy]

        self._dpvo: DPVO = DPVO(dpvo_cfg, weights_path, ht=height, wd=width, viz=False)
        self._latest_pose: Optional[SlamPose] = None

    # ── public API ──────────────────────────────────────────────────────────────

    def push(self, tstamp: float, image_hwc: np.ndarray) -> None:
        """Feed one RGB frame (H×W×3, uint8) to DPVO.

        Args:
            tstamp:    Frame timestamp in seconds (used by DPVO's motion model).
            image_hwc: RGB image as (height, width, 3) uint8 array.
        """
        torch = self._torch
        image_chw = torch.from_numpy(image_hwc).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(self._intrinsics_np).cuda()

        with torch.no_grad():
            self._dpvo(tstamp, image_chw, intrinsics)

        self._latest_pose = self._read_pose()

    def current_pose(self) -> Optional[SlamPose]:
        """Return the latest live pose, or None before initialisation.

        DPVO sets ``is_initialized=True`` at frame 8 (the first frame at
        which the patch graph has been bundle-adjusted).  Before that, poses
        from the patch graph are identity-initialised garbage and are
        suppressed here.
        """
        return self._latest_pose

    def init_progress(self) -> dict:
        """Snapshot of DPVO's internal bootstrap state for diagnostics."""
        dpvo = self._dpvo
        return {
            "n":              int(getattr(dpvo, "n", 0)),
            "is_initialized": bool(getattr(dpvo, "is_initialized", False)),
            "M":              int(getattr(dpvo, "M", 0)),
        }

    def terminate(self) -> tuple[np.ndarray, np.ndarray]:
        """Run global BA and return the full refined trajectory.

        Returns:
            poses:    (N, 7) float64 array [tx, ty, tz, qx, qy, qz, qw],
                      world-frame camera poses (T_world_from_cam).
            tstamps:  (N,) float64 array of timestamps in seconds.

        Blocks for several seconds (12 BA rounds).  Call once at
        end-of-session for evaluation; do not call from the live worker.
        """
        poses, tstamps = self._dpvo.terminate()
        # terminate() already returns T_world_from_cam (poses.inv() applied).
        return poses.astype(np.float64), tstamps.astype(np.float64)

    # ── internals ───────────────────────────────────────────────────────────────

    def _read_pose(self) -> Optional[SlamPose]:
        """Extract the current live pose from the patch graph.

        Returns None until DPVO is initialised (n > 0 and is_initialized).
        """
        dpvo = self._dpvo
        if not dpvo.is_initialized or dpvo.n == 0:
            return None

        SE3 = self._SE3

        # pg.poses_[i] is T_cam_from_world (camera-in-world is inverted).
        # Invert to get T_world_from_cam: pos_world = T.t, R = T.R.
        pose_tensor = dpvo.pg.poses_[dpvo.n - 1]           # (7,) on CUDA
        T_world_from_cam = SE3(pose_tensor.unsqueeze(0)).inv()
        data = T_world_from_cam.data.squeeze(0).cpu().numpy()  # (7,) [x,y,z,qx,qy,qz,qw]

        pos  = data[:3].astype(np.float64)
        quat = data[3:].astype(np.float64)  # [qx, qy, qz, qw]
        R    = _quat_to_rotation_matrix(quat)

        return SlamPose(pos_world=pos, R_world=R, frame_idx=int(dpvo.counter))


# ── helpers ─────────────────────────────────────────────────────────────────────

def _import_SE3():
    """Import lietorch.SE3 from whichever installed location DPVO uses."""
    try:
        from dpvo.lietorch import SE3
        return SE3
    except ImportError:
        from lietorch import SE3
        return SE3


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [qx, qy, qz, qw] to a 3×3 rotation matrix.

    Uses the standard closed-form conversion.  Input is normalised to guard
    against numerical drift in DPVO's pose vector.

    Args:
        q: Array-like of shape (4,) as [qx, qy, qz, qw].

    Returns:
        (3, 3) float64 rotation matrix R_cam_from_world.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.eye(3)
    q = q / norm

    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R


def intrinsics_from_K(K: np.ndarray) -> np.ndarray:
    """Extract DPVO-format [fx, fy, cx, cy] from a 3×3 camera matrix.

    Args:
        K: 3×3 pinhole camera matrix (float64).

    Returns:
        (4,) float32 array [fx, fy, cx, cy].
    """
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
