# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper around DROID-SLAM (princeton-vl/DROID-SLAM, BSD-3-Clause).

DROID-SLAM is monocular (or stereo / RGB-D) deep SLAM — a recurrent
network iteratively updates per-pixel depth + camera poses through a
differentiable bundle adjustment layer.  Lower trajectory error than
classical SLAM on TUM/EuRoC, but heavy: needs CUDA, ~10 GB VRAM at
real-time keyframe density.

We import ``droid_slam`` lazily so the rest of the MCP server still
imports / starts on a GPU-less developer box (e.g. for `--help` /
config validation), and only fall over when the first actual frame
arrives without DROID installed.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Optional

import numpy as np
from loguru import logger


@dataclasses.dataclass(frozen=True)
class DroidPose:
    have_pose:   bool
    state:       str                 # "ok" | "uninitialized" | "lost"
    ts_ns:       int
    translation: tuple[float, float, float]
    quaternion:  tuple[float, float, float, float]   # (w, x, y, z)


class DroidBackend:
    """Holds the DROID-SLAM model + tracker state for one session.

    Single-threaded by design: DROID's BA solver isn't reentrant.  All
    public methods serialise on an internal mutex so concurrent FastMCP
    handlers don't interleave frame submissions."""

    def __init__(
        self, *,
        weights_path: str,
        image_size:   tuple[int, int] = (320, 240),   # (W, H) after resize
        device:       str             = "cuda:0",
        buffer:       int             = 512,
        keyframe_thresh: float        = 4.0,
        frontend_thresh: float        = 16.0,
    ) -> None:
        self._weights         = weights_path
        self._image_size      = image_size
        self._device          = device
        self._buffer          = int(buffer)
        self._keyframe_thresh = float(keyframe_thresh)
        self._frontend_thresh = float(frontend_thresh)

        self._lock         = threading.Lock()
        self._droid        = None                  # lazily built Droid instance
        self._frame_id     = 0
        self._intrinsics:  Optional[np.ndarray] = None  # (fx, fy, cx, cy)
        self._last_pose:   Optional[DroidPose]  = None
        self._frames_seen  = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _ensure_droid(self) -> None:
        if self._droid is not None:
            return
        try:
            import torch
            from droid_slam.droid import Droid  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "droid_slam not available — run "
                "agent-mcp-servers/droid-mcp/scripts/setup_droid.sh "
                "to install it (needs CUDA + PyTorch).  Underlying "
                f"ImportError: {exc}"
            ) from exc

        # DROID's CLI surface takes an argparse.Namespace; we mimic just
        # the fields it reads.
        class _Args:
            weights         = self._weights
            buffer          = self._buffer
            image_size      = list(self._image_size[::-1])  # DROID wants (H, W)
            disable_vis     = True
            upsample        = False
            beta            = 0.3
            filter_thresh   = 2.4
            warmup          = 8
            keyframe_thresh = self._keyframe_thresh
            frontend_thresh = self._frontend_thresh
            frontend_window = 25
            frontend_radius = 2
            frontend_nms    = 1
            backend_thresh  = 22.0
            backend_radius  = 2
            backend_nms     = 3
            stereo          = False

        logger.info("[droid] constructing Droid(weights={}, image_size={})",
                    self._weights, self._image_size)
        self._droid = Droid(_Args())

    def set_intrinsics(self, *, width: int, height: int,
                       fx: float, fy: float, cx: float, cy: float) -> None:
        """Set the calibration that maps the *resized* (image_size)
        tracking frame to the world.  DROID-SLAM uses pinhole; if you
        send a frame at a different resolution we'll fail loudly."""
        if (width, height) != self._image_size:
            raise ValueError(
                f"intrinsics width/height {width}x{height} must match "
                f"DroidBackend image_size {self._image_size} — resize "
                "frames on the caller side before set_intrinsics."
            )
        with self._lock:
            self._intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
        logger.info(
            "[droid] intrinsics set  {}x{}  fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}",
            width, height, fx, fy, cx, cy,
        )

    def track(self, *, ts_ns: int, gray: np.ndarray) -> DroidPose:
        """Push a frame, return the latest pose estimate (which may
        still be the previous one — DROID updates poses asynchronously
        as the frontend refines them)."""
        if self._intrinsics is None:
            raise RuntimeError(
                "set_intrinsics() must be called before track() — "
                "DROID needs a calibration to bootstrap.",
            )
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("DroidBackend.track expects a 2-D uint8 array")
        with self._lock:
            self._ensure_droid()
            import torch
            # Resize / pad here if the caller didn't already match
            # image_size — we accept exact match only, defensively.
            h, w = gray.shape
            if (w, h) != self._image_size:
                raise ValueError(
                    f"frame {w}x{h} must match DroidBackend image_size "
                    f"{self._image_size} (caller is responsible for resize)"
                )

            # DROID's track() expects float image normalised to [0, 255]
            # with shape (3, H, W).  Repeat grayscale across RGB channels.
            img_t = torch.from_numpy(gray).to(self._device).float()
            img_t = img_t.unsqueeze(0).repeat(3, 1, 1)
            intr_t = torch.from_numpy(self._intrinsics).to(self._device)
            tid = self._frame_id
            self._frame_id += 1
            self._droid.track(tid, img_t, intrinsics=intr_t)
            self._frames_seen += 1

            # Pull the latest pose out of DROID's internal video buffer.
            pose_t = self._droid.video.poses[max(0, tid)].cpu().numpy()
            # DROID stores poses as 7-vec (tx, ty, tz, qx, qy, qz, qw).
            tx, ty, tz, qx, qy, qz, qw = (float(v) for v in pose_t)
            n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5 or 1.0
            qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
            state = "ok" if self._frames_seen >= 8 else "uninitialized"
            self._last_pose = DroidPose(
                have_pose   =(state == "ok"),
                state       =state,
                ts_ns       =int(ts_ns),
                translation =(tx, ty, tz),
                quaternion  =(qw, qx, qy, qz),
            )
            return self._last_pose

    def reset(self) -> None:
        with self._lock:
            self._droid       = None
            self._frame_id    = 0
            self._last_pose   = None
            self._frames_seen = 0
            logger.warning("[droid] backend reset")

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def has_intrinsics(self) -> bool:
        return self._intrinsics is not None
