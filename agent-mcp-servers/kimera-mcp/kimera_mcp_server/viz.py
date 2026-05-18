# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rerun visualisation sink for kimera-mcp.

Each pose returned by the live Kimera pipeline gets logged here:
* live camera frustum at ``world/camera`` (driven by set_camera_intrinsics)
* current grayscale frame at ``world/camera/image``
* trajectory polyline at ``world/trail``

Failures (rerun-sdk missing, viewer unreachable, schema mismatch) get
demoted to a single warning and the sink turns into a no-op for the
rest of the session — viz issues never tank the pose path.

Start a viewer from the same venv before launching the MCP server:

    uv run rerun --connect rerun+http://localhost:9876/proxy
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Optional

import numpy as np
from loguru import logger


@dataclasses.dataclass
class _Latest:
    ts_ns:       int
    translation: tuple[float, float, float]
    quaternion:  tuple[float, float, float, float]   # (w, x, y, z)


class RerunSink:
    """Thread-safe sink for streaming Kimera-VIO pose updates to a Rerun
    viewer.  Construct with ``addr=None`` to disable entirely."""

    def __init__(
        self, *,
        addr:        Optional[str]   = None,
        application: str             = "kimera-mcp",
        image_size:  tuple[int, int] = (640, 480),
    ) -> None:
        self._image_size = image_size
        self._lock       = threading.Lock()
        self._trail:     list[tuple[float, float, float]] = []
        self._enabled    = False
        self._rr         = None
        if addr is None:
            logger.info("[kimera-viz] rerun disabled (no rerun_addr)")
            return
        try:
            import rerun as rr
        except ImportError:
            logger.warning(
                "[kimera-viz] rerun-sdk not installed — viz disabled. "
                "`uv pip install rerun-sdk` in this venv to enable.",
            )
            return
        # Accept bare host:port too; rerun expects a full URL.
        if "://" not in addr:
            addr = f"rerun+http://{addr}/proxy"
        try:
            rr.init(application, spawn=False)
            rr.connect_grpc(addr)
        except Exception as exc:
            logger.warning("[kimera-viz] connect to {} failed: {}", addr, exc)
            return
        self._rr = rr
        self._enabled = True
        logger.info("[kimera-viz] connected to {}", addr)

    def set_intrinsics(self, *, width: int, height: int,
                       fx: float, fy: float, cx: float, cy: float) -> None:
        self._image_size = (int(width), int(height))
        if not self._enabled:
            return
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        try:
            self._rr.log(
                "world/camera",
                self._rr.Pinhole(
                    image_from_camera=K,
                    width =self._image_size[0],
                    height=self._image_size[1],
                ),
                static=True,
            )
        except Exception as exc:
            logger.warning("[kimera-viz] log Pinhole failed: {}", exc)
            self._enabled = False

    def log_pose_dict(self, pose: dict, *,
                      gray: Optional[np.ndarray] = None) -> None:
        """Log a pose returned by `estimate_pose` (keys: state,
        translation_m, quaternion, ts_ns).  No-op when state != 'ok'."""
        if not self._enabled:
            return
        if pose.get("state") != "ok":
            return
        t = pose.get("translation_m")
        q = pose.get("quaternion")
        if t is None or q is None:
            return
        latest = _Latest(
            ts_ns       =int(pose.get("ts_ns") or 0),
            translation =(float(t[0]), float(t[1]), float(t[2])),
            quaternion  =(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        )
        try:
            self._rr.set_time("kimera_ts", timestamp=latest.ts_ns / 1e9)
            tx, ty, tz   = latest.translation
            qw, qx, qy, qz = latest.quaternion
            self._rr.log(
                "world/camera",
                self._rr.Transform3D(
                    translation=[tx, ty, tz],
                    rotation=self._rr.Quaternion(xyzw=[qx, qy, qz, qw]),
                ),
            )
            with self._lock:
                self._trail.append((tx, ty, tz))
                if len(self._trail) > 4096:
                    self._trail = self._trail[-4096:]
                trail = list(self._trail)
            self._rr.log("world/trail", self._rr.LineStrips3D([trail]))
            if gray is not None:
                self._rr.log("world/camera/image", self._rr.Image(gray))
        except Exception as exc:
            logger.warning("[kimera-viz] log_pose_dict failed: {}", exc)
            self._enabled = False

    def reset(self) -> None:
        with self._lock:
            self._trail.clear()
        if not self._enabled:
            return
        try:
            self._rr.log("world/trail", self._rr.Clear(recursive=False))
        except Exception:
            pass
