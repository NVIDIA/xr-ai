# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rerun visualisation sink for droid-mcp.

Logs the live trajectory + current camera frustum (and, if the
backend exposes them, the dense per-keyframe depth maps as
back-projected point clouds) to a Rerun viewer.  All Rerun calls are
guarded so a misconfigured / unreachable viewer never breaks the
pose-estimation path.

The viewer runs out of process — start it from the same uv venv:

    uv run rerun

(no flags — the viewer's default mode is a gRPC server on
``localhost:9876`` which is what the SDK pushes to via
``rr.connect_grpc``).  Then point ``rerun_addr`` in the YAML at the
same host:port.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Optional

import numpy as np
from loguru import logger


@dataclasses.dataclass
class _Latest:
    ts_ns:        int
    translation:  tuple[float, float, float]
    quaternion:   tuple[float, float, float, float]   # (w, x, y, z)


class RerunSink:
    """Sink for streaming DROID-SLAM output to a Rerun viewer.

    Thread-safe.  Failures (Rerun not installed, viewer unreachable,
    schema mismatch) are demoted to a single warning log and the sink
    then becomes a no-op for the rest of the session."""

    def __init__(
        self, *,
        addr:         Optional[str]   = None,
        application:  str             = "droid-mcp",
        image_size:   tuple[int, int] = (320, 240),
        intrinsics:   Optional[np.ndarray] = None,
    ) -> None:
        self._image_size = image_size
        self._intrinsics = intrinsics
        self._lock       = threading.Lock()
        self._trail:     list[tuple[float, float, float]] = []
        self._enabled    = False
        self._app        = application

        if addr is None:
            logger.info("[droid-viz] rerun disabled (no rerun_addr)")
            return
        try:
            import rerun as rr
        except ImportError:
            logger.warning(
                "[droid-viz] rerun-sdk not installed — viz disabled. "
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
            logger.warning("[droid-viz] connect to {} failed: {}", addr, exc)
            return
        self._rr = rr
        self._enabled = True
        logger.info("[droid-viz] connected to {}", addr)

    def set_intrinsics(self, K: np.ndarray) -> None:
        with self._lock:
            self._intrinsics = K.copy()
        if not self._enabled:
            return
        # Log pinhole at the camera entity so subsequent frustum draws
        # pick it up.  Will be overwritten on every set_intrinsics.
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
            logger.warning("[droid-viz] log Pinhole failed: {}", exc)
            self._enabled = False

    def log_pose(self, latest: _Latest, *, gray: Optional[np.ndarray] = None) -> None:
        if not self._enabled:
            return
        try:
            self._rr.set_time("kimera_ts", timestamp=latest.ts_ns / 1e9)
            tx, ty, tz = latest.translation
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
            logger.warning("[droid-viz] log_pose failed: {}", exc)
            self._enabled = False

    def log_pose_dict(self, pose: dict, *, gray: Optional[np.ndarray] = None) -> None:
        """Convenience wrapper for the MCP-tool return dict shape."""
        t = pose.get("translation_m")
        q = pose.get("quaternion")
        if t is None or q is None:
            return
        self.log_pose(
            _Latest(
                ts_ns       =int(pose.get("ts_ns") or 0),
                translation =(float(t[0]), float(t[1]), float(t[2])),
                quaternion  =(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            ),
            gray=gray,
        )

    def reset(self) -> None:
        with self._lock:
            self._trail.clear()
        if not self._enabled:
            return
        try:
            self._rr.log("world/trail", self._rr.Clear(recursive=False))
        except Exception as exc:
            logger.debug("[droid-viz] reset trail clear failed: {}", exc)
