# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Rerun viewer sink.

Off unless ``rerun_addr`` is set in the config.  When on, every
``estimate_pose`` call publishes:

* ``/world/origin``                   — coordinate axes at the map origin
* ``/world/keyframes/kf_NNNNNN/...``  — each persistent keyframe's pose +
                                        subsampled metric point cloud
* ``/world/trajectory``               — polyline through the keyframe poses
* ``/world/camera``                   — live camera frustum at the recovered
                                        pose; the current frame's RGB pixels
                                        are logged underneath as ``image``

The :class:`RerunSink` lazy-imports ``rerun`` so projects that don't want the
viewer never pay the dependency cost.  The :class:`VizSink` Protocol lets the
:class:`Localizer` accept any object with the same shape (used by tests to
record calls without dragging in Rerun).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .backends   import GeometryFrame
from .geometry   import rmat_to_quat
from .localizer  import PoseResult
from .store      import Keyframe


class VizSink(Protocol):
    def on_load(self, keyframes: list[Keyframe]) -> None: ...
    def on_frame(
        self,
        image_rgb: np.ndarray,
        geom:      GeometryFrame,
        result:    PoseResult,
        new_keyframe: Keyframe | None,
    ) -> None: ...


class RerunSink:
    """Streams pose + geometry into a Rerun viewer.

    Initialised on first use so the heavy ``rerun`` import never runs when
    the sink is disabled.  Subsequent calls reuse the connection.
    """

    def __init__(
        self,
        *,
        application_id: str = "pose-mcp",
        addr:           str = "127.0.0.1:9876",
        live_max_pts:   int = 8000,
    ) -> None:
        self._app_id        = application_id
        self._addr          = addr
        self._live_max_pts  = live_max_pts
        self._traj:         list[list[float]] = []
        self._rr            = None  # lazy

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._rr is not None:
            return
        import rerun as rr  # heavy import deferred
        rr.init(self._app_id)
        # Rerun >=0.22's connect_grpc requires a full URI (scheme + /proxy
        # path); accept either form in config so operators can write the
        # short `host:port` they're used to.
        url = self._addr if "://" in self._addr else f"rerun+http://{self._addr}/proxy"
        rr.connect_grpc(url)
        # Long-lived axes anchor; users orient quickly with these visible.
        rr.log(
            "/world/origin",
            rr.Transform3D(translation=[0.0, 0.0, 0.0]),
            static=True,
        )
        self._rr = rr

    def on_load(self, keyframes: list[Keyframe]) -> None:
        """Re-publish a persistent map's keyframes when the server restarts.

        Without this, a viewer connecting to a freshly-started server would
        see only the live camera until the next keyframe is inserted.
        """
        self._ensure_connected()
        self._traj.clear()
        for kf in keyframes:
            self._log_keyframe(kf, static=True)
        self._log_trajectory()

    def on_frame(
        self,
        image_rgb: np.ndarray,
        geom:      GeometryFrame,
        result:    PoseResult,
        new_keyframe: Keyframe | None,
    ) -> None:
        self._ensure_connected()
        rr = self._rr
        # Rerun 0.22+ unified its time API: the old `set_time_seconds(name, t)`
        # is now `set_time(name, timestamp=t)` for absolute wall-clock or
        # `set_time(name, duration=t)` for elapsed time.  ts_us is a Unix
        # microseconds timestamp from the worker so use the `timestamp=` form.
        rr.set_time("frame_time", timestamp=result.ts_us / 1_000_000.0)

        if new_keyframe is not None:
            self._log_keyframe(new_keyframe, static=False)
            self._log_trajectory()

        # `Scalar` (singular) was renamed to `Scalars` (which takes a list)
        # in 0.22+ — same semantics, slightly different ergonomics.
        rr.log("/world/inliers",       rr.Scalars([float(result.num_inliers)]))
        rr.log("/world/num_keyframes", rr.Scalars([float(result.num_keyframes)]))

        if result.pose is None:
            # Bootstrap or lost — clear the live camera entity so stale
            # frusta don't linger in the viewer.
            rr.log("/world/camera", rr.Clear(recursive=True))
            return

        self._log_camera("/world/camera", result.pose, geom, image_rgb)

    # ── helpers ────────────────────────────────────────────────────────────

    def _log_keyframe(self, kf: Keyframe, *, static: bool) -> None:
        path = f"/world/keyframes/kf_{kf.id:06d}"
        self._log_pose(path, kf.pose, static=static)
        # Lift the keyframe's stored point map into world coords and log a
        # subsampled cloud — full HxWx3 is too dense to display per kf.
        pts_kf = kf.pts3d.reshape(-1, 3).astype(np.float32)
        mask   = kf.mask.reshape(-1)
        pts_kf = pts_kf[mask]
        if len(pts_kf) > self._live_max_pts:
            idx    = np.linspace(0, len(pts_kf) - 1, self._live_max_pts).astype(np.int64)
            pts_kf = pts_kf[idx]
        pts_world = (kf.pose[:3, :3] @ pts_kf.T).T + kf.pose[:3, 3]
        self._rr.log(
            f"{path}/points",
            self._rr.Points3D(positions=pts_world.astype(np.float32)),
            static=static,
        )
        # Maintain the trajectory polyline as keyframes accumulate.
        self._traj.append(kf.pose[:3, 3].tolist())

    def _log_trajectory(self) -> None:
        if len(self._traj) < 2:
            return
        self._rr.log(
            "/world/trajectory",
            self._rr.LineStrips3D(np.array(self._traj, dtype=np.float32).reshape(1, -1, 3)),
            static=True,
        )

    def _log_camera(
        self,
        path:      str,
        pose:      np.ndarray,
        geom:      GeometryFrame,
        image_rgb: np.ndarray,
    ) -> None:
        self._log_pose(path, pose, static=False)
        fx = 0.5 * geom.width / np.tan(0.5 * np.deg2rad(geom.fov_deg))
        self._rr.log(
            f"{path}/image",
            self._rr.Pinhole(
                resolution=[geom.width, geom.height],
                focal_length=[float(fx), float(fx)],
                principal_point=[geom.width / 2.0, geom.height / 2.0],
            ),
        )
        self._rr.log(f"{path}/image/rgb", self._rr.Image(image_rgb))

    def _log_pose(self, path: str, pose: np.ndarray, *, static: bool) -> None:
        t = pose[:3, 3].astype(np.float32)
        R = pose[:3, :3].astype(np.float64)
        # Use the rotation matrix directly — round-tripping through a
        # quaternion just for the API call doesn't help anything.
        self._rr.log(
            path,
            self._rr.Transform3D(translation=t, mat3x3=R),
            static=static,
        )
