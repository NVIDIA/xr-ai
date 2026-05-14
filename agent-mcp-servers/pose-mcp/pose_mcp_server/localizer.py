# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map + pose estimation core.

A :class:`Localizer` wraps a :class:`KeyframeStore` plus the two backends and
exposes ``process(image_rgb, ts_us)``.  Result states:

* ``"calibrating"`` — the geometry backend is still pinning its FOV prior;
                 no keyframes are created yet so the world frame doesn't
                 get anchored to a wrong intrinsic.
* ``"empty"``   — map had zero keyframes; this call inserted the origin.
                 No pose returned (the origin frame *is* the pose, but we
                 surface it explicitly).
* ``"bootstrap"`` — map has keyframes but the current frame could not be
                 PnP-localized against any of them.  Pose is ``None``.
* ``"localized"`` — PnP succeeded; pose is world ← current-camera.

When localization succeeds and the pose has drifted past the configured
thresholds, a new keyframe is appended automatically so the next call has
something to match.
"""
from __future__ import annotations

import dataclasses
import time

import cv2
import numpy as np

from typing import TYPE_CHECKING

from .backends   import FeatureBackend, GeometryBackend
from .geometry   import (compose_se3, invert_se3, make_se3,
                         rmat_to_quat, se3_rotation_deg, se3_translation)
from .store      import Keyframe, KeyframeStore

if TYPE_CHECKING:
    from .viz    import VizSink


@dataclasses.dataclass(frozen=True)
class PoseResult:
    state:          str                   # "empty" | "bootstrap" | "localized"
    pose:           np.ndarray | None     # 4x4 SE(3) world ← camera, or None
    translation:    list[float] | None    # [x, y, z] metres
    quaternion:     list[float] | None    # [w, x, y, z]
    fov_deg:        float
    num_inliers:    int
    num_keyframes:  int
    matched_kf_id:  int | None
    ts_us:          int


class Localizer:
    def __init__(
        self, *,
        store:              KeyframeStore,
        geometry:           GeometryBackend,
        features:           FeatureBackend,
        max_keyframes:      int   = 200,
        min_translation_m:  float = 0.30,
        min_rotation_deg:   float = 20.0,
        min_inliers:        int   = 15,
        min_matches:        int   = 8,
        pnp_reproj_err_px:  float = 4.0,
        viz:                "VizSink | None" = None,
    ) -> None:
        self._store              = store
        self._geometry           = geometry
        self._features           = features
        self._max_keyframes      = max_keyframes
        self._min_translation_m  = min_translation_m
        self._min_rotation_deg   = min_rotation_deg
        # Two thresholds, not one: `min_matches` is the floor on raw 2D-2D
        # correspondences XFeat+LighterGlue returns; `min_inliers` is the
        # floor on PnP-RANSAC inliers.  Conflating them (which the first
        # cut did at 30) makes the localizer reject many legitimate matches
        # before PnP gets a chance to filter them.
        self._min_matches        = min_matches
        self._min_inliers        = min_inliers
        self._pnp_reproj_err_px  = pnp_reproj_err_px
        self._viz                = viz

    def process(self, image_rgb: np.ndarray, ts_us: int | None = None) -> PoseResult:
        if ts_us is None:
            ts_us = int(time.time() * 1_000_000)
        geom  = self._geometry(image_rgb)

        # If the geometry backend exposes a calibration flag and isn't done
        # yet, defer everything — anchoring the world frame to a keyframe
        # built with a wrong intrinsic poisons the rest of the session.
        if getattr(self._geometry, "is_calibrated", True) is False:
            result = PoseResult(
                state="calibrating", pose=None,
                translation=None, quaternion=None,
                fov_deg=geom.fov_deg, num_inliers=0,
                num_keyframes=len(self._store), matched_kf_id=None,
                ts_us=ts_us,
            )
            self._emit_viz(image_rgb, geom, result, None)
            return result

        feats = self._features.extract(image_rgb)
        from loguru import logger
        logger.debug("current frame  features={}", feats.kp.shape[0])

        if len(self._store) == 0:
            origin = np.eye(4, dtype=np.float64)
            kf = self._store.append(
                ts_us=ts_us, pose=origin, fov_deg=geom.fov_deg,
                kp=feats.kp, desc=feats.desc,
                pts3d=geom.points3d, mask=geom.mask,
                image_rgb=image_rgb,
            )
            result = PoseResult(
                state="empty", pose=origin,
                translation=[0.0, 0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0],
                fov_deg=geom.fov_deg, num_inliers=0,
                num_keyframes=len(self._store), matched_kf_id=kf.id,
                ts_us=ts_us,
            )
            self._emit_viz(image_rgb, geom, result, kf)
            return result

        best = self._best_pnp_against_keyframes(feats, geom)

        if best is None:
            result = PoseResult(
                state="bootstrap", pose=None,
                translation=None, quaternion=None,
                fov_deg=geom.fov_deg, num_inliers=0,
                num_keyframes=len(self._store), matched_kf_id=None,
                ts_us=ts_us,
            )
            self._emit_viz(image_rgb, geom, result, None)
            return result

        kf_id, T_world_cam, inliers = best
        kf = next(k for k in self._store.all() if k.id == kf_id)

        new_kf: Keyframe | None = None
        if self._should_insert_keyframe(T_world_cam):
            new_kf = self._store.append(
                ts_us=ts_us, pose=T_world_cam, fov_deg=geom.fov_deg,
                kp=feats.kp, desc=feats.desc,
                pts3d=geom.points3d, mask=geom.mask,
                image_rgb=image_rgb,
            )
            while len(self._store) > self._max_keyframes:
                self._store.evict_oldest()

        t = se3_translation(T_world_cam)
        q = rmat_to_quat(T_world_cam[:3, :3])
        result = PoseResult(
            state="localized", pose=T_world_cam,
            translation=[float(t[0]), float(t[1]), float(t[2])],
            quaternion=[float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            fov_deg=geom.fov_deg, num_inliers=inliers,
            num_keyframes=len(self._store), matched_kf_id=kf.id,
            ts_us=ts_us,
        )
        self._emit_viz(image_rgb, geom, result, new_kf)
        return result

    def _emit_viz(
        self,
        image_rgb:    np.ndarray,
        geom:         "GeometryFrame",          # noqa: F821
        result:       PoseResult,
        new_keyframe: Keyframe | None,
    ) -> None:
        if self._viz is None:
            return
        try:
            self._viz.on_frame(image_rgb, geom, result, new_keyframe)
        except Exception:
            # The viewer is debugging UI — never let it break localization.
            # Route through loguru (not stdlib) so the user actually sees it
            # alongside the rest of the [pose] log stream.
            from loguru import logger
            logger.opt(exception=True).error("viz sink raised; suppressed")

    # ── matching + PnP ──────────────────────────────────────────────────────

    def _best_pnp_against_keyframes(
        self,
        feats: "FrameFeatures",   # noqa: F821
        geom:  "GeometryFrame",   # noqa: F821
    ) -> tuple[int, np.ndarray, int] | None:
        from loguru import logger
        best: tuple[int, np.ndarray, int] | None = None
        K = self._intrinsics_from_fov(geom.fov_deg, geom.width, geom.height)
        attempts: list[str] = []
        for kf in self._store.all():
            try:
                outcome = self._pnp_against_keyframe(kf, feats, K)
            except cv2.error as exc:
                attempts.append(f"kf{kf.id}=cv2.error({exc})")
                continue
            if isinstance(outcome, str):
                attempts.append(f"kf{kf.id}={outcome}")
                continue
            T_world_cam, inliers = outcome
            attempts.append(f"kf{kf.id}=inliers={inliers}")
            if inliers < self._min_inliers:
                continue
            if best is None or inliers > best[2]:
                best = (kf.id, T_world_cam, inliers)
        logger.info(
            "PnP attempts  feats={}  [{}]",
            feats.kp.shape[0],
            "  ".join(attempts) or "<no keyframes>",
        )
        return best

    def _pnp_against_keyframe(
        self,
        kf:    Keyframe,
        feats: "FrameFeatures",       # noqa: F821
        K:     np.ndarray,
    ) -> tuple[np.ndarray, int] | str:
        """Returns ``(T_world_cam, inliers)`` on success, or a short string
        describing why this keyframe was rejected — lets the caller emit a
        single diagnostic line covering all attempts."""
        from .backends import FrameFeatures
        H, W = kf.pts3d.shape[:2]
        kf_feats = FrameFeatures(
            kp=kf.kp, desc=kf.desc, image_size=(int(W), int(H)),
        )
        matches = self._features.match(kf_feats, feats)
        if matches.shape[0] < self._min_matches:
            return f"matches={matches.shape[0]}<min"

        kf_xy   = kf.kp[matches[:, 0]]
        cur_xy  = feats.kp[matches[:, 1]]
        ix = np.clip(np.round(kf_xy[:, 0]).astype(np.int32), 0, W - 1)
        iy = np.clip(np.round(kf_xy[:, 1]).astype(np.int32), 0, H - 1)
        valid = kf.mask[iy, ix]
        n_valid = int(valid.sum())
        if n_valid < self._min_matches:
            return f"matches={matches.shape[0]} valid={n_valid}<min"

        pts3d_kf = kf.pts3d[iy[valid], ix[valid]].astype(np.float32)
        pts2d    = cur_xy[valid].astype(np.float32)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts3d_kf.reshape(-1, 1, 3),
            imagePoints =pts2d.reshape(-1, 1, 2),
            cameraMatrix=K, distCoeffs=None,
            iterationsCount=200, reprojectionError=self._pnp_reproj_err_px,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
        n_pnp = 0 if inliers is None else int(len(inliers))
        if not ok or n_pnp < self._min_inliers:
            return f"matches={matches.shape[0]} valid={n_valid} pnp={n_pnp}<min"

        R, _ = cv2.Rodrigues(rvec)
        T_cam_kf      = make_se3(R, tvec.ravel())
        T_kf_cam      = invert_se3(T_cam_kf)
        T_world_cam   = compose_se3(kf.pose, T_kf_cam)
        return T_world_cam, n_pnp

    @staticmethod
    def _intrinsics_from_fov(fov_deg: float, width: int, height: int) -> np.ndarray:
        # Assume square pixels and principal point at image centre — same
        # assumption MoGe makes when reporting normalized intrinsics.
        fx = 0.5 * width  / np.tan(0.5 * np.radians(fov_deg))
        fy = fx
        return np.array([
            [fx, 0.0, 0.5 * width],
            [0.0, fy, 0.5 * height],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _should_insert_keyframe(self, T_world_cam: np.ndarray) -> bool:
        nearest = min(
            self._store.all(),
            key=lambda kf: float(np.linalg.norm(kf.pose[:3, 3] - T_world_cam[:3, 3])),
        )
        d_t = float(np.linalg.norm(nearest.pose[:3, 3] - T_world_cam[:3, 3]))
        d_r = se3_rotation_deg(nearest.pose, T_world_cam)
        return d_t > self._min_translation_m or d_r > self._min_rotation_deg
