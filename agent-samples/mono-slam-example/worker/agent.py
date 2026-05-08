# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MonoSlamAgent — per-frame visual odometry pose logger.

Subscribes to FrameSignals from the hub, samples frames at the configured
stride, and runs ORB-based visual odometry to accumulate a relative camera
pose.  Pose is logged per frame as a structured single-line message.

Design notes
------------
- No loop closure, no bundle adjustment, no mapping — tracking only.
- Translation is unit-norm (monocular scale ambiguity); accumulated position
  reflects direction of travel, not metric distance.
- Camera intrinsics are approximated from frame dimensions; provide a
  calibrated focal_length_px in the YAML for better accuracy.
- Per-(pid, track) state is reset when the track changes size or when the
  participant leaves.
- CPU-bound ORB + recoverPose runs in a thread-pool executor to avoid
  blocking the asyncio event loop.
"""
from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from xr_ai_agent import FrameData, FrameSignal, ParticipantEvent, ProcessorEndpoint

from pixels import frame_to_gray
from pose import PoseResult, build_camera_matrix, compute_pose, rotation_to_euler_deg


@dataclass
class _TrackState:
    """Accumulated VO state for a single (participant, track) pair.

    Convention (OpenCV camera frame: X right, Y down, Z forward):

    R_world  — rotation matrix R_curr_from_world.
               Transforms a world-frame point to the current camera frame:
               p_cam = R_world @ p_world + t_world_cam.

    t_world_cam — translation part of T_curr_from_world, i.e. the origin
               of the world (first-frame camera) expressed in the current
               camera frame.

    Camera position in world coordinates:
               pos_world = -R_world.T @ t_world_cam
               (logged as tx/ty/tz — direction only, no metric scale).
    """
    prev_gray: np.ndarray | None = None
    R_world:     np.ndarray = field(default_factory=lambda: np.eye(3))
    t_world_cam: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame_count: int = 0       # frames processed (after stride filter)
    signal_count: int = 0      # FrameSignals seen (before stride filter)
    width: int = 0
    height: int = 0


class MonoSlamAgent:

    def __init__(
        self,
        ep: ProcessorEndpoint,
        *,
        fov_deg: float = 60.0,
        focal_length_px: float | None = None,
        frame_stride: int = 3,
        max_features: int = 500,
        match_ratio: float = 0.75,
        ransac_prob: float = 0.999,
        ransac_threshold: float = 1.0,
        min_inliers: int = 20,
    ) -> None:
        self._ep              = ep
        self._fov_deg         = fov_deg
        self._focal_length_px = focal_length_px
        self._frame_stride    = max(1, frame_stride)
        self._vo_kwargs: dict = dict(
            max_features=max_features,
            match_ratio=match_ratio,
            ransac_prob=ransac_prob,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        )

        # Keyed by (participant_id, track_id).
        self._tracks: dict[tuple[str, str], _TrackState] = {}

        self._ep.on_frame(self._on_frame)
        self._ep.on_participant(self._on_participant)

    # ── frame handling ─────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        key = (sig.participant_id, sig.track_id)
        state = self._tracks.setdefault(key, _TrackState())

        # Reset state when the frame dimensions change (new camera / resolution).
        if state.width != sig.width or state.height != sig.height:
            self._tracks[key] = _TrackState(width=sig.width, height=sig.height)
            state = self._tracks[key]

        state.signal_count += 1
        if state.signal_count % self._frame_stride != 1:
            return

        frame: FrameData | None = await self._ep.request_frame(sig)
        if frame is None:
            return

        loop = asyncio.get_running_loop()
        try:
            gray = await loop.run_in_executor(None, frame_to_gray, frame)
        except ValueError as exc:
            logger.warning("pixel conversion failed  track={}  {}", sig.track_id, exc)
            return

        K = build_camera_matrix(
            sig.width, sig.height,
            fov_deg=self._fov_deg,
            focal_length_px=self._focal_length_px,
        )

        if state.prev_gray is None:
            state.prev_gray = gray
            state.frame_count += 1
            logger.info(
                "slam  pid={!r}  track={}  frame={}  status=first_frame"
                "  size={}x{}",
                sig.participant_id, sig.track_id, state.frame_count,
                sig.width, sig.height,
            )
            return

        result: PoseResult = await loop.run_in_executor(
            None,
            functools.partial(compute_pose, state.prev_gray, gray, K, **self._vo_kwargs),
        )

        state.prev_gray = gray
        state.frame_count += 1

        if not result.ok:
            logger.info(
                "slam  pid={!r}  track={}  frame={}  status=skipped"
                "  inliers={}",
                sig.participant_id, sig.track_id, state.frame_count,
                result.num_inliers,
            )
            return

        # Accumulate T_curr_from_world from successive relative poses.
        #
        # recoverPose convention: p_curr_cam = R @ p_prev_cam + t
        # So T_curr_from_prev = (R, t).
        #
        # T_curr_from_world = T_curr_from_prev @ T_prev_from_world
        #   R_new = R_step @ R_old
        #   t_new = R_step @ t_old + t_step
        state.t_world_cam = result.R @ state.t_world_cam + result.t
        state.R_world     = result.R @ state.R_world

        # Camera position in world (first-frame) coordinates.
        # pos_world = -R_world.T @ t_world_cam  (no metric scale — unit-norm chain).
        pos = -state.R_world.T @ state.t_world_cam

        roll, pitch, yaw = rotation_to_euler_deg(state.R_world)

        # Single structured log line — grep on "slam pose" to extract all poses.
        logger.info(
            "slam pose"
            "  pid={!r}  track={}"
            "  frame={}"
            "  inliers={}"
            "  roll_deg={:.2f}  pitch_deg={:.2f}  yaw_deg={:.2f}"
            "  tx={:.4f}  ty={:.4f}  tz={:.4f}"
            "  [t unit-norm monocular scale]",
            sig.participant_id, sig.track_id,
            state.frame_count,
            result.num_inliers,
            roll, pitch, yaw,
            pos[0], pos[1], pos[2],
        )

    # ── participant lifecycle ──────────────────────────────────────────────────

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if event.joined:
            return
        pid = event.participant_id
        for key in [k for k in self._tracks if k[0] == pid]:
            del self._tracks[key]

    # ── agent lifecycle ────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()
