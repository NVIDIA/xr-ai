# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MonoSlamAgent — per-frame DPVO visual odometry pose logger.

Subscribes to FrameSignals from the hub, feeds frames to a DPVO SLAM
instance, and logs the live world-frame camera pose per accepted frame.
Pose is also published on the viz side channel once DPVO is initialised.

Design notes
------------
- DPVO (MIT-licensed) runs local bundle adjustment every frame once
  initialised (frame ≥ 8); the live pose is read directly from the patch
  graph after each push and has modest drift.
- Translation has DPVO's internal scale (not metric, no ground-truth
  reference).  The log line notes this.
- Camera intrinsics are approximated from frame dimensions; provide a
  calibrated focal_length_px in the YAML for better accuracy.
- Per-(pid, track) state resets when the track changes size or when the
  participant leaves.
- GPU-bound DPVO runs in a thread-pool executor so the asyncio loop stays
  responsive.
- DPVO requires CUDA.  If DPVO is not installed, import succeeds but
  construction of DPVOSlam raises ImportError; the agent logs an error
  and exits cleanly.
"""
from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import msgpack
import numpy as np
from loguru import logger
from xr_ai_agent import DataMessage, FrameData, FrameSignal, MsgType, ParticipantEvent, ProcessorEndpoint
from xr_ai_agent import encode as ipc_encode

from pixels import frame_to_rgb
from pose import build_camera_matrix, rotation_to_euler_deg
from slam import DPVOSlam, intrinsics_from_K

# Synthetic participant / topic that the viz process subscribes to.
_VIZ_PARTICIPANT = "_mono_slam_viz"
_VIZ_TOPIC       = "mono_slam.pose"


def encode_pose(t_world: np.ndarray, R_world: np.ndarray) -> bytes:
    """Encode camera pose as a msgpack list for the viz side channel.

    Args:
        t_world: Camera position in world frame, shape (3,).
        R_world: R_cam_from_world, shape (3, 3).

    Returns:
        msgpack bytes: [[tx, ty, tz], [r00..r22 flat row-major]]
    """
    return msgpack.packb(
        [t_world.tolist(), R_world.flatten().tolist()],
        use_bin_type=True,
    )


@dataclass
class _TrackState:
    """DPVO instance and bookkeeping for a single (participant, track) pair.

    Convention: DPVO camera frame — X right, Y down, Z forward (OpenCV).
    """
    slam:        Optional[DPVOSlam] = None
    frame_count: int = 0       # frames pushed to DPVO
    signal_count: int = 0      # FrameSignals seen (before stride filter)
    width: int = 0
    height: int = 0


class MonoSlamAgent:

    def __init__(
        self,
        ep: ProcessorEndpoint,
        *,
        weights_path: str,
        fov_deg: float = 60.0,
        focal_length_px: float | None = None,
        frame_stride: int = 3,
        publish_viz: bool = True,
        dpvo_cfg_overrides: list[str] | None = None,
    ) -> None:
        self._ep                  = ep
        self._weights_path        = weights_path
        self._fov_deg             = fov_deg
        self._focal_length_px     = focal_length_px
        self._frame_stride        = max(1, frame_stride)
        self._publish_viz         = publish_viz
        self._dpvo_cfg_overrides  = dpvo_cfg_overrides or []

        # Keyed by (participant_id, track_id).
        self._tracks: dict[tuple[str, str], _TrackState] = {}

        self._ep.on_frame(self._on_frame)
        self._ep.on_participant(self._on_participant)

    # ── frame handling ─────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        key = (sig.participant_id, sig.track_id)
        state = self._tracks.setdefault(key, _TrackState())

        state.signal_count += 1
        if state.signal_count % self._frame_stride != 1:
            return

        frame: FrameData | None = await self._ep.request_frame(sig)
        if frame is None:
            return

        loop = asyncio.get_running_loop()
        try:
            rgb = await loop.run_in_executor(None, frame_to_rgb, frame)
        except ValueError as exc:
            logger.warning("pixel conversion failed  track={}  {}", sig.track_id, exc)
            return

        # Lazily construct the DPVO instance on the first real frame, and
        # lock the (W, H) DPVO was built with — its internal patch graph and
        # network buffers are sized at init time and cannot accept a different
        # resolution mid-track.  WebRTC simulcast / ABR can flip the source
        # resolution between frames; resize incoming frames to match the
        # locked size instead of reinitialising DPVO.
        if state.slam is None:
            state.width  = sig.width
            state.height = sig.height
            K = build_camera_matrix(
                sig.width, sig.height,
                fov_deg=self._fov_deg,
                focal_length_px=self._focal_length_px,
            )
            intr = intrinsics_from_K(K)
            try:
                state.slam = await loop.run_in_executor(
                    None,
                    functools.partial(
                        DPVOSlam,
                        self._weights_path,
                        sig.height,
                        sig.width,
                        intr,
                        self._dpvo_cfg_overrides or None,
                    ),
                )
            except Exception as exc:
                logger.error(
                    "DPVO init failed  track={}  {}  "
                    "(is dpvo installed and CUDA available?)",
                    sig.track_id, exc,
                )
                # Mark track as permanently broken so we don't retry every frame.
                del self._tracks[key]
                return
            logger.info(
                "slam  pid={!r}  track={}  status=dpvo_init  size={}x{}",
                sig.participant_id, sig.track_id, sig.width, sig.height,
            )

        # If WebRTC ABR resized the source mid-stream, scale the frame back
        # to DPVO's locked init resolution.  rgb is HxWxC (numpy / uint8);
        # cv2.resize takes (W, H) in pixel coords.
        if rgb.shape[1] != state.width or rgb.shape[0] != state.height:
            rgb = cv2.resize(rgb, (state.width, state.height),
                             interpolation=cv2.INTER_AREA)

        tstamp = time.time()
        await loop.run_in_executor(
            None,
            functools.partial(state.slam.push, tstamp, rgb),
        )
        state.frame_count += 1

        pose = state.slam.current_pose()
        if pose is None:
            # DPVO not yet initialised (needs ≥8 frames).
            logger.debug(
                "slam  pid={!r}  track={}  frame={}  status=initialising",
                sig.participant_id, sig.track_id, state.frame_count,
            )
            return

        roll, pitch, yaw = rotation_to_euler_deg(pose.R_world)
        pos = pose.pos_world

        logger.info(
            "slam pose"
            "  pid={!r}  track={}"
            "  frame={}"
            "  dpvo_frame={}"
            "  roll_deg={:.2f}  pitch_deg={:.2f}  yaw_deg={:.2f}"
            "  tx={:.4f}  ty={:.4f}  tz={:.4f}"
            "  [t in DPVO internal scale]",
            sig.participant_id, sig.track_id,
            state.frame_count,
            pose.frame_idx,
            roll, pitch, yaw,
            pos[0], pos[1], pos[2],
        )

        if self._publish_viz:
            self._publish_pose(pos, pose.R_world)

    # ── viz side channel ───────────────────────────────────────────────────────

    def _publish_pose(self, pos: np.ndarray, R_world: np.ndarray) -> None:
        """Schedule a pose update to the viz process via the hub data channel.

        Fire-and-forget: creates an asyncio task and swallows all errors so
        viz telemetry never stalls or crashes the worker.
        """
        payload = encode_pose(pos, R_world)
        msg = DataMessage(
            participant_id=_VIZ_PARTICIPANT,
            topic=_VIZ_TOPIC,
            pts_us=int(time.time() * 1_000_000),
            data=payload,
        )
        raw = ipc_encode(MsgType.DATA_MESSAGE, msg)

        async def _send() -> None:
            try:
                await self._ep._push.send(raw)
            except Exception:
                pass  # drop silently — viz is decorative, worker must not block

        t = asyncio.create_task(_send())
        t.add_done_callback(lambda _: None)  # suppress "exception never retrieved"

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
