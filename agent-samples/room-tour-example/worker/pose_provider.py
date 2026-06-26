# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MonoPoseProvider — OPTIONAL monocular pose + relocalization for room-tour.

The room-tour map is text-space (TextSLAM): pose-free by design, because the XR
input is a single camera. This module is the *additive* pose sensor the README
flagged as a future variant — it wraps the ``mono-slam-xr`` backbone
(TartanVO up-to-scale VO + Depth-Anything-V2 metric depth for a frozen global
scale + DINOv2-CLS appearance relocalization) and exposes three agent-facing
affordances: a per-frame 6DoF pose, a geometric bearing-to-target, and an
appearance relocalization. None of this replaces the text map; the brain uses it
*when present* to give true geometric bearings instead of the VLM's coarse
left/center/right, and falls back to pose-free behavior whenever it is absent.

Monocular discipline (single camera → scale is estimated, not free):
  * ``metric_valid`` is False until a global metric scale is recovered. While
    False, bearings are still correct but **distances must not be spoken**.
  * ``pose_cam_to_world is None`` means genuine tracking loss → the agent must
    fall back to the VLM/text path, never guess.
  * Pose is camera-to-world in a per-session room frame anchored at the first
    frame (identity); bearings are therefore *relative* ("to your left"), not
    compass-absolute, and the frame does not persist across sessions.
  * Relocalization abstains by default (high precision, low recall) — fuse it
    with the text relocalizer for high-stakes navigation rather than trusting it
    alone.

This is a SOFT dependency. Importing it only succeeds when the backbone code and
its model assets are deployed (see ``MONO_SLAM_WORKSPACE``); ``agent.py`` wraps
the import in try/except and stays pose-free otherwise. The heavy deps
(torch+CUDA, opencv, the vendored backbone) live behind the ``pose`` optional
extra in ``pyproject.toml`` so the base worker pulls none of them.

Validated on Replica only (bearing ~8.9 deg vs VLM ~30 deg, pose availability
1.0, ~31 ms/frame). NOT validated on real glasses footage or absolute metric
distance — treat distances as coarse and always gate on ``metric_valid``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# The backbone lives outside this repo (it carries torch/cv2 + model weights).
# Point MONO_SLAM_WORKSPACE at the deployed `mono-slam-xr/workspace` dir (the one
# containing `baselines/` + `harness/`); importing this module fails cleanly when
# it is unset or wrong, which the brain treats as "stay pose-free".
_WORKSPACE_ENV = "MONO_SLAM_WORKSPACE"


def _resolve_workspace() -> Path:
    root = os.environ.get(_WORKSPACE_ENV)
    if not root:
        raise RuntimeError(
            f"{_WORKSPACE_ENV} is not set; the mono-slam pose backbone is "
            "optional and absent — the agent stays pose-free."
        )
    ws = Path(root).expanduser().resolve()
    if not (ws / "baselines").is_dir() or not (ws / "harness").is_dir():
        raise RuntimeError(
            f"{_WORKSPACE_ENV}={ws} does not look like a mono-slam-xr workspace "
            "(missing baselines/ or harness/)."
        )
    return ws


_WORKSPACE = _resolve_workspace()
for _p in (_WORKSPACE / "baselines", _WORKSPACE / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Pulls in the validated candidate + dataset types; requires the workspace layout
# (baselines/ + harness/) and the model assets under baselines/models/.
from dataset import CameraParams, ImageFrame  # type: ignore  # noqa: E402


# ── pixel-format conversion — mirrors worker/pixels.py exactly ────────────────
# The backbone's VO/depth/descriptor path was validated on cv2.imread() BGR
# frames, so we reproduce pixels.py's reshape + BT.601 limited-range YCbCr→RGB
# math and return BGR. We duck-type FrameData (width/height/fmt/data) and accept
# fmt as the PixelFormat enum, its .name, or its int value, so the adapter does
# not hard-depend on xr_ai_agent being importable.

_FMT_NAMES = {"I420": 0, "NV12": 1, "RGB24": 2, "RGBA": 3, "BGRA": 4}
_FMT_I420, _FMT_NV12, _FMT_RGB24, _FMT_RGBA, _FMT_BGRA = 0, 1, 2, 3, 4


def _fmt_code(fmt) -> int:
    if isinstance(fmt, int):
        return int(fmt)
    name = getattr(fmt, "name", None) or str(fmt).split(".")[-1]
    if name in _FMT_NAMES:
        return _FMT_NAMES[name]
    val = getattr(fmt, "value", None)
    if isinstance(val, int):
        return val
    raise ValueError(f"Unsupported pixel format: {fmt!r}")


def _yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """BT.601 limited-range YCbCr → RGB uint8 (identical math to pixels.py)."""
    Y = Y.astype(np.float32) - 16.0
    U = U.astype(np.float32) - 128.0
    V = V.astype(np.float32) - 128.0
    R = np.clip(1.164 * Y + 1.596 * V, 0, 255)
    G = np.clip(1.164 * Y - 0.392 * U - 0.813 * V, 0, 255)
    B = np.clip(1.164 * Y + 2.017 * U, 0, 255)
    return np.stack([R, G, B], axis=-1).astype(np.uint8)


def frame_data_to_bgr(frame_data) -> np.ndarray:
    """Convert a room-tour FrameData → BGR uint8 HxWx3 (cv2 channel order).

    Same reshape + BT.601 math as ``worker/pixels.py:frame_to_pil``; only the
    final channel order differs (RGB→BGR) so the backbone sees the pixels it was
    validated on. RGB24/RGBA/BGRA are lossless; I420/NV12 are inherently 4:2:0.
    """
    w, h = int(frame_data.width), int(frame_data.height)
    arr = np.frombuffer(frame_data.data, dtype=np.uint8)
    code = _fmt_code(frame_data.fmt)

    if code == _FMT_RGB24:
        rgb = arr.reshape(h, w, 3)
    elif code == _FMT_RGBA:
        rgb = arr.reshape(h, w, 4)[:, :, :3]
    elif code == _FMT_BGRA:
        return np.ascontiguousarray(arr.reshape(h, w, 4)[:, :, :3])
    elif code == _FMT_I420:
        y_end = w * h
        uv_sz = (w // 2) * (h // 2)
        Y = arr[:y_end].reshape(h, w)
        U = arr[y_end:y_end + uv_sz].reshape(h // 2, w // 2).repeat(2, 0).repeat(2, 1)
        V = arr[y_end + uv_sz:].reshape(h // 2, w // 2).repeat(2, 0).repeat(2, 1)
        rgb = _yuv_to_rgb(Y, U, V)
    elif code == _FMT_NV12:
        y_end = w * h
        Y = arr[:y_end].reshape(h, w)
        uv = arr[y_end:].reshape(h // 2, w)
        U = uv[:, 0::2].repeat(2, 0).repeat(2, 1)
        V = uv[:, 1::2].repeat(2, 0).repeat(2, 1)
        rgb = _yuv_to_rgb(Y, U, V)
    else:
        raise ValueError(f"Unsupported pixel format code: {code}")
    return np.ascontiguousarray(rgb[:, :, ::-1])


# ── agent-facing result types ─────────────────────────────────────────────────

@dataclass
class PoseEstimate:
    """Per-frame pose result handed back to the agent."""
    frame_id: int
    timestamp_us: int
    pose_cam_to_world: Optional[np.ndarray]   # (4,4) SE3 in room frame, None if LOST
    metric_valid: bool                        # False = up-to-scale (bearings ok, metres NOT)
    confidence: float
    is_keyframe: bool
    tracking_state: str                       # "initializing" | "tracking" | "lost"
    reloc_claim: bool = False
    reloc_matched_frame_id: Optional[int] = None
    reloc_score: Optional[float] = None

    @property
    def lost(self) -> bool:
        return self.pose_cam_to_world is None

    def position(self) -> Optional[np.ndarray]:
        if self.pose_cam_to_world is None:
            return None
        return self.pose_cam_to_world[:3, 3].copy()


@dataclass
class BearingEstimate:
    """Horizontal bearing from the current camera to a target position."""
    azimuth_deg: float          # signed; +right / -left in the camera's local frame
    label: str                  # "left" | "center" | "right" | "behind"
    distance_m: Optional[float]  # straight-line distance IFF metric_valid, else None
    metric_valid: bool

    def phrase(self) -> str:
        """Agent-speakable phrase, matching the room-tour _bearing_phrase vocabulary."""
        return {
            "left": "to your left",
            "right": "to your right",
            "center": "right in front of you",
            "behind": "behind you",
        }.get(self.label, "")


@dataclass
class RelocResult:
    """'Where am I' relocalization verdict against the per-session keyframe map."""
    localized: bool
    matched_frame_id: Optional[int]
    score: Optional[float]
    reason: str


# ── the provider ──────────────────────────────────────────────────────────────

class MonoPoseProvider:
    """In-process pose provider wrapping the mono-slam-xr backbone for room-tour.

    Lifecycle (call start/ingest_frame/relocalize off the event loop — the
    backbone is synchronous + GPU-bound):
        provider = MonoPoseProvider(camera_params=...)
        provider.start()                                 # load VO + depth + descriptor
        est = provider.ingest_frame(fd, pts_us=..., frame_id=seq)
        bearing = provider.bearing_to(target_world_xyz)
        reloc = provider.relocalize()
        provider.stop()
    """

    def __init__(
        self,
        *,
        camera_params: CameraParams,
        model_path: str = "models/tartanvo_1914.pkl",
        depth_model_dir: str = "models/depth_anything_v2_metric",
        depth_weights: str = "models/depth_anything_v2_metric_hypersim_vits.pth",
        depth_encoder: str = "vits",
        kf_stride: int = 10,
        enable_reloc: bool = True,
        reloc_sim_threshold: float = 0.75,
        reloc_min_temporal_gap: int = 50,
        center_cone_deg: float = 20.0,
        scratch_dir: Optional[str] = None,
    ) -> None:
        """camera_params must match the LIVE stream intrinsics, not Replica's —
        wrong intrinsics bias both VO direction and depth scale. ``center_cone_deg``
        is the half-angle of the "center" cone for the spoken L/C/R label."""
        self.camera_params = camera_params
        self.center_cone_deg = float(center_cone_deg)

        from tartanvo_candidate_metric_scale import TartanVOMetricScaleCandidate
        self._candidate = TartanVOMetricScaleCandidate(
            model_path=model_path,
            image_width=640,
            image_height=448,
            depth_model_dir=depth_model_dir,
            depth_weights=depth_weights,
            depth_encoder=depth_encoder,
            kf_stride=kf_stride,
            enable_reloc=enable_reloc,
            reloc_sim_threshold=reloc_sim_threshold,
            reloc_min_temporal_gap=reloc_min_temporal_gap,
        )

        # The backbone consumes ImageFrame.image_path, so we materialize one temp
        # PNG per frame. A future backbone process_frame_array() removes this.
        self._scratch = Path(scratch_dir) if scratch_dir else Path(
            tempfile.mkdtemp(prefix="mono_pose_provider_")
        )
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._frame_png = self._scratch / "frame.png"

        self._started = False
        self._last: Optional[PoseEstimate] = None
        self._n_seen = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Load models. Blocking + GPU; call off the event loop."""
        if self._started:
            return
        self._candidate.initialize(self.camera_params)
        self._candidate.reset()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._candidate.teardown()
        finally:
            self._started = False

    def reset_session(self) -> None:
        """Start a fresh room frame / place map (call on 'start room tour')."""
        self._candidate.reset()
        self._last = None
        self._n_seen = 0

    # ── per-frame ─────────────────────────────────────────────────────────────
    def ingest_frame(self, frame_data, *, pts_us: int, frame_id: int) -> PoseEstimate:
        """Feed ONE room-tour FrameData; returns its PoseEstimate.

        Call at a continuous tracking rate (~10 Hz) so VO stays continuous —
        separate from the agent's slow ~3 s VLM perceive loop. Synchronous +
        GPU-bound; run via asyncio.to_thread in the agent."""
        if not self._started:
            raise RuntimeError("MonoPoseProvider.start() not called")
        import cv2

        bgr = frame_data_to_bgr(frame_data)
        cv2.imwrite(str(self._frame_png), bgr)
        frame = ImageFrame(
            timestamp_ns=int(pts_us) * 1000,
            image_path=self._frame_png,
            camera_params=self.camera_params,
        )

        po = self._candidate.process_frame(frame, imu_samples=None)
        self._n_seen += 1

        pose = po.pose_cam_to_world
        if pose is None:
            state = "lost"
        elif self._n_seen <= 1:
            state = "initializing"
        else:
            state = "tracking"

        est = PoseEstimate(
            frame_id=frame_id,
            timestamp_us=int(pts_us),
            pose_cam_to_world=None if pose is None else np.asarray(pose, dtype=np.float64),
            metric_valid=bool(getattr(po, "metric_valid", True)),
            confidence=float(po.confidence),
            is_keyframe=bool(po.is_keyframe),
            tracking_state=state,
            reloc_claim=bool(getattr(po, "reloc_claim", False)),
            reloc_matched_frame_id=getattr(po, "reloc_matched_frame_id", None),
            reloc_score=getattr(po, "reloc_score", None),
        )
        self._last = est
        return est

    @property
    def current_pose(self) -> Optional[PoseEstimate]:
        return self._last

    # ── bearing ───────────────────────────────────────────────────────────────
    def bearing_to(
        self,
        target_world_xyz,
        *,
        up_axis: int = 1,
        from_pose: Optional[np.ndarray] = None,
    ) -> Optional[BearingEstimate]:
        """Horizontal bearing from the current camera to a room-frame target.

        Computed in the camera's LOCAL frame: p_cam = R^T (p_w - t), then
        azimuth = atan2(lateral, forward) in the plane perpendicular to up_axis —
        a signed left/right angle relative to where the wearer is facing. Returns
        None if there is no current pose (agent falls back to VLM L/C/R).
        distance_m is filled only when the current pose is metric_valid."""
        T = from_pose
        if T is None:
            if self._last is None or self._last.pose_cam_to_world is None:
                return None
            T = self._last.pose_cam_to_world
        metric_valid = self._last.metric_valid if (self._last and from_pose is None) else True

        p_w = np.asarray(target_world_xyz, dtype=np.float64).reshape(3)
        R = T[:3, :3]
        t = T[:3, 3]
        p_cam = R.T @ (p_w - t)

        horiz = [i for i in range(3) if i != up_axis]
        x = p_cam[horiz[0]]   # lateral
        z = p_cam[horiz[1]]   # forward
        azimuth = float(np.degrees(np.arctan2(x, z)))  # +right, -left, +-180 behind

        a = abs(azimuth)
        if a <= self.center_cone_deg:
            label = "center"
        elif a >= 180.0 - self.center_cone_deg:
            label = "behind"
        elif azimuth > 0:
            label = "right"
        else:
            label = "left"

        dist = float(np.linalg.norm(p_cam)) if metric_valid else None
        return BearingEstimate(
            azimuth_deg=azimuth, label=label, distance_m=dist, metric_valid=metric_valid
        )

    # ── relocalize ────────────────────────────────────────────────────────────
    def relocalize(self) -> RelocResult:
        """'Where am I' against the per-session keyframe map (DINOv2-CLS).

        Abstain-by-default, high-precision: localized=True only when the
        backbone's own claim threshold was met for the current view. Complementary
        to textslam's text relocalization — fuse for high-stakes nav."""
        last = self._last
        if last is None:
            return RelocResult(False, None, None, "no frame ingested yet")
        if last.reloc_claim and last.reloc_matched_frame_id is not None:
            reason = (
                f"matched stored keyframe {last.reloc_matched_frame_id} "
                f"(score {last.reloc_score:.3f})"
                if last.reloc_score is not None else "matched stored keyframe"
            )
            return RelocResult(True, last.reloc_matched_frame_id, last.reloc_score, reason)
        return RelocResult(
            False, None, last.reloc_score,
            "no confident place match (abstain — fall back to text relocalization)",
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def keyframe_poses(self) -> list[Tuple[int, np.ndarray]]:
        """(frame_id, pose) for every stored keyframe — lets the agent attach a
        room-frame position to a textslam place node (frame_id is shared)."""
        cand = self._candidate
        return list(zip(cand._kf_frame_ids, [p.copy() for p in cand._kf_poses]))
