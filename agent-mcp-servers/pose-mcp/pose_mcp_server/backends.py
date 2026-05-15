# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy wrappers around MoGe (metric geometry) and XFeat (feature matching).

Both backends are imported on first use rather than at module load: the MCP
server should start in well under a second even though the model wheels pull
in torch and DINOv2.  The wrappers expose a tiny, dependency-free interface
that ``localizer.py`` consumes, which keeps the rest of the package easy to
unit-test with a fake backend.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Protocol

import numpy as np


@dataclasses.dataclass(frozen=True)
class GeometryFrame:
    """One frame's monocular metric geometry."""
    points3d: np.ndarray   # (H, W, 3) float32; metric, camera frame, +Z forward
    mask:     np.ndarray   # (H, W) bool, True where points3d is valid
    fov_deg:  float        # horizontal FOV in degrees
    width:    int
    height:   int


@dataclasses.dataclass(frozen=True)
class FrameFeatures:
    """One frame's local feature set."""
    kp:    np.ndarray   # (N, 2) float32 (x, y) in pixels
    desc:  np.ndarray   # (N, D) float32
    # (width, height) of the source image — required by LightGlue's positional
    # encoding.  Optional only so tests can construct synthetic feature sets;
    # any real backend must populate it.
    image_size: tuple[int, int] | None = None


class GeometryBackend(Protocol):
    def __call__(self, image_rgb: np.ndarray) -> GeometryFrame: ...


class FeatureBackend(Protocol):
    def extract(self, image_rgb: np.ndarray) -> FrameFeatures: ...
    def match(self, a: FrameFeatures, b: FrameFeatures) -> np.ndarray:
        """Return (M, 2) int32 array of matched indices into a.kp / b.kp."""


# ── MoGe -----------------------------------------------------------------------

class MoGeBackend:
    """Wraps ``moge.model.v2.MoGeModel`` to return :class:`GeometryFrame`.

    The model lives on the configured device after the first call; subsequent
    calls just pay the forward-pass cost.  Concurrent calls are serialized
    behind ``_lock`` because the model isn't thread-safe.
    """

    def __init__(
        self, *,
        model_name:          str,
        device:              str,
        fov_x_deg:           float | None = None,
        calibration_frames:  int          = 8,
        border_px:           int          = 20,
    ) -> None:
        self._model_name         = model_name
        self._device             = device
        self._model              = None  # lazy; loaded on first __call__
        self._lock               = threading.Lock()
        # MoGe's depth is unreliable in the outermost ~20 px of every frame:
        # the receptive field falls off there, and flat surfaces visibly
        # "fold in" toward the camera.  Zero out the validity mask in that
        # margin so those pixels never participate in PnP or the rendered
        # point cloud.  Set to 0 to disable; tune up for wider-FOV cameras.
        self._border_px          = max(0, int(border_px))
        # FOV calibration.  If the operator provided `fov_x_deg` we skip
        # calibration entirely and pin that value forever.  Otherwise we
        # let MoGe guess on the first `calibration_frames` frames, collect
        # its estimates, and pin the median — robust to the occasional
        # extreme outlier on a low-texture frame.
        self._calibration_frames = max(1, int(calibration_frames))
        self._fov_samples: list[float] = []
        self._pinned_fov: float | None = (
            float(fov_x_deg) if fov_x_deg is not None else None
        )

    @property
    def is_calibrated(self) -> bool:
        return self._pinned_fov is not None

    @property
    def pinned_fov_deg(self) -> float | None:
        return self._pinned_fov

    def set_pinned_fov_deg(self, fov_deg: float | None) -> None:
        """Externally pin the FOV — e.g. when the client publishes its
        camera metadata, the worker can extract a known FOV (lookup
        table or resolution-based heuristic) and call this to short-
        circuit MoGe's per-frame estimation.

        Pass ``None`` to clear and resume MoGe-based calibration.
        """
        from loguru import logger
        self._pinned_fov  = (None if fov_deg is None else float(fov_deg))
        # Reset the calibration buffer so a new pin doesn't get averaged
        # with stale samples on the next inference.
        self._fov_samples.clear()
        if self._pinned_fov is not None:
            logger.info("MoGe FOV externally pinned to {:.1f}° (skipping calibration)",
                        self._pinned_fov)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import time
        import torch
        from loguru import logger
        from moge.model.v2 import MoGeModel
        device = _resolve_device(self._device)
        logger.info(
            "MoGe: loading {} on {} (first call — may download weights from HuggingFace)",
            self._model_name, device,
        )
        t0 = time.monotonic()
        m = MoGeModel.from_pretrained(self._model_name).to(device).eval()
        logger.info("MoGe: ready  ({:.1f}s)", time.monotonic() - t0)
        self._model  = m
        self._device = device
        self._torch  = torch

    def __call__(self, image_rgb: np.ndarray) -> GeometryFrame:
        self._ensure_loaded()
        from loguru import logger
        torch = self._torch
        H, W = image_rgb.shape[:2]
        tensor = torch.from_numpy(
            np.ascontiguousarray(image_rgb).transpose(2, 0, 1).astype(np.float32) / 255.0
        ).to(self._device)
        # Once pinned, hand MoGe the FOV as a prior so its points are
        # consistent with that intrinsic.  Before pinning, let it guess.
        infer_kwargs = {}
        if self._pinned_fov is not None:
            infer_kwargs["fov_x"] = float(self._pinned_fov)
        with self._lock, torch.no_grad():
            out = self._model.infer(tensor, **infer_kwargs)
        points = out["points"].detach().cpu().numpy()
        mask   = out["mask"].detach().cpu().numpy().astype(bool)
        # Drop the unreliable border before downstream code ever sees the
        # mask — keyframe storage, PnP lookup, and viz all use this field.
        if self._border_px > 0:
            b = self._border_px
            mask[:b, :] = False
            mask[-b:, :] = False
            mask[:, :b] = False
            mask[:, -b:] = False
        # MoGe normalizes intrinsics so fov_x = 2 * arctan(0.5 / K[0,0]).
        K      = out["intrinsics"].detach().cpu().numpy()
        fov_deg = float(2.0 * np.degrees(np.arctan(0.5 / float(K[0, 0]))))

        if self._pinned_fov is None:
            self._fov_samples.append(fov_deg)
            logger.info(
                "MoGe FOV calibration  sample {}/{}  fov={:.1f}°",
                len(self._fov_samples), self._calibration_frames, fov_deg,
            )
            if len(self._fov_samples) >= self._calibration_frames:
                pinned = float(np.median(self._fov_samples))
                self._pinned_fov = pinned
                logger.info(
                    "MoGe FOV pinned to {:.1f}° (median of {} samples; range {:.1f}°–{:.1f}°)",
                    pinned, len(self._fov_samples),
                    min(self._fov_samples), max(self._fov_samples),
                )

        return GeometryFrame(
            points3d=points.astype(np.float32),
            mask=mask,
            fov_deg=fov_deg,
            width=W,
            height=H,
        )


# ── XFeat + LighterGlue --------------------------------------------------------

class XFeatBackend:
    """Wraps the XFeat detector + LighterGlue matcher (Apache-2.0).

    Loaded via ``torch.hub`` from ``verlab/accelerated_features`` so we don't
    have to vendor the model code.  Re-uses the same model instance across
    calls and serializes them behind a lock.
    """

    def __init__(
        self, *,
        device:           str,
        top_k:            int   = 512,
        min_conf:         float = 0.05,
        use_lighterglue:  bool  = False,
    ) -> None:
        self._device   = device
        self._top_k    = top_k
        # LighterGlue's default min_conf=0.1 is aggressive on indoor / low-
        # texture scenes — most legitimate matches sit around 0.05–0.15 and
        # get filtered out, leaving 0–5 matches per pair.  0.05 keeps the
        # filter on enough to drop obviously-bad matches while letting PnP
        # see real correspondences.
        self._min_conf = float(min_conf)
        # When False, skip LighterGlue entirely and use the mutual-NN
        # cosine fallback — much faster (~5 ms vs ~70 ms) at the cost of
        # weaker outlier rejection (RANSAC PnP picks up the slack).
        self._use_lighterglue = bool(use_lighterglue)
        self._model    = None
        self._lock     = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import time
        import torch
        from loguru import logger
        device = _resolve_device(self._device)
        logger.info("XFeat: loading on {} via torch.hub …", device)
        t0 = time.monotonic()
        # trust_repo=True skips the interactive y/N prompt torch.hub
        # introduced in 1.12 — without it, headless subprocesses (e.g. running
        # under xr-ai-launcher with no TTY) silently hang on first load.
        m = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat", pretrained=True, top_k=self._top_k,
            trust_repo=True,
        )
        # XFeat keeps internal params as buffers; move once and keep them put.
        m = m.to(device)
        # XFeat hard-codes `self.dev` at __init__ from torch.cuda.is_available()
        # and uses it inside `parse_input` / `match_lighterglue` to move
        # incoming tensors.  `.to(device)` doesn't touch that attribute, so
        # on a host where cuda.is_available() lies (e.g. driver mismatch)
        # the weights end up on cpu while inputs go to a dead cuda.
        # Force-overwrite the attribute so it always tracks what we asked for.
        try:
            m.dev = torch.device(device)
        except AttributeError:
            pass
        logger.info("XFeat: ready  ({:.1f}s)", time.monotonic() - t0)
        self._model  = m
        self._device = device
        self._torch  = torch

    def _to_tensor(self, image_rgb: np.ndarray):
        torch = self._torch
        arr = np.ascontiguousarray(image_rgb).transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).to(self._device)

    def extract(self, image_rgb: np.ndarray) -> FrameFeatures:
        self._ensure_loaded()
        from loguru import logger
        torch = self._torch
        H, W = image_rgb.shape[:2]
        with self._lock, torch.no_grad():
            out = self._model.detectAndCompute(self._to_tensor(image_rgb), top_k=self._top_k)[0]
        kp   = out["keypoints"].detach().cpu().numpy().astype(np.float32)
        desc = out["descriptors"].detach().cpu().numpy().astype(np.float32)
        # Sanity log: if this drops far below top_k on every frame the scene
        # is too low-texture for matching to ever work.
        if kp.shape[0] < self._top_k // 4:
            logger.debug(
                "XFeat extracted only {} / {} features from {}x{} image",
                kp.shape[0], self._top_k, W, H,
            )
        return FrameFeatures(kp=kp, desc=desc, image_size=(int(W), int(H)))

    def match(self, a: FrameFeatures, b: FrameFeatures) -> np.ndarray:
        self._ensure_loaded()
        torch = self._torch
        # XFeat exposes match_lighterglue when LighterGlue weights are present;
        # fall back to its mutual-nearest descriptor matcher otherwise.
        if self._use_lighterglue and hasattr(self._model, "match_lighterglue"):
            if a.image_size is None or b.image_size is None:
                raise ValueError(
                    "match_lighterglue requires image_size on both FrameFeatures "
                    "— make sure keyframes were re-loaded after the backend update."
                )
            with self._lock, torch.no_grad():
                _, _, idx_pairs = self._model.match_lighterglue(
                    {"keypoints":   torch.from_numpy(a.kp).to(self._device),
                     "descriptors": torch.from_numpy(a.desc).to(self._device),
                     "image_size":  a.image_size},
                    {"keypoints":   torch.from_numpy(b.kp).to(self._device),
                     "descriptors": torch.from_numpy(b.desc).to(self._device),
                     "image_size":  b.image_size},
                    min_conf=self._min_conf,
                )
            if hasattr(idx_pairs, "detach"):
                idx_pairs = idx_pairs.detach().cpu().numpy()
            return np.asarray(idx_pairs, dtype=np.int32).reshape(-1, 2)
        # Mutual nearest neighbour on L2-normalized descriptors.
        a_d = a.desc / (np.linalg.norm(a.desc, axis=1, keepdims=True) + 1e-9)
        b_d = b.desc / (np.linalg.norm(b.desc, axis=1, keepdims=True) + 1e-9)
        sim = a_d @ b_d.T
        nn_b = sim.argmax(axis=1)
        nn_a = sim.argmax(axis=0)
        mutual = np.arange(len(a_d))[nn_a[nn_b] == np.arange(len(a_d))]
        return np.stack([mutual, nn_b[mutual]], axis=1)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
