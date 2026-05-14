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

    def __init__(self, *, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device     = device
        self._model      = None  # lazy; loaded on first __call__
        self._lock       = threading.Lock()

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
        torch = self._torch
        H, W = image_rgb.shape[:2]
        tensor = torch.from_numpy(
            np.ascontiguousarray(image_rgb).transpose(2, 0, 1).astype(np.float32) / 255.0
        ).to(self._device)
        with self._lock, torch.no_grad():
            out = self._model.infer(tensor)
        points = out["points"].detach().cpu().numpy()             # (H, W, 3)
        mask   = out["mask"].detach().cpu().numpy().astype(bool)  # (H, W)
        K      = out["intrinsics"].detach().cpu().numpy()         # 3x3 normalized
        # MoGe normalizes intrinsics to image dimensions; fx is in [0, 1] of W.
        fx = float(K[0, 0]) * W
        fov_deg = float(2.0 * np.degrees(np.arctan(0.5 * W / fx)))
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

    def __init__(self, *, device: str, top_k: int = 2048) -> None:
        self._device = device
        self._top_k  = top_k
        self._model  = None
        self._lock   = threading.Lock()

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
        torch = self._torch
        H, W = image_rgb.shape[:2]
        with self._lock, torch.no_grad():
            out = self._model.detectAndCompute(self._to_tensor(image_rgb), top_k=self._top_k)[0]
        kp   = out["keypoints"].detach().cpu().numpy().astype(np.float32)
        desc = out["descriptors"].detach().cpu().numpy().astype(np.float32)
        return FrameFeatures(kp=kp, desc=desc, image_size=(int(W), int(H)))

    def match(self, a: FrameFeatures, b: FrameFeatures) -> np.ndarray:
        self._ensure_loaded()
        torch = self._torch
        # XFeat exposes match_lighterglue when LighterGlue weights are present;
        # fall back to its mutual-nearest descriptor matcher otherwise.
        if hasattr(self._model, "match_lighterglue"):
            if a.image_size is None or b.image_size is None:
                raise ValueError(
                    "match_lighterglue requires image_size on both FrameFeatures "
                    "— make sure keyframes were re-loaded after the backend update."
                )
            with self._lock, torch.no_grad():
                idx_a, idx_b = self._model.match_lighterglue(
                    {"keypoints":   torch.from_numpy(a.kp).to(self._device),
                     "descriptors": torch.from_numpy(a.desc).to(self._device),
                     "image_size":  a.image_size},
                    {"keypoints":   torch.from_numpy(b.kp).to(self._device),
                     "descriptors": torch.from_numpy(b.desc).to(self._device),
                     "image_size":  b.image_size},
                )
            ia = idx_a.detach().cpu().numpy().astype(np.int32).reshape(-1)
            ib = idx_b.detach().cpu().numpy().astype(np.int32).reshape(-1)
            return np.stack([ia, ib], axis=1)
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
