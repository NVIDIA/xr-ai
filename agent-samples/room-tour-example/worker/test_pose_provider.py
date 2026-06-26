# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backbone-free unit tests for the room-tour pose adapter.

These cover the two pieces of ``pose_provider.py`` that are pure (no GPU, no
mono-slam backbone, no model assets) and carry the integration's correctness:

  1. ``frame_data_to_bgr`` — the FrameData→BGR pixel conversion the backbone is
     fed. Its whole point is to hand the SLAM the SAME pixels the VLM path sees
     via ``pixels.frame_to_pil``, just in BGR order. We assert byte-exact parity
     for all 5 hub pixel formats, so a future drift in ``pixels.py`` (or the
     adapter) that silently desynchronises the two can't slip through.
  2. ``compute_bearing`` — the geometric bearing that replaces the VLM's coarse
     left/center/right. We assert the local-frame azimuth sign + L/C/R/behind
     label and that distance is gated on ``metric_valid``.

Also asserts the soft-dependency contract: the module imports, and the pure
functions work, with NO backbone deployed (``MONO_SLAM_WORKSPACE`` unset).

Run standalone (no pytest needed) from the worker dir:
    .venv/bin/python test_pose_provider.py
or under pytest:
    .venv/bin/python -m pytest test_pose_provider.py
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
from PIL import Image  # noqa: F401 - ensures the worker's imaging stack is present

from pixels import frame_to_pil
from pose_provider import compute_bearing, frame_data_to_bgr, make_camera_params
from xr_ai_agent import PixelFormat


# ── fixtures ────────────────────────────────────────────────────────────────

def _sample_rgb(w: int = 16, h: int = 8) -> np.ndarray:
    """A deterministic, non-degenerate RGB image with all three channels varying
    (gradients + a couple of solid blocks) so a channel swap or reshape bug shows."""
    rgb = np.zeros((h, w, 3), np.uint8)
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.linspace(0, 255, h, dtype=np.uint8)
    rgb[:, :, 0] = xs[None, :]            # R ramps across x
    rgb[:, :, 1] = ys[:, None]            # G ramps down y
    rgb[:, :, 2] = 128                    # B constant
    rgb[:2, :2] = (255, 0, 0)             # red corner — catches R/B swap
    rgb[-2:, -2:] = (0, 0, 255)           # blue corner
    return rgb


def _encode(rgb: np.ndarray, fmt: PixelFormat) -> bytes:
    """Pack an RGB image into the raw bytes the hub delivers for *fmt* — the
    inverse of pixels.py's reshape, with the forward BT.601 matrix for YUV."""
    h, w = rgb.shape[:2]
    if fmt == PixelFormat.RGB24:
        return np.ascontiguousarray(rgb).tobytes()
    if fmt == PixelFormat.RGBA:
        a = np.full((h, w, 1), 255, np.uint8)
        return np.ascontiguousarray(np.concatenate([rgb, a], axis=2)).tobytes()
    if fmt == PixelFormat.BGRA:
        bgr = rgb[:, :, ::-1]
        a = np.full((h, w, 1), 255, np.uint8)
        return np.ascontiguousarray(np.concatenate([bgr, a], axis=2)).tobytes()
    # YUV 4:2:0 — forward BT.601 limited range (matches pixels.py's inverse).
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    Y = np.clip(0.257 * R + 0.504 * G + 0.098 * B + 16.0, 0, 255).astype(np.uint8)
    U = np.clip(-0.148 * R - 0.291 * G + 0.439 * B + 128.0, 0, 255)
    V = np.clip(0.439 * R - 0.368 * G - 0.071 * B + 128.0, 0, 255)
    Us = U[0::2, 0::2].astype(np.uint8)
    Vs = V[0::2, 0::2].astype(np.uint8)
    if fmt == PixelFormat.I420:
        return Y.tobytes() + Us.tobytes() + Vs.tobytes()
    if fmt == PixelFormat.NV12:
        uv = np.empty((h // 2, w), np.uint8)
        uv[:, 0::2] = Us
        uv[:, 1::2] = Vs
        return Y.tobytes() + uv.tobytes()
    raise ValueError(fmt)


def _frame(rgb: np.ndarray, fmt: PixelFormat):
    """Duck-typed FrameData stand-in (the fields both converters read)."""
    h, w = rgb.shape[:2]
    return SimpleNamespace(width=w, height=h, fmt=fmt, data=_encode(rgb, fmt))


# ── frame_data_to_bgr ↔ pixels.frame_to_pil parity ───────────────────────────

def test_frame_data_to_bgr_matches_pixels_for_every_format():
    """The backbone must see byte-identical pixels to the VLM path — i.e.
    frame_data_to_bgr(fd) == BGR(frame_to_pil(fd)) — for ALL 5 hub formats.
    Both apply the same reshape + BT.601 math, so this is exact even for the
    lossy 4:2:0 formats (same input bytes → same conversion, only channel order
    differs)."""
    rgb = _sample_rgb()
    for fmt in (PixelFormat.RGB24, PixelFormat.RGBA, PixelFormat.BGRA,
                PixelFormat.I420, PixelFormat.NV12):
        fd = _frame(rgb, fmt)
        bgr = frame_data_to_bgr(fd)
        pil_rgb = np.asarray(frame_to_pil(fd))
        assert bgr.shape == (rgb.shape[0], rgb.shape[1], 3), f"{fmt.name}: shape {bgr.shape}"
        assert bgr.dtype == np.uint8, f"{fmt.name}: dtype {bgr.dtype}"
        assert np.array_equal(bgr, pil_rgb[:, :, ::-1]), \
            f"{fmt.name}: frame_data_to_bgr diverged from pixels.frame_to_pil"


def test_frame_data_to_bgr_rgb_formats_are_lossless():
    """RGB24/RGBA/BGRA carry full chroma — the conversion must be lossless."""
    rgb = _sample_rgb()
    for fmt in (PixelFormat.RGB24, PixelFormat.RGBA, PixelFormat.BGRA):
        bgr = frame_data_to_bgr(_frame(rgb, fmt))
        assert np.array_equal(bgr, rgb[:, :, ::-1]), f"{fmt.name}: not lossless"


def test_frame_data_to_bgr_accepts_enum_name_and_int():
    """fmt is duck-typed: the PixelFormat enum, its .name, or its int all work
    (so the adapter doesn't hard-depend on the xr_ai_agent enum identity)."""
    rgb = _sample_rgb()
    raw = _encode(rgb, PixelFormat.RGB24)
    h, w = rgb.shape[:2]
    for fmt in (PixelFormat.RGB24, "RGB24", int(PixelFormat.RGB24)):
        fd = SimpleNamespace(width=w, height=h, fmt=fmt, data=raw)
        assert np.array_equal(frame_data_to_bgr(fd), rgb[:, :, ::-1]), f"fmt={fmt!r}"


# ── compute_bearing geometry ─────────────────────────────────────────────────

def test_compute_bearing_labels_and_sign():
    """Camera at the origin facing +z (identity pose): a target to the +x side is
    a +azimuth 'right', -x is 'left', straight ahead is 'center', and behind the
    camera (−z) reads 'behind' near ±180°."""
    eye = np.eye(4)
    right = compute_bearing(eye, [3.0, 0.0, 1.0], metric_valid=True)
    assert right.azimuth_deg > 0 and right.label == "right"
    left = compute_bearing(eye, [-3.0, 0.0, 1.0], metric_valid=True)
    assert left.azimuth_deg < 0 and left.label == "left"
    center = compute_bearing(eye, [0.0, 0.0, 5.0], metric_valid=True)
    assert center.label == "center" and abs(center.azimuth_deg) < 1e-6
    behind = compute_bearing(eye, [0.0, 0.0, -5.0], metric_valid=True)
    assert behind.label == "behind" and abs(abs(behind.azimuth_deg) - 180.0) < 1e-6


def test_compute_bearing_distance_gated_on_metric_valid():
    """Distance is spoken only for a metric-valid pose; an up-to-scale pose must
    report distance_m = None (a single camera's absolute scale is unreliable)."""
    eye = np.eye(4)
    metric = compute_bearing(eye, [0.0, 0.0, 4.0], metric_valid=True)
    assert metric.distance_m is not None and abs(metric.distance_m - 4.0) < 1e-6
    up_to_scale = compute_bearing(eye, [0.0, 0.0, 4.0], metric_valid=False)
    assert up_to_scale.distance_m is None
    assert up_to_scale.label == "center"   # bearing still valid up-to-scale


def test_compute_bearing_uses_camera_orientation():
    """Bearing is in the camera's LOCAL frame: rotating the camera 90° about the
    up axis (so it faces +x) flips a +x-world target to straight ahead."""
    # Camera faces +x: columns are the camera axes expressed in world.
    # forward(+z_cam)→+x_world, so a target at +x world is 'center'.
    Ry = np.eye(4)
    Ry[:3, :3] = np.array([[0.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0]])
    b = compute_bearing(Ry, [5.0, 0.0, 0.0], metric_valid=True)
    assert b.label == "center", f"expected center, got {b.label} ({b.azimuth_deg:.1f}°)"


# ── soft-dependency contract ─────────────────────────────────────────────────

def test_make_camera_params_raises_without_backbone():
    """With no backbone deployed (MONO_SLAM_WORKSPACE unset), the pure functions
    still work (tested above) and only backbone construction fails — cleanly, so
    the brain stays pose-free."""
    old = os.environ.pop("MONO_SLAM_WORKSPACE", None)
    try:
        raised = False
        try:
            make_camera_params(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
        except RuntimeError:
            raised = True
        assert raised, "make_camera_params must raise RuntimeError when the backbone is absent"
    finally:
        if old is not None:
            os.environ["MONO_SLAM_WORKSPACE"] = old


# ── standalone runner (no pytest required) ───────────────────────────────────

def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
    print(f"[PASS] {len(tests)} pose-adapter tests (backbone-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
