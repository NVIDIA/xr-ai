# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
droid-mcp server.

FastMCP wrapper around DROID-SLAM (princeton-vl/DROID-SLAM, BSD-3).
Same tool surface as pose-mcp / kimera-mcp so callers swap by URL.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  estimate_pose(image_path, timestamp_us=0) → dict
      Push a grayscale frame to DROID and return the latest pose.

  push_imu(samples) → dict
      Accepted for tool-surface parity, currently dropped — DROID is
      monocular only.  Wire it in if you ever add an IMU prior path.

  set_camera_intrinsics(width, height, fx, fy, cx, cy) → dict
      Set the pinhole K for the tracked image size.

  get_map_stats() → dict
      Frames seen, intrinsics set?  Cheap snapshot.

  reset_map() → dict
      Reset the SLAM session.

GPU-only — DROID needs CUDA.  See scripts/setup_droid.sh to install it.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib

import numpy as np
import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger
from PIL import Image

from xr_ai_logging import setup_logging

from .backend import DroidBackend
from .build   import ensure_droid_installed
from .viz     import RerunSink


def _load_image_gray(path: pathlib.Path,
                     image_size: tuple[int, int]) -> np.ndarray:
    """Load a path as 8-bit grayscale and resize to ``image_size``
    (W, H).  Frames coming through the worker should already be the
    right size; we resize defensively in case a caller skips that."""
    with Image.open(path) as img:
        if img.size != image_size:
            img = img.convert("L").resize(image_size, Image.Resampling.BILINEAR)
        else:
            img = img.convert("L")
        return np.asarray(img, dtype=np.uint8)


def build_mcp(backend: DroidBackend, cfg: dict,
              sink: RerunSink | None = None) -> FastMCP:
    mcp   = FastMCP("droid-mcp")
    state = {"intrinsics": None}

    @mcp.tool()
    def estimate_pose(image_path: str, timestamp_us: int = 0) -> dict:
        path = pathlib.Path(image_path)
        if not path.exists():
            return {"error": f"image not found: {image_path!r}"}
        try:
            gray = _load_image_gray(path, backend.image_size)
        except Exception as exc:
            return {"error": f"failed to load image: {exc}"}

        ts_ns = (
            int(timestamp_us) * 1000 if timestamp_us
            else int(__import__("time").time() * 1e9)
        )
        try:
            pose = backend.track(ts_ns=ts_ns, gray=gray)
        except Exception as exc:
            return {"error": str(exc)}
        result = {
            "state":         pose.state,
            "translation_m": list(pose.translation) if pose.have_pose else None,
            "quaternion":    list(pose.quaternion)  if pose.have_pose else None,
            "frames_sent":   backend.frames_seen,
            "ts_ns":         pose.ts_ns,
        }
        if sink is not None and pose.have_pose:
            sink.log_pose_dict(result, gray=gray)
        return result

    @mcp.tool()
    def push_imu(samples: list[list[float]]) -> dict:
        # Accepted for parity with pose-mcp / kimera-mcp but DROID is
        # monocular-only — silently drop.  If you wire an IMU prior in
        # later, count samples here and emit a metric.
        return {"appended": 0, "dropped_reason": "droid-mcp is monocular-only"}

    @mcp.tool()
    def set_camera_intrinsics(width: int, height: int,
                              fx: float, fy: float,
                              cx: float, cy: float) -> dict:
        try:
            backend.set_intrinsics(width=int(width), height=int(height),
                                   fx=float(fx), fy=float(fy),
                                   cx=float(cx), cy=float(cy))
        except Exception as exc:
            return {"error": str(exc)}
        state["intrinsics"] = dict(width=int(width), height=int(height),
                                   fx=float(fx), fy=float(fy),
                                   cx=float(cx), cy=float(cy))
        if sink is not None:
            K = np.array([[fx, 0,  cx],
                          [0,  fy, cy],
                          [0,  0,   1]], dtype=np.float64)
            sink.set_intrinsics(K)
        return {"ok": True, "intrinsics": state["intrinsics"]}

    @mcp.tool()
    def get_map_stats() -> dict:
        return {
            "frames_sent":      backend.frames_seen,
            "intrinsics":       state["intrinsics"],
            "has_intrinsics":   backend.has_intrinsics,
            "image_size":       list(backend.image_size),
        }

    @mcp.tool()
    def reset_map() -> dict:
        backend.reset()
        if sink is not None:
            sink.reset()
        return {"ok": True}

    return mcp


def build_app(backend: DroidBackend, cfg: dict,
              sink: RerunSink | None = None):
    return build_mcp(backend, cfg, sink=sink).http_app(path="/mcp")


async def _serve(cfg: dict, ready_file: pathlib.Path | None) -> None:
    host = cfg.get("host", "0.0.0.0")
    port = int(cfg.get("port", 8260))

    # Auto-install droid_slam on first launch if it isn't importable.
    # Set `auto_install: false` in the YAML to opt out (e.g. on
    # sandboxed CI hosts) and fail fast instead of running the clone +
    # CUDA-compile + weights download flow implicitly.
    if bool(cfg.get("auto_install", True)):
        ensure_droid_installed()

    weights = cfg.get("weights_path", "~/.cache/xr-ai/droid.pth")
    weights = str(pathlib.Path(weights).expanduser())
    img_w   = int(cfg.get("image_width",  320))
    img_h   = int(cfg.get("image_height", 240))

    backend = DroidBackend(
        weights_path=weights,
        image_size  =(img_w, img_h),
        device      =cfg.get("device", "cuda:0"),
        buffer      =int(cfg.get("buffer", 512)),
    )

    sink: RerunSink | None = None
    rerun_addr = cfg.get("rerun_addr")
    if rerun_addr:
        sink = RerunSink(addr=str(rerun_addr), image_size=(img_w, img_h))

    # Push default intrinsics so the first frame doesn't fail with
    # "set_intrinsics first" if the worker isn't fast enough.
    if "default_fx" in cfg:
        fx = float(cfg["default_fx"])
        fy = float(cfg.get("default_fy", fx))
        cx = float(cfg.get("default_cx", img_w / 2.0))
        cy = float(cfg.get("default_cy", img_h / 2.0))
        backend.set_intrinsics(width=img_w, height=img_h,
                               fx=fx, fy=fy, cx=cx, cy=cy)
        if sink is not None:
            sink.set_intrinsics(np.array([[fx, 0, cx],
                                          [0, fy, cy],
                                          [0,  0,  1]], dtype=np.float64))

    app    = build_app(backend, cfg, sink=sink)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info("droid-mcp-server  port={}  image={}x{}  weights={}",
                port, img_w, img_h, weights)
    if ready_file:
        ready_file.touch()
    await server.serve()


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    setup_logging("droid-mcp")
    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
