# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pose MCP server.

Pure FastMCP — every operation is an MCP tool at ``/mcp``.  Takes monocular
images and returns approximate 6DoF poses anchored to a persistent indoor
map.  The first frame ever seen becomes the map origin; subsequent frames
are localized against stored keyframes via XFeat + LighterGlue matches and
OpenCV's PnP RANSAC.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  estimate_pose(image_path, timestamp_us=0) → dict
      Process one frame.  Returns:
        state         : "empty" | "bootstrap" | "localized"
        translation_m : [x, y, z]  (world frame, metres; None if not localized)
        quaternion    : [w, x, y, z]               (None if not localized)
        fov_deg       : horizontal FOV inferred from MoGe
        num_inliers   : PnP inliers used for the pose (0 if not localized)
        num_keyframes : current map size
        matched_kf_id : id of the keyframe we localized against (or None)
        ts_us         : echo of the timestamp used

  get_map_stats() → dict
      Map size + last pose + origin status.  Cheap; does not run any model.

  reset_map() → dict
      Wipe the persistent map.  Next estimate_pose call becomes the new
      origin frame.

Config (pose_mcp_server.yaml)
─────────────────────────────
    map_dir:            /tmp/xr-ai/pose-map           # `~` is expanded
    device:             auto              # "auto" | "cpu" | "cuda" | "cuda:0"
    moge_model:         Ruicheng/moge-2-vits-normal
    max_keyframes:      200
    min_translation_m:  0.30
    min_rotation_deg:   20.0
    min_inliers:        30
    pnp_reproj_err_px:  4.0
    host:               0.0.0.0
    port:               8240
    rerun_addr:         null              # e.g. "127.0.0.1:9876" to stream
                                          # poses + point clouds to a Rerun
                                          # viewer; null disables the sink.
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

from .backends    import MoGeBackend, XFeatBackend
from .localizer   import Localizer
from .store       import KeyframeStore


def _load_image_rgb(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def build_mcp(localizer: Localizer, store: KeyframeStore) -> FastMCP:
    mcp = FastMCP("pose-mcp")

    @mcp.tool()
    def estimate_pose(image_path: str, timestamp_us: int = 0) -> dict:
        """Return an approximate 6DoF pose for *image_path* in the persistent
        map's world frame.

        Parameters
        ----------
        image_path
            Absolute path to a still image (PNG / JPG / WebP) on the server.
        timestamp_us
            Optional Unix-microseconds timestamp; defaults to the server's
            wall clock.  Stored with the keyframe (if one is created) so
            callers can correlate poses with their own frame stream.
        """
        path = pathlib.Path(image_path)
        if not path.exists():
            return {"error": f"image not found: {image_path!r}"}
        try:
            rgb = _load_image_rgb(path)
        except Exception as exc:
            return {"error": f"failed to load image: {exc}"}
        try:
            result = localizer.process(rgb, ts_us=timestamp_us or None)
        except Exception as exc:
            logger.exception("estimate_pose failed for {}", image_path)
            return {"error": f"localization failed: {exc}"}
        logger.info(
            "estimate_pose  state={}  inliers={}  kfs={}  fov={:.1f}°",
            result.state, result.num_inliers, result.num_keyframes, result.fov_deg,
        )
        return {
            "state":         result.state,
            "translation_m": result.translation,
            "quaternion":    result.quaternion,
            "fov_deg":       result.fov_deg,
            "num_inliers":   result.num_inliers,
            "num_keyframes": result.num_keyframes,
            "matched_kf_id": result.matched_kf_id,
            "ts_us":         result.ts_us,
        }

    @mcp.tool()
    def get_map_stats() -> dict:
        """Cheap snapshot of the persistent map (no model inference)."""
        return store.stats()

    @mcp.tool()
    def reset_map() -> dict:
        """Wipe the persistent map.  Next ``estimate_pose`` call seeds the
        origin keyframe."""
        store.reset()
        logger.warning("pose-mcp: map wiped")
        return {"ok": True, "num_keyframes": 0}

    return mcp


def build_app(localizer: Localizer, store: KeyframeStore):
    return build_mcp(localizer, store).http_app(path="/mcp")


async def _serve(cfg: dict, ready_file: pathlib.Path | None) -> None:
    # `~` is expanded so configs can opt into a per-user path without baking
    # a hardcoded /home/<user>/… into version control.
    map_dir = pathlib.Path(
        cfg.get("map_dir", "/tmp/xr-ai/pose-map")
    ).expanduser()
    host    = cfg.get("host", "0.0.0.0")
    port    = int(cfg.get("port", 8240))

    store = KeyframeStore(map_dir)
    geometry = MoGeBackend(
        model_name=cfg.get("moge_model", "Ruicheng/moge-2-vits-normal"),
        device=cfg.get("device", "auto"),
        fov_x_deg=cfg.get("camera_fov_deg") or None,
        calibration_frames=int(cfg.get("fov_calibration_frames", 8)),
    )
    features = XFeatBackend(
        device=cfg.get("device", "auto"),
        top_k=int(cfg.get("xfeat_top_k", 2048)),
        min_conf=float(cfg.get("lighterglue_min_conf", 0.05)),
    )

    viz = None
    rerun_addr = cfg.get("rerun_addr") or None
    if rerun_addr:
        from .viz import RerunSink
        viz = RerunSink(addr=str(rerun_addr))
        # Replay any pre-existing keyframes so the viewer shows the persistent
        # map immediately on connect, not just frames going forward.
        viz.on_load(store.all())
        logger.info("pose-mcp: streaming to Rerun viewer at {}", rerun_addr)

    localizer = Localizer(
        store=store, geometry=geometry, features=features,
        max_keyframes     = int(cfg.get("max_keyframes",     200)),
        min_translation_m = float(cfg.get("min_translation_m", 0.30)),
        min_rotation_deg  = float(cfg.get("min_rotation_deg",  20.0)),
        min_inliers       = int(cfg.get("min_inliers",        30)),
        pnp_reproj_err_px = float(cfg.get("pnp_reproj_err_px", 4.0)),
        viz               = viz,
    )

    app = build_app(localizer, store)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info(
        "pose-mcp-server  map_dir={} (keyframes={})  port={}",
        map_dir, len(store), port,
    )
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

    setup_logging("pose-mcp")
    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
