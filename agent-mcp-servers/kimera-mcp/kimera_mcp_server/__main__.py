# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
kimera-mcp server.

Watch-folder wrapper around MIT-SPARK Kimera-VIO.  Exposes the same
MCP tool surface as pose-mcp (`estimate_pose`, `get_map_stats`,
`reset_map`, `push_imu`, `set_camera_intrinsics`) so existing callers
swap by flipping a single config URL.

Each ``estimate_pose`` call:
  1. Atomically writes the incoming frame as
     ``<dataset_dir>/mav0/cam0/data/<ts_ns>.png`` + appends to
     ``cam0/data.csv``.
  2. Optionally consumes any IMU samples that arrived via ``push_imu``
     since the last call and appends them to ``imu0/data.csv``.
  3. Invokes Kimera-VIO in the ``kimera_vio`` docker image over the
     window of recent frames currently on disk.
  4. Parses the latest pose out of Kimera's ``traj_vio.csv`` and
     returns it.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  estimate_pose(image_path, timestamp_us=0) → dict
      state         : "ok" | "uninitialized" | "lost"
      translation_m : [x, y, z]   world ← camera
      quaternion    : [w, x, y, z]
      num_frames    : frames currently staged
      ts_ns         : nanosecond timestamp echo

  push_imu(samples) → dict
      samples is a list of [ts_ms, gx, gy, gz, ax, ay, az].
      Appended to imu0/data.csv on the next estimate_pose.

  set_camera_intrinsics(width, height, fx, fy, cx, cy) → dict
      Rewrites cam0/sensor.yaml so Kimera uses these intrinsics.

  get_map_stats() → dict
      Cheap snapshot.

  reset_map() → dict
      Wipe staged data and any previous trajectory output.
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

from .dataset import CameraIntrinsics, EurocDatasetWriter
from .runner  import KimeraRunner


def _load_image_rgb(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def build_mcp(writer: EurocDatasetWriter, runner: KimeraRunner,
              cfg: dict) -> FastMCP:
    mcp = FastMCP("kimera-mcp")
    window_frames = int(cfg.get("window_frames", 100))
    state = {"frame_skip": int(cfg.get("frame_skip", 1)),
             "seen_frames": 0}

    @mcp.tool()
    def estimate_pose(image_path: str, timestamp_us: int = 0) -> dict:
        """Stage the frame, invoke Kimera, return the latest pose."""
        path = pathlib.Path(image_path)
        if not path.exists():
            return {"error": f"image not found: {image_path!r}"}
        try:
            rgb = _load_image_rgb(path)
        except Exception as exc:
            return {"error": f"failed to load image: {exc}"}

        state["seen_frames"] += 1
        if state["frame_skip"] > 1 and (state["seen_frames"] % state["frame_skip"]) != 0:
            return {"state": "skipped", "translation_m": None, "quaternion": None,
                    "num_frames": writer.num_frames, "ts_ns": 0}

        ts_ns = int(timestamp_us) * 1000 if timestamp_us else int(__import__("time").time() * 1e9)
        writer.append_frame(ts_ns, rgb)

        pose = runner.run_once(num_frames=writer.num_frames)
        if pose is None:
            return {"state": "uninitialized" if writer.num_frames < 8 else "lost",
                    "translation_m": None, "quaternion": None,
                    "num_frames": writer.num_frames, "ts_ns": ts_ns}

        t = pose.pose[:3, 3]
        return {
            "state":         "ok",
            "translation_m": [float(t[0]), float(t[1]), float(t[2])],
            "quaternion":    list(pose.quaternion),
            "num_frames":    writer.num_frames,
            "ts_ns":         pose.ts_ns,
        }

    @mcp.tool()
    def push_imu(samples: list[list[float]]) -> dict:
        """Append IMU samples to the staged dataset.

        Each sample is a 7-element list: [ts_ms, gx, gy, gz, ax, ay, az].
        Gyro in rad/s, accel in m/s².  Timestamps in milliseconds (the
        worker converts the web client's batched JSON into this form).
        """
        n = 0
        for s in samples:
            if len(s) != 7:
                continue
            ts_ns = int(float(s[0]) * 1_000_000.0)
            writer.append_imu(ts_ns, (s[1], s[2], s[3]), (s[4], s[5], s[6]))
            n += 1
        return {"appended": n}

    @mcp.tool()
    def set_camera_intrinsics(width: int, height: int,
                              fx: float, fy: float,
                              cx: float, cy: float) -> dict:
        """Rewrite cam0/sensor.yaml with the operator-provided intrinsics
        (e.g. from the web client's `camera_meta` topic + a FOV heuristic).
        Wipes the current dataset because old frames assume the old K."""
        intr = CameraIntrinsics(width=int(width), height=int(height),
                                fx=float(fx), fy=float(fy),
                                cx=float(cx), cy=float(cy))
        writer.reset()
        writer._intrinsics = intr      # in-place — re-init writes the new yaml
        writer._init_layout()
        return {"ok": True, "intrinsics": dataclasses_to_dict(intr)}

    @mcp.tool()
    def get_map_stats() -> dict:
        return {
            "num_frames":      writer.num_frames,
            "window_frames":   window_frames,
            "dataset_dir":     str(writer.root),
            "latest_ts_ns":    writer.latest_ts_ns,
            "frames_seen":     state["seen_frames"],
        }

    @mcp.tool()
    def reset_map() -> dict:
        writer.reset()
        state["seen_frames"] = 0
        logger.warning("kimera-mcp: dataset wiped")
        return {"ok": True, "num_frames": 0}

    return mcp


def dataclasses_to_dict(obj) -> dict:
    import dataclasses as _dc
    return _dc.asdict(obj) if _dc.is_dataclass(obj) else dict(obj)


def build_app(writer: EurocDatasetWriter, runner: KimeraRunner, cfg: dict):
    return build_mcp(writer, runner, cfg).http_app(path="/mcp")


async def _serve(cfg: dict, ready_file: pathlib.Path | None) -> None:
    dataset_dir = pathlib.Path(cfg.get("dataset_dir", "/tmp/xr-ai/kimera-dataset")).expanduser()
    output_dir  = pathlib.Path(cfg.get("output_dir",  "/tmp/xr-ai/kimera-output")).expanduser()
    host        = cfg.get("host", "0.0.0.0")
    port        = int(cfg.get("port", 8250))

    # Default intrinsics — operator overrides via set_camera_intrinsics or
    # the worker's camera_meta heuristic.  EuRoC-ish numbers as a starting
    # point so even unconfigured first frames are sane.
    default_intr = CameraIntrinsics(
        width=int(cfg.get("default_width",  640)),
        height=int(cfg.get("default_height", 480)),
        fx=float(cfg.get("default_fx", 458.654)),
        fy=float(cfg.get("default_fy", 457.296)),
        cx=float(cfg.get("default_cx", 367.215)),
        cy=float(cfg.get("default_cy", 248.375)),
    )
    writer = EurocDatasetWriter(
        dataset_dir, intrinsics=default_intr,
        window_frames=int(cfg.get("window_frames", 100)),
    )
    runner = KimeraRunner(
        dataset_dir=dataset_dir, output_dir=output_dir,
        docker_image=cfg.get("docker_image", "kimera_vio"),
        params_folder_in_image=cfg.get("params_folder_in_image",
                                       "/root/Kimera-VIO/params/EurocMono"),
    )
    app = build_app(writer, runner, cfg)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info("kimera-mcp-server  dataset={}  port={}", dataset_dir, port)
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

    setup_logging("kimera-mcp")
    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
