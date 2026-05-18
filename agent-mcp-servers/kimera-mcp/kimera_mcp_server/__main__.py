# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
kimera-mcp server.

Thin FastMCP front-end for ``kimera_live_vio`` — a long-running
Kimera-VIO pipeline in the ``kimera_vio`` docker image, talking to us
over an AF_UNIX socket.  The container is spawned at startup and torn
down on shutdown.

Tools (FastMCP, mounted at /mcp)
────────────────────────────────
  estimate_pose(image_path, timestamp_us=0) → dict
      Streams the grayscale frame to the pipeline; returns the latest
      pose the backend has produced so far.  Fields:
          state         : "ok" | "uninitialized" | "lost"
          translation_m : [x, y, z]   world ← camera
          quaternion    : [w, x, y, z]
          ts_ns         : nanosecond timestamp of the returned pose

  push_imu(samples) → dict
      samples is a list of [ts_ms, gx, gy, gz, ax, ay, az].
      Forwarded immediately to the pipeline (no buffering on this side).

  set_camera_intrinsics(width, height, fx, fy, cx, cy) → dict
      Restarts the pipeline with the new K.  Call once after the web
      client posts its camera_meta payload.

  get_map_stats() → dict
      Cheap snapshot (frames pushed, pipeline up?).

  reset_map() → dict
      Tear down the current pipeline.  The next frame rebuilds with
      the most recent intrinsics (or the YAML defaults if none).
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import pathlib
import signal

import numpy as np
import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger
from PIL import Image

from xr_ai_logging import setup_logging

from .live import KimeraLiveClient, KimeraLiveError


def _load_image_gray(path: pathlib.Path) -> np.ndarray:
    """Load an image as 8-bit grayscale (the format kimera_live_vio
    expects on the wire)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def build_mcp(client: KimeraLiveClient, cfg: dict) -> FastMCP:
    mcp   = FastMCP("kimera-mcp")
    state = {"frames_sent": 0, "intrinsics": None}

    @mcp.tool()
    def estimate_pose(image_path: str, timestamp_us: int = 0) -> dict:
        """Push the frame to the live pipeline and return the latest pose."""
        path = pathlib.Path(image_path)
        if not path.exists():
            return {"error": f"image not found: {image_path!r}"}
        try:
            gray = _load_image_gray(path)
        except Exception as exc:
            return {"error": f"failed to load image: {exc}"}

        ts_ns = (
            int(timestamp_us) * 1000 if timestamp_us
            else int(__import__("time").time() * 1e9)
        )
        try:
            pose = client.send_frame(ts_ns=ts_ns, gray=gray)
        except KimeraLiveError as exc:
            return {"error": str(exc)}
        state["frames_sent"] += 1

        if not pose.have_pose:
            return {
                "state":         pose.state,
                "translation_m": None,
                "quaternion":    None,
                "frames_sent":   state["frames_sent"],
                "ts_ns":         pose.ts_ns,
            }
        return {
            "state":         pose.state,
            "translation_m": list(pose.translation),
            "quaternion":    list(pose.quaternion),
            "frames_sent":   state["frames_sent"],
            "ts_ns":         pose.ts_ns,
        }

    @mcp.tool()
    def push_imu(samples: list[list[float]]) -> dict:
        """Forward IMU samples to the live pipeline.

        Each sample is a 7-element list: [ts_ms, gx, gy, gz, ax, ay, az].
        Gyro in rad/s, accel in m/s².  Timestamps in milliseconds (the
        worker converts the web client's batched JSON to this form).
        """
        out: list[tuple[int, tuple[float, ...]]] = []
        for s in samples:
            if len(s) != 7:
                continue
            ts_ns = int(float(s[0]) * 1_000_000.0)
            out.append((ts_ns, (s[1], s[2], s[3], s[4], s[5], s[6])))
        try:
            n = client.push_imu(out)
        except KimeraLiveError as exc:
            return {"error": str(exc)}
        return {"appended": n}

    @mcp.tool()
    def set_camera_intrinsics(width: int, height: int,
                              fx: float, fy: float,
                              cx: float, cy: float) -> dict:
        """Restart the live pipeline with these intrinsics (zero
        distortion — phone cameras after JS-level cropping are close
        enough to pinhole for VIO bootstrap)."""
        try:
            client.set_intrinsics(width=int(width), height=int(height),
                                  fx=float(fx), fy=float(fy),
                                  cx=float(cx), cy=float(cy))
        except KimeraLiveError as exc:
            return {"error": str(exc)}
        state["intrinsics"] = dict(width=int(width), height=int(height),
                                   fx=float(fx), fy=float(fy),
                                   cx=float(cx), cy=float(cy))
        state["frames_sent"] = 0
        return {"ok": True, "intrinsics": state["intrinsics"]}

    @mcp.tool()
    def get_map_stats() -> dict:
        return {
            "frames_sent":   state["frames_sent"],
            "intrinsics":    state["intrinsics"],
            "socket_path":   str(client._sock_path),     # noqa: SLF001
        }

    @mcp.tool()
    def reset_map() -> dict:
        try:
            client.reset()
        except KimeraLiveError as exc:
            return {"error": str(exc)}
        state["frames_sent"] = 0
        logger.warning("kimera-mcp: live pipeline reset")
        return {"ok": True}

    return mcp


def build_app(client: KimeraLiveClient, cfg: dict):
    return build_mcp(client, cfg).http_app(path="/mcp")


async def _serve(cfg: dict, ready_file: pathlib.Path | None) -> None:
    sock_dir = pathlib.Path(
        cfg.get("sock_dir", "/tmp/xr-ai/kimera-sock"),
    ).expanduser()
    host = cfg.get("host", "0.0.0.0")
    port = int(cfg.get("port", 8250))

    client = KimeraLiveClient(
        sock_dir              =sock_dir,
        docker_image          =cfg.get("docker_image", "kimera_vio"),
        container_name        =cfg.get("container_name", "xr_ai_kimera_live"),
        params_folder_in_image=cfg.get("params_folder_in_image",
                                       "/opt/kimera-params/EurocMonoLive"),
        startup_timeout_s     =float(cfg.get("startup_timeout_s", 30.0)),
    )

    # If a default intrinsics block is present, push it eagerly so the
    # pipeline is up before the first frame.  The worker overrides this
    # once the web client posts its camera_meta.
    default_intr = None
    if "default_fx" in cfg:
        default_intr = dict(
            width =int(cfg.get("default_width",  640)),
            height=int(cfg.get("default_height", 480)),
            fx    =float(cfg["default_fx"]),
            fy    =float(cfg.get("default_fy", cfg["default_fx"])),
            cx    =float(cfg.get("default_cx", cfg.get("default_width",  640) / 2)),
            cy    =float(cfg.get("default_cy", cfg.get("default_height", 480) / 2)),
        )

    logger.info("kimera-mcp-server  sock_dir={}  port={}", sock_dir, port)
    client.start()
    # Belt-and-braces shutdown: uvicorn's signal handlers can leave the
    # container alive if the process exits before our finally: runs.
    # Register atexit + explicit SIGINT/SIGTERM so the docker container
    # never outlives this MCP server.
    atexit.register(client.stop)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, client.stop)
        except (NotImplementedError, RuntimeError):
            pass  # not available on this platform

    if default_intr is not None:
        try:
            client.set_intrinsics(**default_intr)
            logger.info("[kimera-mcp] applied default intrinsics: {}", default_intr)
        except KimeraLiveError as exc:
            logger.warning("[kimera-mcp] default intrinsics rejected: {}", exc)

    app    = build_app(client, cfg)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    if ready_file:
        ready_file.touch()
    try:
        await server.serve()
    finally:
        client.stop()


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
