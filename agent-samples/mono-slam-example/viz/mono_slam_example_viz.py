# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
mono-slam-example visualizer — entry point.

Launched as a subprocess by ``uv run mono_slam_example`` (the orchestrator).
Do not run this directly.

Subscribes to pose updates published by the worker on the hub under
participant ``_mono_slam_viz`` / topic ``mono_slam.pose``, then renders a
real-time 3-D trajectory plot with a camera orientation triad.

Config (mono_slam_example_viz.yaml — auto-passed by the launcher)
-----------------------------------------------------------------
    hub_pub:            ipc:///tmp/xr_hub_pub   hub PUB socket address
    hub_push:           ipc:///tmp/xr_hub_in    hub PUSH socket address
    backend:            auto    auto|qt|save
    save_frames_dir:    null    write PNGs here instead of opening a window
    history_window_s:   5.0     fading trail length in seconds (0 = full)
    axis_length_m:      0.1     triad arm length in world units
    fps_target:         20      max plot update rate (Hz)

Backend selection (backend: auto):
    Priority order:
    1. If save_frames_dir is set → Agg + that directory (PNG capture wins
       over any display).
    2. Else if $DISPLAY / $WAYLAND_DISPLAY / $MIR_SOCKET is set → TkAgg
       (falls back to Qt5Agg if tkinter is unavailable).
    3. Else → Agg + /tmp/mono_slam_frames (fully headless).
    backend: qt forces Qt5Agg; backend: save forces Agg+save regardless of
    $DISPLAY.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import signal
import struct
import time

import msgpack
import numpy as np
import yaml
from loguru import logger
from xr_ai_agent import DataMessage, ProcessorEndpoint, Subscribe
from xr_ai_logging import setup_logging

from plot3d import auto_scale_axes, build_figure, update_plot

# Synthetic participant and topic the worker publishes pose updates under.
_VIZ_PARTICIPANT = "_mono_slam_viz"
_VIZ_TOPIC       = "mono_slam.pose"

_HUB_PUB_DEFAULT  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH_DEFAULT = "ipc:///tmp/xr_hub_in"


# ── payload helpers ────────────────────────────────────────────────────────────

def decode_pose(data: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode a pose payload produced by ``encode_pose``.

    Returns (t_xyz, R_3x3) or None on malformed input.

    Wire format: msgpack list [t_xyz: list[float, float, float],
                                R_flat: list[float * 9]]
    """
    try:
        fields = msgpack.unpackb(data, raw=False)
        t = np.array(fields[0], dtype=np.float64)
        R = np.array(fields[1], dtype=np.float64).reshape(3, 3)
        if t.shape != (3,) or R.shape != (3, 3):
            return None
        if not (np.isfinite(t).all() and np.isfinite(R).all()):
            return None
        return t, R
    except Exception:
        return None


# ── visualizer main loop ───────────────────────────────────────────────────────

class MonoSlamVizProcess:
    """Receives pose DataMessages from the hub and refreshes a 3-D plot."""

    def __init__(
        self,
        ep: ProcessorEndpoint,
        *,
        save_frames_dir: pathlib.Path | None,
        history_window_s: float,
        axis_length: float,
        fps_target: int,
        backend_name: str,
    ) -> None:
        self._ep               = ep
        self._save_frames_dir  = save_frames_dir
        self._history_window_s = history_window_s
        self._axis_length      = axis_length
        self._frame_interval   = 1.0 / max(1, fps_target)
        self._backend_name     = backend_name

        self._queue: asyncio.Queue[tuple[np.ndarray, np.ndarray]] = asyncio.Queue(maxsize=20)
        self._trajectory: list[np.ndarray] = []
        self._R_world    = np.eye(3)
        self._frame_idx  = 0
        self._running    = False

        self._ep.on_data(self._on_data)

    # ── IPC ────────────────────────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        if msg.participant_id != _VIZ_PARTICIPANT or msg.topic != _VIZ_TOPIC:
            return
        result = decode_pose(msg.data)
        if result is None:
            logger.warning("viz: malformed pose payload — dropped")
            return
        # Drop frames if the plot thread is backlogged — viz must never block worker.
        try:
            self._queue.put_nowait(result)
        except asyncio.QueueFull:
            pass

    # ── asyncio task: drain queue ──────────────────────────────────────────────

    async def _drain_queue(self) -> None:
        """Move pose updates from the IPC queue into the trajectory buffer."""
        while self._running:
            try:
                t, R = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            self._trajectory.append(t)
            self._R_world = R

    # ── render ─────────────────────────────────────────────────────────────────

    def _render_frame(self, fig, ax) -> None:
        if not self._trajectory:
            return

        fps_str = str(int(1.0 / self._frame_interval))
        history_window = (
            int(self._history_window_s / self._frame_interval)
            if self._history_window_s > 0
            else 0
        )

        update_plot(
            ax,
            self._trajectory,
            self._R_world,
            history_window=history_window,
            axis_length=self._axis_length,
        )
        auto_scale_axes(ax, self._trajectory)
        fig.canvas.draw()

        if self._save_frames_dir is not None:
            path = self._save_frames_dir / f"frame_{self._frame_idx:06d}.png"
            fig.savefig(path, dpi=80)
            if self._frame_idx % 20 == 0:
                logger.info("viz: saved frame {} → {}", self._frame_idx, path)
            self._frame_idx += 1
        else:
            fig.canvas.flush_events()

    # ── main coroutine ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True

        import matplotlib
        matplotlib.use(self._backend_name)
        import matplotlib.pyplot as plt

        fig, ax = build_figure()
        drain_task = asyncio.create_task(self._drain_queue())

        loop = asyncio.get_running_loop()

        if self._save_frames_dir is not None:
            self._save_frames_dir.mkdir(parents=True, exist_ok=True)
            logger.info("viz: headless mode — writing PNGs to {}", self._save_frames_dir)
            plt.close(fig)   # don't try to display; re-create each render cycle
            try:
                while self._running:
                    next_tick = loop.time() + self._frame_interval
                    fig_h, ax_h = build_figure()
                    self._render_frame(fig_h, ax_h)
                    plt.close(fig_h)
                    wait = max(0.0, next_tick - loop.time())
                    await asyncio.sleep(wait)
            finally:
                drain_task.cancel()
                try:
                    await drain_task
                except asyncio.CancelledError:
                    pass
        else:
            # Interactive window: use FuncAnimation on the main thread while IPC
            # runs as an asyncio background task pumped by a recurring call_later.
            plt.ion()
            plt.show(block=False)

            last_render = 0.0
            try:
                while self._running:
                    now = time.monotonic()
                    if now - last_render >= self._frame_interval:
                        self._render_frame(fig, ax)
                        last_render = now
                    # Yield back to asyncio so the IPC loop (ep.run) can progress.
                    await asyncio.sleep(0.02)
            finally:
                drain_task.cancel()
                try:
                    await drain_task
                except asyncio.CancelledError:
                    pass
                plt.close("all")

    def shutdown(self) -> None:
        self._running = False
        self._ep.stop()
        self._ep.close()


# ── helpers ────────────────────────────────────────────────────────────────────

def _detect_backend(cfg_backend: str, save_dir: pathlib.Path | None) -> tuple[str, pathlib.Path | None]:
    """Return (matplotlib_backend_name, effective_save_dir).

    Priority for ``backend: auto``:
    1. If ``save_frames_dir`` is explicitly set → Agg + that directory
       (operator chose offline PNG capture; display availability is irrelevant).
    2. Else if a display environment variable is set → interactive TkAgg/Qt5Agg.
    3. Else headless → Agg + default tmp directory.
    """
    if cfg_backend == "save":
        return "Agg", save_dir or pathlib.Path("/tmp/mono_slam_frames")
    if cfg_backend == "qt":
        return "Qt5Agg", None
    # auto: honour explicit save_frames_dir before checking for a display.
    if save_dir is not None:
        return "Agg", save_dir
    has_display = bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or os.environ.get("MIR_SOCKET")
    )
    if has_display:
        try:
            import tkinter  # noqa: F401
            return "TkAgg", None
        except ImportError:
            pass
        try:
            import PyQt5  # noqa: F401
            return "Qt5Agg", None
        except ImportError:
            pass
        # No interactive toolkit available despite a display — fall back to headless.
        logger.warning(
            "viz: display found but neither tkinter nor PyQt5 is available; "
            "falling back to headless PNG mode"
        )
    # headless, no save_dir configured — write to a safe default.
    return "Agg", pathlib.Path("/tmp/mono_slam_frames")


# ── asyncio entry ──────────────────────────────────────────────────────────────

async def _main(cfg: dict, ready_file: pathlib.Path | None) -> None:
    setup_logging("viz", namespace="mono-slam-example")

    hub_pub  = cfg.get("hub_pub",  _HUB_PUB_DEFAULT)
    hub_push = cfg.get("hub_push", _HUB_PUSH_DEFAULT)

    cfg_backend    = str(cfg.get("backend", "auto")).lower()
    save_dir_raw   = cfg.get("save_frames_dir", None)
    save_dir       = pathlib.Path(save_dir_raw) if save_dir_raw else None
    history_window = float(cfg.get("history_window_s", 5.0))
    axis_length    = float(cfg.get("axis_length_m",    0.1))
    fps_target     = int(  cfg.get("fps_target",        20))

    backend_name, effective_save_dir = _detect_backend(cfg_backend, save_dir)

    ep = ProcessorEndpoint(
        sub_addr=hub_pub,
        push_addr=hub_push,
        auto_subscribe=False,      # viz doesn't process real participants
        filter=Subscribe.DATA,
    )
    # Subscribe to the synthetic participant the worker publishes under.
    ep.subscribe(_VIZ_PARTICIPANT, filter=Subscribe.DATA)

    viz = MonoSlamVizProcess(
        ep,
        save_frames_dir=effective_save_dir,
        history_window_s=history_window,
        axis_length=axis_length,
        fps_target=fps_target,
        backend_name=backend_name,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, viz.shutdown)

    # Signal ready immediately after IPC subscription — this is the point where
    # the viz is able to receive pose messages.  Doing it here (rather than
    # inside viz.run() after build_figure) means a slow or failing matplotlib
    # backend cannot prevent the launcher from advancing to the monitor phase.
    if ready_file:
        ready_file.touch()

    # Run the IPC receive loop alongside the render loop.
    ep_task = asyncio.create_task(ep.run())
    try:
        await viz.run()
    finally:
        viz.shutdown()
        ep_task.cancel()
        try:
            await ep_task
        except asyncio.CancelledError:
            pass
    logger.info("mono-slam-example viz stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(_main(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
