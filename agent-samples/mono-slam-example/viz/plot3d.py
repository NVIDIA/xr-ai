# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
3-D trajectory and orientation triad renderer for the mono-SLAM visualizer.

All functions are pure (no I/O, no global state) except ``build_figure``,
which creates and returns a matplotlib Figure + Axes3D. Stateful animation
is managed by the caller (``MonoSlamVizProcess``).

Coordinate frame (OpenCV camera, world frame)
---------------------------------------------
- X right, Y down, Z forward.
- ``t_world`` is the accumulated camera position vector expressed in the
  world (first-frame) frame: pos = -R_world.T @ t_world_cam.
- ``R_world`` encodes R_curr_from_world (transforms world point to current
  camera frame); the triad columns are the world-frame body axes of the
  camera, i.e. R_world.T @ I.

All positions are in monocular unit-norm distance (no metric scale).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection


def build_figure(title: str = "mono-SLAM trajectory") -> tuple[plt.Figure, plt.Axes]:
    """Create the Figure and 3-D Axes used by the animation loop.

    Returns a (figure, ax) pair.  Call this once at startup; pass both
    to ``update_plot`` on every animation tick.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (down)")
    ax.set_zlabel("Z (fwd)")
    return fig, ax


def update_plot(
    ax: plt.Axes,
    trajectory: list[np.ndarray],
    R_world: np.ndarray,
    *,
    history_window: int = 0,
    axis_length: float = 0.1,
    fading: bool = True,
) -> None:
    """Redraw the trajectory and orientation triad on *ax*.

    Args:
        ax:             Matplotlib 3-D axes — cleared and redrawn each call.
        trajectory:     List of position vectors (3,) in world frame,
                        accumulated camera positions.
        R_world:        Current R_curr_from_world (3×3).  Triad columns are
                        R_world.T columns = camera body axes in world frame.
        history_window: If > 0, only the most recent ``history_window`` points
                        are drawn with full opacity; older points are dimmer
                        (fading trail).  0 means draw the whole trajectory.
        axis_length:    Length of each triad arm in world-unit fractions.
        fading:         If True and history_window > 0, dim points outside
                        the window to 0.25 alpha.
    """
    ax.cla()
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (down)")
    ax.set_zlabel("Z (fwd)")

    if not trajectory:
        return

    pts = np.array(trajectory)           # (N, 3)

    if history_window > 0 and len(pts) > history_window:
        recent_start = len(pts) - history_window
        old_pts = pts[:recent_start]
        recent_pts = pts[recent_start:]
        if fading and len(old_pts) >= 2:
            ax.plot(old_pts[:, 0], old_pts[:, 1], old_pts[:, 2],
                    "-", color="steelblue", alpha=0.25, linewidth=1)
        ax.plot(recent_pts[:, 0], recent_pts[:, 1], recent_pts[:, 2],
                "-o", color="steelblue", alpha=0.9, linewidth=1.5, markersize=2)
    else:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                "-o", color="steelblue", alpha=0.9, linewidth=1.5, markersize=2)

    # Orientation triad at the latest position.
    pos = pts[-1]
    # R_world.T columns are the camera body axes expressed in the world frame.
    axes_world = R_world.T  # shape (3, 3); column i = i-th body axis in world

    colors = ("red", "green", "blue")
    labels = ("X_cam", "Y_cam", "Z_cam")
    for i, (col, lab) in enumerate(zip(colors, labels)):
        direction = axes_world[:, i] * axis_length
        ax.quiver(
            pos[0], pos[1], pos[2],
            direction[0], direction[1], direction[2],
            color=col, linewidth=2, label=lab,
        )

    ax.scatter(*pos, color="orange", s=40, zorder=5)   # current position marker
    ax.set_title(f"mono-SLAM  frames={len(trajectory)}", fontsize=9)


def auto_scale_axes(ax: plt.Axes, trajectory: list[np.ndarray]) -> None:
    """Set equal aspect ratio on all three axes to avoid shape distortion.

    Matplotlib's 3-D equal-aspect isn't automatic; this uses the bounding
    box of the trajectory with a small margin.
    """
    if len(trajectory) < 2:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return

    pts = np.array(trajectory)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    max_range = max(ranges.max(), 0.2)   # avoid zero-range on degenerate trajectories
    centres = (mins + maxs) / 2
    half = max_range / 2 * 1.2          # 20 % margin
    ax.set_xlim(centres[0] - half, centres[0] + half)
    ax.set_ylim(centres[1] - half, centres[1] + half)
    ax.set_zlim(centres[2] - half, centres[2] + half)
