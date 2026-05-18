# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage the EuRoC-format dataset folder that Kimera-VIO ingests.

Layout we maintain on disk::

    <dataset_dir>/
      mav0/
        body.yaml
        cam0/
          sensor.yaml
          data.csv             # rolling: timestamp_ns,filename
          data/<ts_ns>.png     # grayscale PNGs, oldest pruned past `window_frames`
        imu0/
          sensor.yaml
          data.csv             # rolling: timestamp_ns,gx,gy,gz,ax,ay,az

The writer is process-local (no locking across writers); kimera-mcp
owns the folder exclusively.  PNG writes are atomic (tmp+rename) so
Kimera never reads a half-written file.
"""
from __future__ import annotations

import dataclasses
import pathlib
import shutil
from collections import deque

import numpy as np
from PIL import Image


_BODY_YAML = """%YAML:1.0
# Body frame definition — minimal; we don't use IMU-camera extrinsics for
# this watch-folder integration because we generate both from the same
# device's perspective.
"""

_CAM_SENSOR_YAML = """%YAML:1.0
sensor_type: camera
comment: live-stream camera
T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 30
resolution: [{w}, {h}]
camera_model: pinhole
intrinsics: [{fx}, {fy}, {cx}, {cy}]
distortion_model: radial-tangential
distortion_coefficients: [0.0, 0.0, 0.0, 0.0]
"""

_IMU_SENSOR_YAML = """%YAML:1.0
sensor_type: imu
comment: live-stream IMU
T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 200
gyroscope_noise_density:    0.00016968
gyroscope_random_walk:      1.9393e-05
accelerometer_noise_density: 0.002
accelerometer_random_walk:  0.003
"""

# Kimera's EurocDataProvider scans the dataset for a state_groundtruth_estimate0
# subfolder and reads its sensor.yaml unconditionally — even when no GT is
# actually consumed at runtime.  A degenerate stub satisfies the load.
_GT_SENSOR_YAML = """%YAML:1.0
sensor_type: position
comment: stub — no live ground truth
T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
"""


@dataclasses.dataclass(frozen=True)
class CameraIntrinsics:
    width:  int
    height: int
    fx:     float
    fy:     float
    cx:     float
    cy:     float


class EurocDatasetWriter:
    """Append-only writer for the EuRoC dataset folder structure with a
    sliding-window prune."""

    def __init__(
        self,
        root: pathlib.Path,
        *,
        intrinsics:     CameraIntrinsics,
        window_frames:  int = 100,
    ) -> None:
        self._root           = pathlib.Path(root)
        self._intrinsics     = intrinsics
        self._window_frames  = max(2, int(window_frames))
        self._frame_log: deque[tuple[int, pathlib.Path]] = deque()
        self._init_layout()

    # ── layout / lifecycle ─────────────────────────────────────────────────

    @property
    def root(self) -> pathlib.Path:
        return self._root

    @property
    def mav0(self) -> pathlib.Path:
        return self._root / "mav0"

    def _init_layout(self) -> None:
        m = self.mav0
        (m / "cam0" / "data").mkdir(parents=True, exist_ok=True)
        # Kimera's EurocDataProvider unconditionally opens cam1/data.csv
        # even in mono (`frontend_type: 0`) mode, so we mirror cam0 into
        # cam1 — Kimera reads it but the mono frontend never consumes the
        # actual right-eye PNGs.  Cheap, ~150 KB per frame at 752x480.
        (m / "cam1" / "data").mkdir(parents=True, exist_ok=True)
        (m / "imu0").mkdir(parents=True, exist_ok=True)
        (m / "body.yaml").write_text(_BODY_YAML)
        sensor_yaml = _CAM_SENSOR_YAML.format(
            w=self._intrinsics.width, h=self._intrinsics.height,
            fx=self._intrinsics.fx, fy=self._intrinsics.fy,
            cx=self._intrinsics.cx, cy=self._intrinsics.cy,
        )
        (m / "cam0" / "sensor.yaml").write_text(sensor_yaml)
        (m / "cam1" / "sensor.yaml").write_text(sensor_yaml)
        (m / "imu0" / "sensor.yaml").write_text(_IMU_SENSOR_YAML)
        (m / "cam0" / "data.csv").write_text("#timestamp [ns],filename\n")
        (m / "cam1" / "data.csv").write_text("#timestamp [ns],filename\n")
        (m / "imu0" / "data.csv").write_text(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
            "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
            "a_RS_S_z [m s^-2]\n"
        )
        # Stub ground-truth folder + minimal data.csv (Kimera reads its
        # sensor.yaml unconditionally and tolerates an empty data.csv).
        gt = m / "state_groundtruth_estimate0"
        gt.mkdir(parents=True, exist_ok=True)
        (gt / "sensor.yaml").write_text(_GT_SENSOR_YAML)
        # Kimera's parseGtData asserts deltaCount > 0 and uses the GT to
        # build its initial nav-state.  We seed it with identity rows
        # spanning the live-stream timestamp range so the parse succeeds
        # and the initialization sees a static prior at the origin.
        gt_header = (
            "#timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z,"
            "v_RS_R_x,v_RS_R_y,v_RS_R_z,b_w_RS_S_x,b_w_RS_S_y,b_w_RS_S_z,"
            "b_a_RS_S_x,b_a_RS_S_y,b_a_RS_S_z\n"
        )
        (gt / "data.csv").write_text(gt_header)
        self._gt_path = gt / "data.csv"
        self._gt_header = gt_header

    def reset(self) -> None:
        if self.mav0.exists():
            shutil.rmtree(self.mav0)
        self._frame_log.clear()
        self._init_layout()

    # ── writers ────────────────────────────────────────────────────────────

    def _ensure_gt_covers(self, ts_ns: int) -> None:
        """Maintain two GT rows spanning the live-stream timestamp range
        so Kimera's parseGtData has the deltaCount > 0 it asserts.  The
        rows are identity poses; the pipeline never reads them for
        tracking (no use_gt_for_initialization in our PipelineParams)."""
        # Cheap: rewrite the file each frame with the current span.
        if not hasattr(self, "_gt_first_ts"):
            self._gt_first_ts = int(ts_ns)
        last = max(int(ts_ns), self._gt_first_ts + 1)
        identity = "1,0,0,0,0,0,0,0,0,0,0,0,0"   # qw=1, rest zero
        self._gt_path.write_text(
            self._gt_header
            + f"{self._gt_first_ts},0,0,0,{identity}\n"
            + f"{last},0,0,0,{identity}\n"
        )

    def append_frame(self, ts_ns: int, image_rgb: np.ndarray) -> pathlib.Path:
        """Write a grayscale PNG for the frame (Kimera's mono pipeline
        expects 8-bit gray).  Atomic via tmp+rename."""
        if image_rgb.ndim == 3:
            gray = (0.2989 * image_rgb[..., 0] + 0.5870 * image_rgb[..., 1]
                    + 0.1140 * image_rgb[..., 2]).astype(np.uint8)
        else:
            gray = image_rgb.astype(np.uint8)
        fname = f"{int(ts_ns):019d}.png"
        out   = self.mav0 / "cam0" / "data" / fname
        tmp   = out.with_suffix(".png.tmp")
        Image.fromarray(gray, "L").save(tmp, format="PNG")
        tmp.replace(out)
        # Mirror into cam1 — Kimera's data provider opens both even in
        # mono mode.  Hard-link instead of copy so we don't pay 2× disk.
        out_r = self.mav0 / "cam1" / "data" / fname
        try:
            out_r.hardlink_to(out)
        except (FileExistsError, OSError):
            try: out_r.unlink(missing_ok=True)
            except Exception: pass
            try: out_r.hardlink_to(out)
            except Exception: Image.fromarray(gray, "L").save(out_r, format="PNG")
        with (self.mav0 / "cam0" / "data.csv").open("a", encoding="utf-8") as f:
            f.write(f"{int(ts_ns)},{fname}\n")
        with (self.mav0 / "cam1" / "data.csv").open("a", encoding="utf-8") as f:
            f.write(f"{int(ts_ns)},{fname}\n")
        self._frame_log.append((int(ts_ns), out))
        self._ensure_gt_covers(int(ts_ns))
        self._prune_window()
        return out

    def append_imu(self, ts_ns: int, gyro: tuple[float, float, float],
                   accel: tuple[float, float, float]) -> None:
        with (self.mav0 / "imu0" / "data.csv").open("a", encoding="utf-8") as f:
            f.write(f"{int(ts_ns)},{gyro[0]},{gyro[1]},{gyro[2]},"
                    f"{accel[0]},{accel[1]},{accel[2]}\n")

    # ── window pruning ─────────────────────────────────────────────────────

    def _prune_window(self) -> None:
        # Keep at most `window_frames` PNGs on disk.  Old PNGs are removed
        # along with their entry in cam0/data.csv (which we rewrite —
        # Kimera reads the whole file at startup, so a stale entry
        # pointing at a deleted PNG would crash it).
        if len(self._frame_log) <= self._window_frames:
            return
        keep_n  = self._window_frames
        drop_n  = len(self._frame_log) - keep_n
        dropped = [self._frame_log.popleft() for _ in range(drop_n)]
        for _ts, path in dropped:
            try:
                path.unlink(missing_ok=True)
                (self.mav0 / "cam1" / "data" / path.name).unlink(missing_ok=True)
            except Exception:
                pass
        # Rewrite cam0/data.csv + cam1/data.csv with the surviving rows.
        rows = "#timestamp [ns],filename\n" + "".join(
            f"{ts},{path.name}\n" for ts, path in self._frame_log
        )
        (self.mav0 / "cam0" / "data.csv").write_text(rows)
        (self.mav0 / "cam1" / "data.csv").write_text(rows)

    @property
    def num_frames(self) -> int:
        return len(self._frame_log)

    @property
    def latest_ts_ns(self) -> int:
        return self._frame_log[-1][0] if self._frame_log else 0
