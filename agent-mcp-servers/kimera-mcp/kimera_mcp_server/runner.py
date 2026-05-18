# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Kimera-VIO in its docker container on the staged dataset and
parse the resulting trajectory."""
from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import time

import numpy as np
from loguru import logger


@dataclasses.dataclass(frozen=True)
class KimeraPose:
    ts_ns:    int
    pose:     np.ndarray            # 4x4 SE(3), world ← camera
    quaternion: tuple[float, float, float, float]  # (w, x, y, z)


class KimeraRunner:
    def __init__(
        self, *,
        dataset_dir:             pathlib.Path,
        output_dir:              pathlib.Path,
        docker_image:            str  = "kimera_vio",
        params_folder_in_image:  str  = "/opt/kimera-params/EurocMonoLive",
    ) -> None:
        self._dataset = pathlib.Path(dataset_dir)
        self._output  = pathlib.Path(output_dir)
        self._output.mkdir(parents=True, exist_ok=True)
        self._image   = docker_image
        self._params  = params_folder_in_image

    def run_once(self, *, num_frames: int) -> KimeraPose | None:
        """Invoke Kimera-VIO on the current contents of ``dataset_dir``.
        Returns the latest pose from ``traj_vio.csv`` if Kimera produced
        one, else None."""
        # Wipe previous output so we don't accidentally read stale traj.
        traj = self._output / "traj_vio.csv"
        if traj.exists():
            traj.unlink()

        t0 = time.monotonic()
        try:
            subprocess.run([
                "docker", "run", "--rm",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{self._dataset}:/dataset:ro",
                "-v", f"{self._output}:/output:rw",
                self._image,
                "bash", "-lc",
                "cd /root/Kimera-VIO/build && ./stereoVIOEuroc "
                f"--dataset_type=0 "
                f"--dataset_path=/dataset "
                f"--params_folder_path={self._params} "
                f"--log_output=true "
                f"--output_path=/output "
                f"--initial_k=0 --final_k={num_frames} "
                f"--visualize=false --use_lcd=false "
                f"--vocabulary_path=/root/Kimera-VIO/vocabulary/ORBvoc.yml "
                f"--logtostderr=1 --colorlogtostderr=0 --minloglevel=2",
            ], check=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning("Kimera invocation timed out after 60 s")
            return None
        except subprocess.CalledProcessError as exc:
            logger.warning("Kimera invocation failed (rc={}): {}",
                           exc.returncode, exc.stderr.decode(errors="replace")[:500])
            return None

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if not traj.exists():
            logger.debug("Kimera wrote no traj_vio.csv (window may be too short)")
            return None
        latest = self._latest_pose(traj)
        if latest is not None:
            logger.info(
                "Kimera ok  ts={}  pos=({:+.2f},{:+.2f},{:+.2f})  ({:.0f} ms)",
                latest.ts_ns, *latest.pose[:3, 3], elapsed_ms,
            )
        return latest

    @staticmethod
    def _latest_pose(traj_path: pathlib.Path) -> KimeraPose | None:
        last = None
        for line in traj_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            last = line
        if last is None:
            return None
        parts = last.split(",")
        # Format: timestamp, x, y, z, qw, qx, qy, qz, vx,vy,vz, bg…, ba…
        ts_ns = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        qw, qx, qy, qz = (float(parts[4]), float(parts[5]),
                          float(parts[6]), float(parts[7]))
        n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
        if n > 1e-9:
            qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
        R = np.array([
            [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = (x, y, z)
        return KimeraPose(ts_ns=ts_ns, pose=T,
                          quaternion=(qw, qx, qy, qz))
