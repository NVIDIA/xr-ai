<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# kimera-mcp

FastMCP wrapper around [MIT-SPARK Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO)
(BSD-2-Clause).  Kimera is a production-grade visual-inertial SLAM
system in C++ — vastly more accurate and robust than the monocular
feature+PnP path that lives in `pose-mcp/`.  This server runs Kimera
as a child process (typically inside Docker), feeds it live frames +
IMU in EuRoC dataset format, and surfaces the resulting trajectory
over MCP using the same tool surface (`estimate_pose`, `get_map_stats`,
`reset_map`).

## Status

**Scaffold + setup scripts only.**  Kimera-VIO is a batch-mode binary;
the C++ pipeline is designed to ingest an EuRoC-style folder up
front and emit a trajectory file when done.  Wiring it to a live
stream requires either a small C++ patch (a TCP/named-pipe input
adapter) or a watch-folder protocol on top of the existing dataset
loader.  This branch sets up the docker build + EuRoC demo so you can
verify Kimera works end-to-end on this machine, then iterate on the
streaming integration once the binary is known-good.

## Build

```bash
cd /tmp && git clone --depth 1 https://github.com/MIT-SPARK/Kimera-VIO.git
cd Kimera-VIO
docker build -f Dockerfile_20_04 -t kimera_vio .
```

~40-60 min on a 4-core CPU; ~10 GB final image.  Pulls GTSAM 4.2,
OpenGV, DBoW2, Kimera-RPGO from source.

## Verify on EuRoC

```bash
bash scripts/setup_euroc.sh           # downloads V1_01_easy (~700 MB)
bash scripts/run_euroc.sh             # runs Kimera-VIO in the container
bash scripts/eval_euroc.sh            # computes ATE vs EuRoC ground truth
```

The expected ATE-RMSE on V1_01_easy is ~5-8 cm in stereo mode and ~10-15 cm
in monocular mode.  If you see something close to that, Kimera is
healthy on this machine.

## Integration plan (next commit)

1. **Watch-folder protocol** — extend the EuRoC dataset loader so it
   tails `cam0/<ts>.png` + `imu0.csv` and dispatches new samples to the
   pipeline as they appear.  Upstream PR-worthy; for now a thin
   subprocess-level wrapper rewrites the folder before each batch.
2. **Worker** writes incoming camera frames to `cam0/<ts>.png`
   (atomic tmp+rename) and appends every IMU batch to `imu0.csv`.
3. **kimera-mcp** runs Kimera-VIO with `--initial_k=0 --final_k=-1`
   so it processes the entire current contents on each tick, then
   parses the latest output pose from `traj_vio.csv` and serves it.
4. **MCP surface** mirrors `pose-mcp` exactly (`estimate_pose`,
   `get_map_stats`, `reset_map`) so the existing
   `simple-vlm-example` worker doesn't have to change apart from
   `pose_mcp_url` → `kimera_mcp_url`.

## License notes

* Kimera-VIO and Kimera-RPGO — BSD-2-Clause (MIT SPARK)
* GTSAM — BSD-3-Clause (Georgia Tech)
* OpenGV — BSD-2-Clause (Laurent Kneip et al.)
* DBoW2 — modified BSD (Dorian Gálvez-López et al.)
* OpenCV — Apache-2.0
* Eigen — MPL2

All permissive; fits the project-wide "Apache-2.0 / MIT / BSD only"
rule.
