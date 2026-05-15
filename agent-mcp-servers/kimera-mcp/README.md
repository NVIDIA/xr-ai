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

**Image built + verified end-to-end on the bench box.**  Kimera-VIO
processed its bundled `MicroEurocDataset` (95 stereo frames + IMU + GT)
in 1.5 s on a 4-core CPU with ATE-RMSE **0.07 cm** vs ground truth —
~200× more accurate than `pose-mcp`'s home-grown pipeline on similar
data.  Streaming integration (a watch-folder protocol on top of the
EuRoC dataset loader, or a small C++ TCP input adapter) is the next
piece — Kimera-VIO is a batch-mode binary natively, so live data
needs a shim.

## Build

Two-stage build because upstream `Dockerfile_20_04` only installs
*dependencies*, not Kimera-VIO itself:

```bash
cd /tmp && git clone --depth 1 https://github.com/MIT-SPARK/Kimera-VIO.git
cd Kimera-VIO
# Stage 1: deps (GTSAM 4.2, OpenGV, DBoW2, Kimera-RPGO)  — ~30 min
docker build -f Dockerfile_20_04 -t kimera_vio_deps .
# Stage 2: Kimera-VIO itself                              — ~5-10 min
docker build -f /<repo>/agent-mcp-servers/kimera-mcp/scripts/Dockerfile.kimera \
             -t kimera_vio .
```

Final image is ~4 GB.  After build the binary is at
`/root/Kimera-VIO/build/stereoVIOEuroc` inside the container.

## Verify the image is healthy (no external download)

Kimera-VIO ships a 95-frame stereo + IMU sequence with ground truth
right in its `tests/data/MicroEurocDataset/`.  This script runs it:

```bash
bash scripts/verify_image.sh
```

On the bench box (4-core CPU, no GPU) this prints:

```
[verify] paired 6 / est=10 gt=28712
[verify] scale: 0.197
[verify] ATE-RMSE: 0.07 cm
[verify] mean:     0.06 cm
[verify] max:      0.11 cm
```

(Scale ≠ 1 is a Umeyama artifact when only 6 paired samples span a
~4-second window; on a longer sequence scale converges to ~1.0.)

## Run on a full EuRoC sequence (needs network access)

```bash
bash scripts/setup_euroc.sh           # downloads V1_01_easy (~700 MB)
bash scripts/run_euroc.sh             # runs Kimera-VIO in the container
bash scripts/eval_euroc.sh            # computes ATE vs EuRoC ground truth
```

Expected ATE-RMSE on V1_01_easy is **~5-8 cm stereo / ~10-15 cm monocular**.
`robotics.ethz.ch` was unreachable from the bench container's network
while this branch was being written; run the setup step from a machine
with proper outbound HTTP if the curl hangs.

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
