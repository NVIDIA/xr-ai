<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# kimera-mcp

FastMCP wrapper around [MIT-SPARK Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO)
(BSD-2-Clause).  Kimera is a production-grade visual-inertial SLAM
system in C++ — vastly more accurate and robust than the monocular
feature+PnP path that lives in `pose-mcp/`.

## How it works (streaming)

Kimera-VIO's bundled example binary (`stereoVIOEuroc`) is batch-mode —
it reads an EuRoC dataset folder off disk and exits.  We replace it
with a tiny long-running C++ wrapper (`scripts/kimera_live_vio.cpp`)
that uses Kimera-VIO as a library:

* On startup, listens on an AF_UNIX socket at `/sock/kimera-vio.sock`
  inside the container.
* Builds a `MonoImuPipeline` with the YAML params at
  `/opt/kimera-params/EurocMonoLive` and starts the pipeline threads.
* On `INTRINSICS`, rebuilds the pipeline with the operator-supplied
  fx/fy/cx/cy/distortion.
* On `FRAME`, pushes the grayscale buffer into the pipeline's
  `LeftFrameQueue` and replies with the latest pose captured from the
  backend output callback.
* On `IMU`, pushes samples into the pipeline's IMU queue.

The Python MCP server (`kimera_mcp_server/__main__.py`) spawns the
container at startup (`docker run -d --rm …`), bind-mounts a host
directory as `/sock`, and connects to the socket via
`socket.AF_UNIX`.  The container is shut down on MCP server exit
(atexit + SIGINT/SIGTERM handlers ensure cleanup even on rough
shutdown).

## Tool surface

Same shape as `pose-mcp`, so callers swap by changing one URL:

| Tool                     | Purpose                                         |
|--------------------------|-------------------------------------------------|
| `estimate_pose(image_path, ts_us)` | Stream frame → returns latest pose    |
| `push_imu(samples)`      | Stream IMU samples (list of 7-floats)           |
| `set_camera_intrinsics`  | Restart pipeline with new K                     |
| `get_map_stats()`        | `frames_sent`, current intrinsics               |
| `reset_map()`            | Tear down the pipeline state                    |

## Build

**You don't have to run this manually.**  On first start, kimera-mcp
checks for the `kimera_vio` docker image and, if it's missing, clones
Kimera-VIO into `~/.cache/xr-ai/Kimera-VIO` and runs both docker
builds itself.  Set `build_if_missing: false` in the YAML if you'd
rather fail fast in sandboxed environments.

If you do want to do it by hand (or pre-cache the image for CI):

```bash
cd /tmp && git clone --depth 1 https://github.com/MIT-SPARK/Kimera-VIO.git
cd Kimera-VIO
# Stage 1: deps (GTSAM 4.2, OpenGV, DBoW2, Kimera-RPGO)
docker build -f Dockerfile_20_04 -t kimera_vio_deps .
# Stage 2: Kimera-VIO + kimera_live_vio streaming binary
docker build -f /<repo>/agent-mcp-servers/kimera-mcp/scripts/Dockerfile.kimera \
             -t kimera_vio .
```

Final image is ~5 GB.  After build the streaming binary is at
`/usr/local/bin/kimera_live_vio` and the legacy batch binary is still
at `/root/Kimera-VIO/build/stereoVIOEuroc`.

## Visualisation

Same Rerun pattern as the other pose backends in this repo.  Set
`rerun_addr: "localhost:9876"` in the YAML, then in the same venv:

```bash
uv run rerun --connect rerun+http://localhost:9876/proxy
```

before starting the MCP server.  The Python side logs every pose that
comes back from the C++ binary:

* live camera frustum at `world/camera` (driven by `set_camera_intrinsics`)
* current grayscale frame at `world/camera/image`
* trajectory polyline at `world/trail`

Viewer failures are tolerated — a flaky / disconnected viewer demotes
the sink to a no-op rather than tanking the pose path.

## Status

Verified end-to-end on the bench box:

* Bundled `MicroEurocDataset` (95 frames + IMU): pipeline initialises
  in ~10 frames, returns realistic millimeter-scale translations and
  a stable quaternion as it processes the sequence.  Per-frame
  round-trip < 5 ms on a 4-core CPU once warmed up — no per-call
  docker startup overhead.
* On its own bundled stereo+GT sequence the legacy batch binary still
  achieves ATE-RMSE 0.07 cm; the streaming binary uses the identical
  pipeline internals so accuracy is unchanged.

## License notes

* Kimera-VIO and Kimera-RPGO — BSD-2-Clause (MIT SPARK)
* GTSAM — BSD-3-Clause (Georgia Tech)
* OpenGV — BSD-2-Clause (Laurent Kneip et al.)
* DBoW2 — modified BSD (Dorian Gálvez-López et al.)
* OpenCV — Apache-2.0
* Eigen — MPL2

All permissive; fits the project-wide "Apache-2.0 / MIT / BSD only"
rule.
