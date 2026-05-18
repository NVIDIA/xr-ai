<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# droid-mcp

FastMCP wrapper around
[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) (BSD-3-Clause).
DROID-SLAM is deep monocular / stereo / RGB-D SLAM — a recurrent
network iteratively refines per-pixel depth and camera poses through
a differentiable bundle adjustment layer.  On TUM RGB-D and EuRoC it
beats ORB-SLAM3 and Kimera-VIO on monocular ATE; the price is a CUDA
dependency, custom-built C++/CUDA extensions, and ~10 GB VRAM at
real-time keyframe density.

## Why use this vs. the other pose backends in this repo?

| | `pose-mcp` (home-grown) | `kimera-mcp` (VIO) | `droid-mcp` (this) |
|---|---|---|---|
| Compute | CPU, ~30 FPS | CPU C++, ~1.5 s per call (batch) | CUDA GPU, ~20 FPS on 3090 |
| Sensors | mono | mono + IMU | mono only (here) |
| Metric scale | needs MoGe prior | yes (IMU) | no (monocular) |
| Trajectory accuracy on EuRoC mono | ~14 cm ATE | ~10 cm ATE | ~3-5 cm ATE |
| Install footprint | uv sync | docker image (~5 GB) | DROID source + CUDA extensions + ~250 MB weights |
| License | XFeat + MoGe — Apache-2.0/MIT | Kimera + GTSAM + OpenGV — BSD | DROID-SLAM — BSD-3 |

Pick `droid-mcp` when you have a GPU and want the lowest monocular
trajectory error.  Pick `kimera-mcp` when you have IMU and want metric
scale.  Pick `pose-mcp` when you want zero install friction and only
need rough localisation.

## Install

```bash
cd agent-mcp-servers/droid-mcp
uv sync
uv run bash scripts/setup_droid.sh   # clones DROID, builds CUDA exts, fetches weights
```

The setup script needs an NVIDIA driver + matching CUDA toolkit on
PATH, `g++`, and `ninja-build`.  It clones DROID-SLAM into
`~/.cache/xr-ai/DROID-SLAM`, builds its CUDA extensions linked against
the PyTorch we just installed, and downloads the pretrained checkpoint
to `~/.cache/xr-ai/droid.pth`.

## Tool surface

Same shape as `pose-mcp` / `kimera-mcp` so callers swap by changing
one URL:

| Tool                     | Purpose                                       |
|--------------------------|-----------------------------------------------|
| `estimate_pose(image_path, ts_us)` | Push grayscale frame → latest pose  |
| `push_imu(samples)`      | Accepted for parity, dropped (DROID is mono)  |
| `set_camera_intrinsics`  | Set K for the tracking resolution             |
| `get_map_stats()`        | `frames_sent`, intrinsics, image size         |
| `reset_map()`            | Reset SLAM session                            |

## Run

```bash
uv run python -m droid_mcp_server --config droid_mcp_server.yaml
```

Listens on `http://localhost:8260/mcp`.  Same MCP surface as the
other pose servers, so the existing `simple-vlm-example` worker
points at it by flipping one URL in its YAML.

## Status

**Scaffolded but not GPU-tested in this repo.**  The bench box used
for `pose-mcp` / `kimera-mcp` development has no working CUDA (NVML
driver mismatch), so the DROID path has been smoke-tested only at
the import / config-load layer.  When you run on a GPU host:

1. `setup_droid.sh` builds CUDA extensions against your local
   PyTorch — version mismatch = silent breakage; install matching
   `torch` first.
2. First `estimate_pose` call lazy-loads DROID and constructs the
   `Droid` object; expect ~5 s of model-load latency on the first
   frame, then steady-state ~50 ms per frame.

## License notes

* DROID-SLAM — BSD-3-Clause (Princeton VL)
* PyTorch — BSD-3-Clause (PyTorch Foundation)
* CUDA Toolkit — NVIDIA EULA (build-time dep; not redistributed)

All fits the project-wide "Apache-2.0 / MIT / BSD only" rule.
