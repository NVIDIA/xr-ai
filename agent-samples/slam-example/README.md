<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# slam-example

SLAM-only agent sample.  Brings up:

* `xr-media-hub`         — LiveKit ingress + web client at `http://<host>:8080`
* a SLAM MCP backend     — branch-specific (this branch wires `kimera-mcp`)
* `slam-example-worker`  — frames + IMU + camera_meta → SLAM → `pose.update`

Unlike `simple-vlm-example`, this sample has no VLM / STT / TTS — it
exists to exercise the SLAM path in isolation.  The web client is the
same one the VLM sample uses; participants connect over LiveKit and
receive `pose.update` data messages on the data channel.

## Branch ↔ backend matrix

| Branch                 | SLAM MCP backend wired in `main.py` |
|------------------------|-------------------------------------|
| `feat/pose-mcp-fast`   | `pose-mcp`   (CPU, 30 FPS, no IMU)  |
| `feat/spatial-memory`  | `space-mcp`  (topological, DINOv2)  |
| `feat/kimera-vio`      | `kimera-mcp` (CPU C++, IMU)         |
| `feat/droid-mcp`       | `droid-mcp`  (GPU, monocular deep)  |

All four expose the same MCP tool surface (`estimate_pose`,
`push_imu`, `set_camera_intrinsics`, `get_map_stats`, `reset_map`)
so the worker is backend-agnostic.

## Run

```bash
cd agent-samples/slam-example
uv sync
uv run slam_example
```

then open `http://<host>:8080` in a phone browser, allow camera +
device motion, and watch the orchestrator's log for `slam ... state=ok`
lines.  The pose stream also goes back to the client on data topic
`pose.update`.

## Optional: live Rerun viz

Enable `rerun_addr` in `yaml/slam_mcp_server.yaml`, then in another
terminal launch the viewer from the SLAM backend's venv:

```bash
cd agent-mcp-servers/<backend>-mcp
uv run rerun
```

See the per-backend README under `agent-mcp-servers/*/README.md` for
details.
