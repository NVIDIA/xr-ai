<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# slam-example

SLAM-only agent sample.  Brings up:

* `xr-media-hub`         — LiveKit ingress + web client at `http://<host>:8080`
* a SLAM MCP backend     — branch-specific (this branch wires `space-mcp`)
* `slam-example-worker`  — frames → SLAM → `place.update` (or
  `pose.update` on the metric branches)

Unlike `simple-vlm-example`, this sample has no VLM / STT / TTS — it
exists to exercise the SLAM path in isolation.  The web client is the
same one the VLM sample uses; participants connect over LiveKit and
receive data messages on the data channel.

## Branch ↔ backend matrix

| Branch                 | SLAM MCP backend wired in `main.py` | Status   | Out-going topic |
|------------------------|-------------------------------------|----------|-----------------|
| `feat/pose-mcp-fast`   | `pose-mcp`   (CPU, 30 FPS, no IMU)  | wired    | `pose.update`   |
| `feat/spatial-memory`  | `space-mcp`  (topological, DINOv2)  | wired    | `place.update`  |
| `feat/kimera-vio`      | `kimera-mcp` (CPU C++, IMU)         | TBD      | `pose.update`   |
| `feat/droid-mcp`       | `droid-mcp`  (GPU, monocular deep)  | wired    | `pose.update`   |

The three metric-SLAM backends (`pose-mcp` / `kimera-mcp` / `droid-mcp`)
share a tool surface (`estimate_pose`, `push_imu`,
`set_camera_intrinsics`, `get_map_stats`, `reset_map`), so the worker
on those branches is backend-agnostic.  `space-mcp` is different — it
exposes `process_frame`, `where_am_i`, `list_regions`,
`describe_region`, `label_region`, `find_object`, `remember_objects`,
`reset_map`.  No metric pose, no IMU, no intrinsics.  The worker on
this branch is therefore a thinner variant that emits the inferred
region info on `place.update` instead of a metric pose.

## Run

```bash
cd agent-samples/slam-example
uv sync
uv run slam_example
```

then open `http://<host>:8080` in a phone browser, allow camera, and
watch the orchestrator's log for `slam ... state=... region=...` lines.
The same payload is echoed back to the client on data topic
`place.update`.

## Optional: live Rerun viz

Enable `rerun_addr` in `yaml/slam_mcp_server.yaml`, then in another
terminal launch the viewer from the SLAM backend's venv:

```bash
cd agent-mcp-servers/space-mcp
uv run rerun
```

See the per-backend README under `agent-mcp-servers/*/README.md` for
details.
