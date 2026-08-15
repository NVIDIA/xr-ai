<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Video memory service

`video-memory-service` is a private typed RPC service for recorded XR camera
history. XR Media Hub writes the H.264 chunks; this process reads those chunks,
extracts PNG frames with NVDEC, and writes requested clips or frames to
`out_dir`. It does not subscribe to live hub frames.

Applications construct `xr_ai_tools.video_memory.VideoMemoryTools` with the
private service endpoint. The native tool group
contains discovery operations plus two selection groups:

- `list_recorded_participants` returns exact participant identities.
- `get_video_stats` returns the available Unix-epoch microsecond range.
- Latest selection ends at the newest recorded timestamp:
  - `get_latest_video(participant_id, duration_seconds)` writes the latest H.264
    window.
  - `get_latest_frames(participant_id, duration_seconds, ...)` returns
    evenly distributed timestamped PNG frames from the same window.
- Historical selection begins at one absolute `start_us`:
  - `get_historical_frame(participant_id, start_us)` returns the nearest
    recorded frame.
  - `get_historical_video(participant_id, start_us, duration_seconds)` writes
    the H.264 window beginning there.
  - `get_historical_frames(participant_id, start_us, duration_seconds, ...)`
    samples that same window.

All video windows are bounded to 300 seconds. Sampling adds a hard total
`frame_budget` cap of 256; sparse recordings and unavailable or corrupt chunks
may return fewer frames. The budget controls selection and PNG export, not VLM
inference: the shipped Cosmos configuration accepts at most four images per
prompt. Native-resolution PNGs and large budgets can produce substantial disk,
decode, and RPC work, so interactive callers should use a modest budget and set
paired `max_width` and `max_height` values. Those values fit each PNG within the
box while preserving aspect ratio and never upscaling. Each sampled or
historical frame exposes its exported PNG only through the shared
`TimedImage.image` reference accepted by `query_video`, after the caller selects
no more than the VLM's image limit.

Every `*_us` field is a Unix-epoch timestamp in microseconds. Exported-frame
timestamps are estimates linearly interpolated from each chunk's start, end,
and declared frame count because the recorder does not persist per-frame
presentation timestamps. A current camera frame is not recorded history; obtain it through
`xr_ai_tools.current_frame.CurrentFrameTool` in the process that owns the hub
connection. Together, `get_current_frame`, `get_latest_video`, and
`get_latest_frames` form the latest-media surface; the three
`*_historical_*` tools use the shared `start_us` convention.

```yaml
endpoint: tcp://0.0.0.0:8310
recordings_dir: /dev/shm/xr-ai/recordings
out_dir: /tmp/xr_video_queries
gpu_id: 0
```

`recordings_dir` must match XR Media Hub's `video_recording.out_dir`. Omit it
only when running the service for a compatibility health check.
`list_recorded_participants` then returns an empty list; the remaining recorded
operations return `recording_disabled`.
