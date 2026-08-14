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
contains five recorded-history operations:

- `list_recorded_participants` returns exact participant identities.
- `get_video_stats` returns the available Unix-epoch microsecond range.
- `get_recorded_video` writes a clip for an absolute Unix-epoch microsecond window.
- `sample_recorded_frames` returns evenly distributed timestamped PNG frames from the
  `duration_seconds` ending at `reference_time_us`. `frame_budget` is a hard
  total cap; sparse recordings may return fewer frames. Requests are bounded to
  300 seconds and 256 frames. Optional paired `max_width` and `max_height` values fit
  each PNG within that box while preserving aspect ratio and never upscaling.
  Each frame retains its `path` and implements the shared `TimedImage` contract,
  so the returned list can be passed directly to `query_video` inference.
- `get_frame_from_time` selects a frame at `reference_time_us - second_ago`
  seconds, where `reference_time_us` is the workflow's event timestamp, and
  returns the same path-compatible image reference.

Every `*_us` field is a Unix-epoch timestamp in microseconds. Keep the
model-facing offset coarse: use whole `second_ago` values for temporal
reasoning and use the returned `timestamp_us` to inspect the exact selected
frame. A current camera frame is not recorded history; obtain it through
`xr_ai_tools.current_frame.CurrentFrameTool` in the process that owns the hub
connection.

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
