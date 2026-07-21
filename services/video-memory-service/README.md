<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Video memory service

`video-memory-service` is a private typed RPC service for recorded XR camera
history. XR Media Hub writes the H.264 chunks; this process reads those chunks,
extracts PNG frames with NVDEC, and writes requested clips or frames to
`out_dir`. It does not subscribe to live hub frames.

Applications expose the service through
`xr_ai_nat.functions.video_memory.VideoMemoryFunctionsConfig`. The native group
contains four recorded-history operations:

- `list_recorded_participants` returns exact participant identities.
- `get_video_stats` returns the available Unix-epoch microsecond range.
- `query_video` writes a clip for an absolute Unix-epoch microsecond window.
- `get_frame_from_time` selects a frame at `reference_time_us - second_ago`
  seconds, where `reference_time_us` is the workflow's event timestamp.

Every `*_us` field is a Unix-epoch timestamp in microseconds. Keep the
model-facing offset coarse: use whole `second_ago` values for temporal
reasoning and use the returned `timestamp_us` to inspect the exact selected
frame. A current camera frame is not recorded history; obtain it through
`xr_ai_agent.LiveFrameSource` or, while it remains supported, Video MCP's live
compatibility tools.

```yaml
endpoint: tcp://0.0.0.0:8310
recordings_dir: /dev/shm/xr-ai/recordings
out_dir: /tmp/xr_video_queries
gpu_id: 0
```

`recordings_dir` must match XR Media Hub's `video_recording.out_dir`. Omit it
only when running the service for a compatibility health check; all recorded
operations then return `recording_disabled`.
