<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI functions for NeMo Agent Toolkit

`xr-ai-nat` provides typed, in-process XR functions for NVIDIA NeMo Agent
Toolkit (NAT). Applications compose these functions directly; process-backed
or MCP compatibility adapters remain separate boundaries.

## Spatial math

The `xr_spatial_math` function group contains deterministic coordinate
operations. Callers supply an explicit `SpatialFrame`, so the functions do not
depend on OpenXR, a tracking service, or an MCP server.

```yaml
functions:
  spatial_math:
    _type: xr_spatial_math
```

The group exposes:

- `spatial_math__compute_gaze_target(user_frame, distance_meters)`
- `spatial_math__compute_user_relative_position(user_frame, direction_from_user, distance_meters)`
- `spatial_math__compute_position_relative_to_anchor(user_frame, anchor_position, relation_to_anchor, distance_meters)`
- `spatial_math__offset_position_in_user_frame(user_frame, start_position, forward_meters, right_meters, up_meters)`
- `spatial_math__compute_position_toward_or_away_from_reference(start_position, reference_position, movement_direction, distance_meters)`
- `spatial_math__compute_midpoint(first_position, second_position)`

Every operation returns a `Vector3` and only calculates coordinates. Creating,
moving, or associating a scene object remains the caller's responsibility.

Install the package in the NAT environment so NAT discovers the spatial-math
registration directly through its capability-specific `nat.plugins` entry point.

## Text memory

The `xr_text_memory` function group owns persistent, per-source JSONL text
history:

```yaml
functions:
  text_memory:
    _type: xr_text_memory
    directory: /tmp/xr-text-memory
```

It exposes `add_transcript`, `query_transcripts`, `list_sources`, and
`get_transcript_stats` as native functions. Source identifiers are preserved in
sidecar files even when their filesystem names require sanitization.

## Vision

Install `xr-ai-nat[vision]` to use the `xr_vision` function group. The group
accepts an injected `xr-ai-models` `VLMService` and exposes `ask_image` for a
local PNG or JPEG path:

```python
config = VisionFunctionsConfig(vlm=vlm, system_prompt="Answer briefly.")
await builder.add_function_group("vision", config)
```

Image acquisition is intentionally separate. A live-frame or video-memory
function obtains the image first, then passes its exact returned path to
`vision__ask_image`; callers must never invent or guess a path. The vision
function performs image I/O off the event loop, normalizes the input to JPEG,
and makes the model request through `xr-ai-models`.

For live voice workflows, `StreamingVisionConfig` accepts a hub
`ProcessorEndpoint` and exposes one native function with complete and streaming
invocation modes. It owns fresh-frame acquisition and VLM invocation; Pipecat
continues to own audio framing, interruption, and TTS.

## XR tracking

Install `xr-ai-nat[services]` and configure `xr_tracking` with the private
endpoint of `services/openxr-service`:

```yaml
functions:
  tracking:
    _type: xr_tracking
    endpoint: tcp://127.0.0.1:8330
```

`tracking__get_user_frame` returns one typed `SpatialFrame` containing the
user's current origin and basis vectors. Pass that value directly to the
spatial-math functions; tracking owns pose acquisition while spatial math
remains deterministic and service-independent.

## Video memory

Install `xr-ai-nat[services]` and configure `xr_video_memory` with the private
endpoint of `services/video-memory-service`:

```yaml
functions:
  video_memory:
    _type: xr_video_memory
    endpoint: tcp://127.0.0.1:8310
```

The group exposes recorded-participant discovery, recording statistics, H.264
clip queries, and timestamp-anchored PNG frame lookup. All `*_us` fields are
Unix-epoch microseconds: use `get_video_stats` to find the available absolute
range, use absolute `start_us`/`end_us` for a clip, and pass the workflow's
event timestamp as `reference_time_us` for a frame. `second_ago` is
intentionally a whole-second offset before that event; the returned
`timestamp_us` reports the exact frame selected. Current live frames are a
separate hub capability, not part of `xr_video_memory`.

## MCP compatibility

Install `xr-ai-nat[mcp]` and pass an explicit list of native functions to
`xr_ai_nat.adapters.mcp.create_mcp_server` when an application must serve
MCP-only agents. The adapter publishes one MCP tool per selected function and
supports aliases for compatibility names; MCP is not used for in-process NAT
composition.
