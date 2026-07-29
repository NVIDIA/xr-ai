<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI functions for NeMo Agent Toolkit

`xr-ai-nat` provides typed, in-process XR functions for NVIDIA NeMo Agent
Toolkit (NAT). Applications compose these functions directly; process-backed
or MCP compatibility adapters remain separate boundaries.

## Shared value models and the service boundary

Capabilities draw their shared value types from one capability-neutral module,
`xr_ai_nat.functions.types`: `Vector3`, `SpatialFrame`, and `Color`, all built
on the exported `ServiceResult` base (later capability result models are
expected to subclass `ServiceResult`; it renders as JSON via `__str__` and
retains unknown fields with `extra="allow"`).

Capabilities that talk to an out-of-process service share one private transport,
`xr_ai_nat.functions._service.rpc` (a correlated msgpack/ZMQ `RPCClient` /
`RPCServer`). `_service` owns only the transport; the value models above live in
`functions.types`, not in `_service`.

## Model-backed agents

Install `xr-ai-nat[agents]` to make an `xr-ai-models` `LLMService` available
to NAT's built-in LangChain-backed agent types without bypassing the repository
model-service boundary:

```python
from nat.plugin_api import LLMRef
from xr_ai_nat.llm import ModelsLLMConfig

llm_ref = LLMRef("scene_llm")
await builder.add_llm(llm_ref, ModelsLLMConfig(service=llm))
```

`ModelsLLMConfig` accepts either an already-created `LLMService` for
programmatic applications or a deployment-profile path and role. NAT owns the
agent graph, function registry, schemas, and tracing; the adapter translates
messages and tools while every model request still goes through
`xr-ai-models`.

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

Install `xr-ai-nat[vision]` to use the `xr_vision_tools` function group. The
group accepts a hub frame `endpoint`, an injected `xr-ai-models` `VLMService`,
and a reference to the `video_memory` group for recorded lookups:

```python
config = VisionToolsConfig(
    endpoint=frame_endpoint,
    vlm=vlm,
    video_memory=FunctionGroupRef("video_memory"),
)
await builder.add_function_group("vision", config)
```

It exposes two native tools over the always-on live-frame source. Frame
acquisition happens inside the tools, so callers pass a participant and a
question — never an image path:

- `look_at_current_frame(participant_id, question)` — inspect the participant's
  present live camera frame.
- `look_at_past_frame(participant_id, question, second_ago, reference_time_us)` —
  inspect a recorded frame `second_ago` whole seconds before the reference time,
  resolved lazily through the `video_memory` group.

Each tool performs image I/O off the event loop, normalizes the frame to JPEG,
and makes the model request through `xr-ai-models`.

For live voice workflows, `StreamingVisionConfig` (`xr_streaming_vision`) accepts
a hub `ProcessorEndpoint` and exposes one native function with complete and
streaming invocation modes. It owns fresh-frame acquisition and VLM invocation;
Pipecat continues to own audio framing, interruption, and TTS.

Its complete invocation returns a `VisionResult` with `status` set to `ok` or
`unavailable`; callers must handle an unavailable result without treating its
text as an answer about the scene.

MCP-only agents that already hold a local image path can still reach the legacy
file-path `ask_image` tool through the vlm-mcp compatibility server
(`agent-mcp-servers/vlm-mcp/`), which now owns that path-based surface directly.

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
`xr_ai_nat.mcp.create_mcp_server` when an application must serve
MCP-only agents. The adapter publishes one MCP tool per selected function and
supports aliases for compatibility names; MCP is not used for in-process NAT
composition.

`xr_ai_nat.mcp.create_mcp_server` is the canonical import. The former
`xr_ai_nat.adapters.mcp.create_mcp_server` path still works as a deprecated
forwarding alias (it emits a `DeprecationWarning` on import) and will be removed
in a future version; update callers to import from `xr_ai_nat.mcp`.
