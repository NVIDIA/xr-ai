<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Video memory tools (agent-sdk/xr-ai-tools/xr_ai_tools/video_memory.py, rpc.py)

## Role in a turn
Lookup of recorded camera history ("what was on the table a minute ago"). Tools return frames or video file paths; they sit on the critical path before a VLM call (frames then fed to `query_images` / `query_video`). All heavy lifting (decode H.264, interpolate, resize, PNG-encode) is in the remote video-memory service; this client is a thin RPC wrapper.

## Critical-path calls
1. Tool validation + Relay wrapper.
2. One `RPCClient.call(op, request.model_dump())` per tool (`video_memory.py:284-335`), ZMQ DEALER round trip with timeout (`rpc.py:75-132`). Serial.
3. Pydantic `model_validate` of the result: `SampleFramesResult` contains up to `frame_budget` (<=256) frames, each `TimedImage` holding an `ImageReference` URI (not pixel bytes) + width/height/timestamp, so result parsing is cheap; the frame bytes live where the URI points (server-registered or file path).
4. Typical agent chain (serial): `get_video_stats` (to learn time range) -> `get_historical_frames`/`get_latest_frames` -> `query_images` (VLM). Each hop is a separate tool-loop iteration if the LLM drives it (each hop = one LLM call, see tools-tool-calling-loop.md).

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| RPC timeout | 30.0 s | `video_memory.py:217` |
| health timeout | 2.0 s | `:341` |
| `duration_seconds` | 1..300 | `:58-62` |
| `frame_budget` | 1..256 (required, no default) | `:97-102` |
| `max_width`/`max_height` | None (no resize unless both given) | `:104-122` |
| Frames per VLM tool call | 4 max downstream (`vision.py:36`) | tools-vision.md |
| `RecordedVideoResult` | H.264 file path + size (not bytes) | `:80-93` |
| Historical frame format | "recorded PNG frame" (tool description) | `:249` |
| Retries / polling | none | - |

## Cost drivers
`duration_seconds` and `frame_budget` (server decode/seek work and PNG encoding of N frames), `max_width/height` (resizing cost vs transfer/VLM savings), number of chunks in the window (`num_chunks`, `total_bytes` from stats), number of serial tool hops chosen by the LLM.

## Estimate formula
`t_vm = t_rpc_rtt + t_server(seek + decode(duration) + frame_extract(frame_budget) + resize + PNG_encode(frame_budget, px)) + t_validate(frames * ~20 us)`. The server term is the unknown: needs measurement. Downstream: PNG frames at native res are 2-6 MB each; if they flow to a VLM unresized the base64 upload and image tokens dominate (see VLM card).

## Hidden latency / improvement candidates
* Always pass `max_width/max_height` (e.g. 768x432): saves server PNG-encode time, file reads, base64 bytes and VLM prefill tokens; typical saving hundreds of ms per frame set (guess; needs measurement).
* `frame_budget` is not defaulted: LLM may request 256 frames; the VLM path accepts only 4 frames per call (`vision.py:68`), so >4 sampled frames cannot be queried in one request: cap `frame_budget` at 4 to avoid wasted extraction.
* Serial stats -> frames -> VLM chain: each is an LLM iteration (~0.5-2 s each); exposing `get_latest_frames` directly (no stats first) saves one iteration. Time-window ambiguity is the reason for stats; supplying time range in the prompt context avoids the hop.
* `get_historical_frame` returns one PNG per call: multiple calls serial per frame (N+1 pattern) if LLM walks timestamps; prefer `get_historical_frames`.
* Video export returns a path to an H.264 file (`RecordedVideoResult.path`): consumer then has to read and base64 it for `ask_video` (full file in JSON).

## What to measure live
`VideoMemoryTools(endpoint).get_latest_frames.execute(LatestFramesRequest(participant_id=..., duration_seconds=10, frame_budget=4, max_width=640, max_height=360))` vs without resize; duration 5/30/300 s; `get_video_stats` first. Compare tool wall time to raw `await RPCClient.call` to separate wrapper overhead.

## Confidence
Client path: code-derived. Server decode/encode: not read, needs measurement.
