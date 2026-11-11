<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tracking, spatial math, RPC transport, capture (agent-sdk/xr-ai-tools/xr_ai_tools/tracking.py, spatial.py, rpc.py, capture.py, types.py)

## Role in a turn
`get_user_frame` fetches the user's head pose from the OpenXR service (small ZMQ RPC) so spatial helpers can compute placements; `spatial.py` is pure local math with no latency. `capture.py` start/stop are fire-and-forget hub messages. `rpc.py` is the shared transport under RAG, video-memory, tracking.

## Critical-path calls
1. `Tool.execute` (Relay wrapper).
2. `await rpc.call("get_head_pose", {})` `tracking.py:73-76` -> `rpc.py:75-132`: `uuid4().hex`, create future, `msgpack.packb`, `async with _send_lock: await socket.send` `:100-110`, `await wait_for(future, timeout)` `:111-115`; receiver task (`:134-155`) unpacks and resolves.
3. `HeadPose.model_validate` + `SpatialFrame` build `tracking.py:74-84`. Invalid pose -> `RuntimeError`.
4. `spatial.*` functions: synchronous math (`spatial.py`), microseconds.
5. Capture: `endpoint.send_return_data(DataMessage)` per start/stop `capture.py:157-163`; no ack awaited beyond the send.

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| Tracking RPC timeout | 10.0 s | `tracking.py:63`, `rpc.py:48` |
| Tracking health timeout | 2.0 s | `tracking.py:90` |
| RPC retries / reconnect backoff | none (ZMQ handles reconnect internally; a call to a missing server waits the full timeout) | `rpc.py:111-120` |
| ZMQ `LINGER` | 0 | `rpc.py:67` |
| Request frame | msgpack map; protocol v1 | `rpc.py:18, :104-109` |
| Server side: one `asyncio.Task` per request, replies under a lock | - | `rpc.py:204-209, :252-254` |
| Spatial rounding | 3 decimals | `spatial.py:15` |

## Cost drivers
RPC round trip (local IPC/TCP) and service handler time (OpenXR pose read); number of tool calls. Payload tiny (<1 KB).

## Estimate formula
`t_pose = t_pydantic(<0.1 ms) + t_relay_tool(measure) + t_zmq_rtt(0.1-0.5 ms localhost, guess) + t_service(OpenXR pose read, needs measurement) + t_validate(<0.1 ms)`. Best ~0.5 ms / typical 1-3 ms / worst 10 s timeout when the service is down (no fast-fail). Spatial math: ~10 us.

## Hidden latency / improvement candidates
* No fast-fail: a dead/absent tracking service stalls each call 10 s (`rpc.py:111-120`). A health pre-check or shorter timeout (e.g. 1-2 s) caps the worst case.
* Each tool hop costs one LLM iteration; `get_user_frame` then multiple spatial tools = several LLM calls. Spatial functions are plain Python helpers (not Tools) in this package, so composing them inside a single tool avoids extra LLM hops (saves ~0.5-2 s per avoided iteration).
* Pose is re-fetched per call (no short-TTL cache): fine for freshness.
* Sequential `_send_lock` around a ZMQ send serializes concurrent RPC senders, but sends are microseconds.

## What to measure live
`TrackingTools(endpoint).get_user_frame.execute(EmptyRequest())` p50/p99; with the service stopped (timeout behaviour); raw `RPCClient.call("get_health", {})` as baseline round trip.

## Confidence
Code-derived; service time needs measurement.
