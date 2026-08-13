<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-hub

The minimal agent-side client for XR-Media-Hub. It depends only on pyzmq and
msgpack and deliberately contains no LiveKit, web-server, model, NAT, or voice
pipeline code.

```python
from xr_ai_hub import DataMessage, ProcessorEndpoint

endpoint = ProcessorEndpoint(
    sub_addr="ipc:///tmp/xr_hub_pub",
    push_addr="ipc:///tmp/xr_hub_in",
)

async def on_data(message: DataMessage) -> None:
    print(message.participant_id, message.topic)

unsubscribe = endpoint.on_data(on_data)
await endpoint.run()

unsubscribe()  # Remove the callback when its owner stops.
```

`LiveFrameSource` adds raw frame acquisition without adding image conversion or
model dependencies. Use `xr_ai_tools.streaming_vision.StreamingVisionTool` when the
desired interface is a model-facing async vision tool rather than raw pixels.

## Subscription and roster contract

Participants are the subscription unit. With `auto_subscribe=True` (the
default), the endpoint subscribes when a participant joins and unsubscribes on
leave. Set `filter` to a `Subscribe` flag combination to drop data, audio, or
video at the ZMQ socket; set `auto_subscribe=False` and call `subscribe(pid)`
for explicitly assigned participants.

`run()` requests a roster automatically for auto-subscribing endpoints so an
agent started mid-session learns participants who already joined. The hub
replays ordinary joined events, so `on_participant` callbacks must be
idempotent. `request_roster()` triggers the same replay explicitly.

ZMQ subscriptions take effect asynchronously. Await
`wait_for_subscriptions(timeout=...)` before advertising availability when a
caller manages readiness itself. It returns `False` when the hub cannot confirm
the subscription before the timeout.

## Frames

`on_frame` receives metadata without copying pixels. Call
`await endpoint.request_frame(signal)` to obtain `FrameData` only when needed;
concurrent requests for the same participant and track are coalesced.
`LiveFrameSource` provides a higher-level fresh-frame lookup and releases cached
participant state on departure.

## Return path

| Method | Effect |
|---|---|
| `send_return_data(message)` | Route text or binary data to `message.participant_id`. |
| `send_return_audio(chunk)` | Route PCM audio to `chunk.participant_id`. |
| `flush_return_audio(pid)` | Drop that participant's hub-queued return audio when interrupting playback. |
| `set_status(status, participant_id=None)` | Record this agent's status; omitting the participant broadcasts the default to every connected participant in scope. |
| `mark_ready()` | Set the endpoint's default status to `ready`, including for participants who join later. |
| `republish_statuses()` | Re-send recorded state after a client reconnect or missed one-shot update. |
| `request_roster()` | Ask the hub to replay joined events for the current roster. |

Return traffic is always participant-routed. The endpoint never publishes a
worker's availability directly to clients; the hub aggregates agent state.

## Readiness

Readiness participation is opt-in with `announces_readiness=True`. An endpoint
answers only for participants it subscribes to, and `set_status` is a logged
no-op for passive endpoints. The hub combines every responsible agent's state
and confirms a live subscription before exposing `ready` to the client.

`agent_id` identifies this endpoint's status. Pass it explicitly or set
`XR_AI_AGENT_ID`; otherwise the endpoint generates a process-local value.

## Shared memory and codec extensions

`ShmRingBuffer` and `SlotView` expose zero-copy frame views for callers that
read raw pixels. A slot remains owned until `release_slot()`. The msgpack codec
supports application payload types through `register_encoder` and
`register_decoder`; choose new type identifiers without changing existing
wire tags.
