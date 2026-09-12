<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-hub

`xr-ai-hub-client` is the minimal agent-side DeviceIOHub boundary. It depends
only on pyzmq and msgpack and contains no LiveKit, web-server, model, tool, or
voice-pipeline code. Refer to the generated {doc}`python/index` for public types
and signatures.

(subscription-and-roster-contract)=
## Processor endpoints

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
unsubscribe()
```

Participants are the subscription unit. With `auto_subscribe=True`, an endpoint
subscribes on join and unsubscribes on leave. Set its `Subscribe` filter to drop
unneeded data, audio, or frame signals at the ZMQ socket. Explicit-assignment
workers disable automatic subscription and call `subscribe(participant_id)`.

`run()` requests the current roster so a worker started mid-session receives
ordinary joined events for participants already present. Join callbacks must
therefore be idempotent. Because ZMQ subscription changes are asynchronous,
workers that own readiness wait for `wait_for_subscriptions()` before announcing
availability.

### Completed files

File transfer is opt-in and uses a bounded IPC lane separate from real-time messages. Pass the file publisher address and include `Subscribe.FILE`, then register an async callback:

```python
from xr_ai_hub import FileMessage, ProcessorEndpoint, Subscribe

endpoint = ProcessorEndpoint(
    sub_addr="ipc:///tmp/xr_hub_pub",
    push_addr="ipc:///tmp/xr_hub_in",
    file_sub_addr="ipc:///tmp/xr_hub_file_pub",
    filter=Subscribe.DEFAULT | Subscribe.FILE,
)

async def on_file(message: FileMessage) -> None:
    print(message.participant_id, message.topic, message.name, len(message.data))

unsubscribe = endpoint.on_file(on_file)
```

The callback runs only after DeviceIOHub has received and validated the complete LiveKit byte stream. File callbacks are awaited serially on their own receive loop, so a slow file handler does not block audio, lifecycle, or packet-data traffic. `attributes` contains only application metadata; StreamKit's reserved routing attributes are removed.

<a id="frames"></a>
(agent-sdk-hub-frames)=
## Frames and return routing

Frame callbacks carry metadata without copying pixels. `request_frame()` obtains
the selected `FrameData` only on demand and coalesces concurrent requests for the
same participant and track. `LiveFrameSource` adds a fresh-frame cache and
participant cleanup. Conversion to opaque image references belongs in
`xr_ai_tools.current_frame`; VLM inference remains a separate vision tool.

(return-path)=
Return data, audio, and flush messages always name their participant. Status is
agent state: `set_status()` and `mark_ready()` record it on the endpoint, while
DeviceIOHub aggregates responsible agents and publishes client readiness.
`republish_statuses()` restores one-shot state after reconnects.

<a id="readiness"></a>
(agent-sdk-hub-readiness)=
Readiness participation is opt-in with `announces_readiness=True`. Passive
endpoints do not contribute status. An endpoint answers only for subscribed
participants, and a live subscription must be confirmed before the hub exposes
it as ready. `agent_id` comes from the constructor, `XR_AI_AGENT_ID`, or a
process-local generated value.

## Shared memory and codec extensions

`ShmRingBuffer` and `SlotView` expose zero-copy frame views. The reader retains a
slot until `release_slot()`. The creating process may repeat ring-buffer cleanup
after one of its own shutdown paths has removed the shared-memory name; unrelated
processes must not unlink it. Application payloads
extend the msgpack codec with `register_encoder()` and `register_decoder()`;
new types use new wire identifiers rather than changing existing tags.
