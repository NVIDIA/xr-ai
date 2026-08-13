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
