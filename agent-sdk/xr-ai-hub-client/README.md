<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-hub-client

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

endpoint.on_data(on_data)
await endpoint.run()
```

`LiveFrameSource` adds raw frame acquisition without adding image conversion or
model dependencies. Use `xr_ai_nat.functions.vision.StreamingVisionConfig` when the desired
interface is a model-facing vision function rather than raw pixels.

## Migrating from `xr-ai-agent`

This package was previously the distribution `xr-ai-agent`, living at the
`agent-sdk/` root and importable as `xr_ai_agent`.

**The dependency rename is breaking.** `xr-ai-agent` no longer resolves; declare
`xr-ai-hub-client` instead, and point the path at this directory:

```toml
[project]
dependencies = ["xr-ai-hub-client"]

[tool.uv.sources]
xr-ai-hub-client = { path = "../../agent-sdk/xr-ai-hub-client", editable = true }
```

**The import rename is not.** Once the dependency is switched, `import
xr_ai_agent` keeps working: this distribution also ships an `xr_ai_agent`
package that forwards every public name to `xr_ai_hub` and emits a
`DeprecationWarning` on import. The public API is identical, so migrating is a
find-and-replace:

```python
from xr_ai_hub import DataMessage, ProcessorEndpoint   # was: from xr_ai_agent import ...
```

The alias exists to keep out-of-tree code importing while it migrates, and will
be removed in a future version.
