<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo

Voice-driven XR scene manipulation sample. A supervisor routes natural-language
commands to five focused subagents; each subagent calls typed function groups
from `xr-ai-tools` to read and mutate the live XR scene.
See the [sample guide](../../docs/source/reference/xr-render-demo.md) for the
process stack, extension points, eval methodology, and tracing guidance.

## Running

```bash
# Start model services first:
uv run --project agent-samples/model-servers model_servers
uv run --project services/piper-tts piper_tts_server \
  --config services/piper-tts/piper_tts_server.yaml

# Start the demo stack in another terminal:
uv run --project agent-samples/xr-render-demo xr_render_demo

# Stop: send SIGTERM to the orchestrator python process (not individual services).
```
