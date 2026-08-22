<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Agent SDK

Packages are named for the Python surface developers import. Each package has
its own environment and concise entry-point README.

| Directory | Import | Distribution | Use it for |
|---|---|---|---|
| [`xr-ai-hub`](xr-ai-hub/) | `xr_ai_hub` | `xr-ai-hub-client` | Minimal msgpack/ZMQ IPC with DeviceIOHub |
| [`xr-ai-models`](xr-ai-models/) | `xr_ai_models` | `xr-ai-models` | Typed model services and OpenAI-compatible clients |
| [`xr-ai-runtime`](xr-ai-runtime/) | `xr_ai_runtime` | `xr-ai-agent-runtime` | Agent registration and typed publish/fan-out |
| [`xr-ai-tools`](xr-ai-tools/) | `xr_ai_tools` | `xr-ai-tools` | Relay-managed tools and model tool-call helpers |
| [`xr-ai-voice`](xr-ai-voice/) | `xr_ai_voice` | `xr-ai-voice` | Voice agent, session, transport, and pipeline |
| [`xr-ai-web-events`](xr-ai-web-events/) | `xr_ai_web_events` | `xr-ai-web-events` | Bounded live browser views over selected application events |

Start with `xr_ai_hub` for raw media/data IPC, or compose `xr_ai_runtime`,
`xr_ai_tools`, `xr_ai_models`, and `xr_ai_voice` for a tool-using voice agent.
See the [Agent SDK guide](../docs/source/components/agent-sdk.md) for the package
boundaries.

Breaking-change replacements are maintained in the
[migration guide](../docs/source/reference/migrations.md).
