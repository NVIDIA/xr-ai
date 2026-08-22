<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo

This sample connects a conversational agent to a live CloudXR scene. A
supervisor routes natural-language requests to focused subagents that inspect
and modify scene objects, tracking, spatial state, and recorded visual context
through native `xr-ai-tools` functions.

The sample launches DeviceIOHub, the CloudXR runtime, the scene and capability
services, and its worker. It reuses STT, Piper TTS, Nemotron-3 Nano Omni, and
Cosmos3 Nano Reasoner from the shared model stack. The default client is the
browser-based Web-XR experience.

## Prerequisites

Install the Vulkan loader and headers and make Node.js 18 or newer with npm
available on `PATH`. On its first run, the orchestrator downloads the pinned
LOVR build and creates the Web-XR vendor bundle; later runs reuse those files.

## Run

Run all commands from `agent-samples/xr-render-demo/`. Start the shared models
first:

```bash
uv run --project ../model-servers model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample from the same terminal:

```bash
uv sync
uv run xr_render_demo
```

Open the authenticated client URL printed by DeviceIOHub and connect. Press
Ctrl+C to stop the demo stack; do not stop its individual child processes.

The shared models remain running after the sample stops. From this directory,
stop them with `uv run --project ../model-servers model_servers --stop`.

Refer to the [sample reference](../../docs/source/reference/xr-render-demo.md)
for client selection, the process stack, extension points, evaluation, GPU
placement, and tracing guidance.
