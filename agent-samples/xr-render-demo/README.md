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

## Configure

Edit the sample-owned configuration before starting the demo; the launcher
passes each file to the process that owns it:

| File | Common changes |
|---|---|
| `yaml/cloudxr_runtime.yaml` | Client profile, EULA acceptance, and compositor GPU |
| `yaml/xr_render_demo_worker.yaml` | Capability endpoints, text memory, VAD, and idle timeout |
| `yaml/voice_gate.yaml` | Always-on speech or wake phrases |
| `yaml/models.json` | Reused model adapters and endpoint addresses |
| `yaml/device_io_hub.yaml` | Room, ports, web client, networking, and video recording |
| `yaml/video_memory_service.yaml` | Recorded-query output and GPU |
| `yaml/openxr_service.yaml` | OpenXR endpoint and eval-only simulated pose |
| `scene/scene_service.yaml` | LOVR binary, app directory, and scene endpoint |

For example, select native iOS or visionOS clients and the compositor GPU in
`yaml/cloudxr_runtime.yaml`:

```yaml
cloudxr_env:
  NV_DEVICE_PROFILE: auto-native
gpu_index: 1
```

Choose a `gpu_index` that exists in `nvidia-smi`. Restart the demo after an
edit. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/xr-render-demo.html#configuration)
for precedence and GPU-placement details and to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html) for
every checked-in field and example value.

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

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

Open the authenticated client URL printed by DeviceIOHub and connect. Press
Ctrl+C to stop the demo stack; do not stop its individual child processes.

The shared models remain running after the sample stops. From this directory,
stop them with `uv run --project ../model-servers model_servers --stop`.

Refer to the [sample reference](https://nvidia.github.io/xr-ai/latest/reference/xr-render-demo.html)
for client selection, the process stack, extension points, evaluation, GPU
placement, and tracing guidance.
