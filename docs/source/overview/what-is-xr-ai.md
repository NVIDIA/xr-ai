<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# What is XR-AI?

XR-AI is a developer stack for building powerful XR and AI systems across
devices, platforms, and deployment environments. It connects web,
iOS/visionOS, AR glasses, and XR headset clients to GPU-accelerated AI
services, tool-using agents, and the CloudXR stack for remote rendering.

With XR-AI, developers can build agents that see and hear what the user
experiences, reason over live physical context, call external tools through
MCP, and respond with audio or data in the same XR session. The stack provides
an end-to-end foundation for multimodal spatial computing applications:
real-time media routing, participant-aware response handling, agent interfaces,
AI service integration, remote rendering, and sample applications that show the
pieces working together.

The value is speed without lock-in. XR-AI is designed to work quickly with
NVIDIA open models for vision, language, and speech, plus swappable speech
synthesis services, while still giving developers the flexibility to bring
their own models, services, tools, and application logic. Because it is built
around NVIDIA GPU infrastructure, the same architecture can be deployed where
the workload needs to run: cloud, data center, workstation, or edge.

XR-AI also gives developers a practical path across product categories. Teams
can start with AI glasses-style experiences that use live camera, audio, and
agent responses, then extend the same framework to richer AR glasses or XR
headset experiences that use CloudXR remote rendering. This lets developers
build for today's lightweight AI devices while keeping a clear path to
immersive, GPU-rendered spatial applications.

XR-AI is especially useful when you need to:

- **Build multimodal XR agents** that can see, hear, reason, use tools, and
  respond in real time.
- **Target multiple client platforms** including web, iOS/visionOS, AR
  glasses, and XR headsets.
- **Use NVIDIA open models out of the box** while preserving the flexibility to
  bring your own models and services.
- **Deploy wherever NVIDIA GPUs are available**, from cloud and data center to
  workstation and edge.
- **Start with AI glasses-style experiences and scale to CloudXR remote
  rendering** for richer AR and XR applications.
- **Keep transport, rendering, model services, tools, and agent logic
  separated** so teams can evolve each layer independently.

## The layer model

XR-AI separates concerns into independent layers so teams can evolve each one
on its own. Clients connect to the server runtime, agents attach over an IPC
boundary, and AI inference is reached through swappable model services.

| Layer | Directory | Description |
|---|---|---|
| Clients | `client-samples/` | Android, iOS/visionOS, Web, and native StreamKit clients |
| Server runtime | `server-runtime/` | XR-Media-Hub + LiveKit internal transport |
| Launcher | `utils/xr-ai-launcher/` | stdlib-only process manager used by samples |
| Logging | `utils/xr-ai-logging/` | shared loguru sink + stdlib bridge for every process |
| Agent interfaces | `agent-mcp-servers/` | MCP adapters for XR data & rendering |
| Agent demos | `agent-samples/` | End-to-end agent pipelines |
| Tests | `tests/` | Multi-client / multi-agent integration tests |

Lightweight samples (`simple-vlm-example`) are self-contained — one command
starts everything. Heavier demos (`xr-render-demo`) split model loading from
the demo itself: start `model-servers` once, then run the demo as many times
as you like without reloading weights.

Every sample worker depends on `agent-sdk/xr-ai-models` — one SDK that
abstracts the OpenAI-compatible HTTP wire format for LLM / VLM / STT / TTS
behind four service protocols. Each sample ships a `yaml/models.yaml` that
names the logical models the worker needs (`llm`, `vlm`, `stt`, …) with preset
references that pre-fill model-specific quirks (reasoning-field aliasing,
`chat_template_kwargs`, served-model-name strings). Workers call
`make_llm(config, "llm")` / `make_vlm(config, "vlm")` / `make_stt(config,
"stt")` / `make_tts(config, "tts")` — no hand-rolled httpx clients, no model
quirks leaking out of the SDK. Full quickstart and the built-in preset table
live in `agent-sdk/xr-ai-models/README.md`.

For how the hub, transport, and agents fit together, see {doc}`architecture`.
To run a sample, see the {doc}`/getting_started/quickstart`.
