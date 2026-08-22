<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI

Agentic AI for XR — an open-source foundation for multimodal, real-time
conversational AI in the NVIDIA CloudXR ecosystem.

XR AI connects web, Android, iOS/visionOS, and native clients to a shared media
hub, GPU-accelerated AI services, tool-using agents, and optional CloudXR remote
rendering. Agents can see and hear what a participant experiences, call native
tools, and return audio or data to the same participant.

This project is a public beta. APIs and behavior may change as it evolves.

## Get started

Use the [versioned documentation](https://nvidia.github.io/xr-ai/) for setup,
requirements, credentials, networking, architecture, and troubleshooting:

- [Set up with a coding agent](https://nvidia.github.io/xr-ai/main/getting_started/skills.html)
- [Manual quickstart](https://nvidia.github.io/xr-ai/main/getting_started/quickstart.html)
- [System requirements](https://nvidia.github.io/xr-ai/main/getting_started/requirements.html)
- [Architecture](https://nvidia.github.io/xr-ai/main/overview/architecture.html)

The documentation defaults to the latest release and also publishes the current
`main` branch and release-tagged versions.

## Samples

| Sample | Purpose |
|---|---|
| [`model-servers`](agent-samples/model-servers/README.md) | Start and persist the shared model stack |
| [`simple-vlm-example`](agent-samples/simple-vlm-example/README.md) | Voice and text questions about the current camera frame |
| [`lab-instrument-monitoring`](agent-samples/lab-instrument-monitoring/README.md) | Marker-associated visual monitoring with a foreground voice agent |
| [`tea-making-sample`](agent-samples/tea-making-sample/README.md) | Guided workflow with visual evidence and background observations |
| [`xr-render-demo`](agent-samples/xr-render-demo/README.md) | Voice-driven CloudXR scene manipulation |

Each sample README gives the shortest runnable command sequence. The linked
documentation contains architecture, behavior, configuration, output contracts,
evaluation, and adaptation guidance.

## Repository map

| Directory | Contents |
|---|---|
| `client-samples/` | Platform clients and shared StreamKit implementations |
| `agent-sdk/` | Hub IPC, model clients, runtime, tools, voice, and web events |
| `agent-samples/` | Runnable agent stacks |
| `services/` | Hub, model servers, and typed capability services |
| `utils/` | Launcher, logging, VAD, vLLM, and voice-gate utilities |
| `tests/` | Cross-package and integration tests |
| `docs/source/` | Canonical user and contributor documentation |
| `skills/` | Setup skills for coding agents |

For repository constraints and dependency boundaries, read
[`AGENTS.md`](AGENTS.md) and [`DEPENDENCIES.md`](DEPENDENCIES.md).

## Contributing

Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution process and
[`tests/README.md`](tests/README.md) for the shortest test commands. Report
security issues according to [`SECURITY.md`](SECURITY.md).

XR AI is licensed under [Apache-2.0](LICENSE). Third-party components are listed
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
