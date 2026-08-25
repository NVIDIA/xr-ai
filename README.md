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

Refer to the [versioned documentation](https://nvidia.github.io/xr-ai/) for
setup, requirements, credentials, networking, architecture, and
troubleshooting. The landing page opens the newest complete documentation set;
a release takes precedence once it contains the current entry points. Start
with:

- [Set up with a coding agent](https://nvidia.github.io/xr-ai/latest/getting_started/skills.html)
- [Manual quickstart](https://nvidia.github.io/xr-ai/latest/getting_started/quickstart.html)
- [System requirements](https://nvidia.github.io/xr-ai/latest/getting_started/requirements.html)
- [Architecture](https://nvidia.github.io/xr-ai/latest/overview/architecture.html)

The site also publishes the current `main` branch and release-tagged versions.

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

For repository constraints and dependency boundaries, refer to
[`AGENTS.md`](AGENTS.md) and [`DEPENDENCIES.md`](DEPENDENCIES.md).

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="public-beta-notice"></a><a id="what-is-xr-ai"></a>
<a id="requirements"></a><a id="architecture"></a><a id="quickstart"></a>
<a id="model-servers-shared-ai-services"></a>
<a id="simple-vlm-example-vision-qa-over-voice--text"></a>
<a id="step-1--start-the-server"></a><a id="step-2--connect-a-client"></a>
<a id="lab-instrument-monitoring-marker-associated-readings--foreground-voice"></a>
<a id="xr-render-demo-voice-driven-sphere-in-cloudxr"></a>
<a id="step-1--start-model-servers-once"></a><a id="step-2--start-the-demo"></a>
<a id="hub-only-standalone"></a><a id="clients"></a><a id="web"></a>
<a id="android"></a><a id="ios-and-visionos"></a><a id="networking"></a>
<a id="tests"></a><a id="deeper-docs"></a><a id="project-meta"></a>
<a id="ios--visionos"></a>

## Contributing

Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution process and
[`tests/README.md`](tests/README.md) for the shortest test commands. Refer to
[`SECURITY.md`](SECURITY.md) to report security issues.

XR AI is licensed under [Apache-2.0](LICENSE). Third-party components are listed
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
