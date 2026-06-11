<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI

Build AI agents that see and hear what your users experience in XR, and respond
in real time.

XR AI is an open-source stack that connects web, iOS/visionOS, AR-glasses, and
XR-headset clients to GPU-accelerated AI services and tool-using agents. An agent
can perceive live physical context, call tools through MCP, and reply with audio
or data in the same session. For remote-rendered AR and XR, XR AI integrates
[NVIDIA CloudXR](https://developer.nvidia.com/cloudxr-sdk), as the `xr-render-demo`
sample shows.

It is especially useful when you need to:

- **Build multimodal XR agents** that see, hear, reason, use tools, and respond
  in real time.
- **Target multiple client platforms** — web, iOS/visionOS, AR glasses, and XR
  headsets.
- **Use NVIDIA open models out of the box** while keeping the freedom to bring
  your own.
- **Deploy wherever NVIDIA GPUs run** — cloud, data center, workstation, or edge.
- **Integrate [NVIDIA CloudXR](https://developer.nvidia.com/cloudxr-sdk)** for
  remote-rendered AR and XR, as the `xr-render-demo` sample shows.
- **Keep transport, rendering, model services, tools, and agent logic separated**
  so each layer evolves independently.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🚀 Get started
:link: getting_started/quickstart
:link-type: doc
Run a sample in minutes: the model servers, the simple VLM agent, or the full
xr-render demo.
:::

:::{grid-item-card} 🧩 Architecture
:link: overview/architecture
:link-type: doc
How XR-Media-Hub, the transport, and agents fit together.
:::

:::{grid-item-card} 🛠️ Components
:link: components/index
:link-type: doc
The server runtime, agent SDK, MCP servers, AI services, and the launcher.
:::

:::{grid-item-card} 📦 Build a sample
:link: guides/adding-a-sample
:link-type: doc
Wire your own agent worker into the stack.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

overview/index
getting_started/index
components/index
guides/index
reference/index
```
