<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance

This sample demonstrates a guided, evidence-aware workflow. A foreground tea
guide answers questions and applies explicit workflow controls, while
independent background agents can record transcripts, watch for visual
changes, and produce periodic video observations.

The workflow keeps state transitions deterministic: visual evidence can satisfy
the current step, but only an explicit user command advances the procedure. The
sample launches its application services and reuses STT, Piper TTS, Nemotron-3
Nano Omni, and embedding endpoints from the shared model stack.

## Run

Run all commands from `agent-samples/tea-making-sample/`. Start the shared
models first:

```bash
uv run --project ../model-servers model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample from the same terminal:

```bash
uv sync
uv run tea_making_sample
```

To make the unauthenticated event viewer reachable from a trusted private
network, use this alternative sample command:

```bash
uv run tea_making_sample --expose-web-events
```

Open `https://localhost:8080`, accept the development certificate on first use,
allow microphone and camera access, and connect. Begin voice commands with
“Agent” or “Hey Agent.” The optional event viewer is available at
`http://127.0.0.1:8092` by default, and durable output is written below
`artifacts/`.

The shared models remain running after the sample stops. From this directory,
stop them with `uv run --project ../model-servers model_servers --stop`.

Refer to the [sample reference](../../docs/source/reference/tea-making-sample.md)
for architecture, configuration, output contracts, safety, foreground-routing
behavior, evaluation, and adaptation guidance.
