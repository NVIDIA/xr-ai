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

## Configure

Edit the sample-owned files before starting the stack; the launcher reads them
automatically:

| File | Common changes |
|---|---|
| `yaml/tea_making_worker.yaml` | Selected configuration files, observation cadence, VAD, timeouts, artifacts, and event viewer |
| `yaml/workflow.yaml` | Workflow state, steps, evidence rules, tools, and messages |
| `yaml/voice_gate*.yaml` | Wake-word or always-on speech behavior |
| `yaml/rag_service.yaml` | Documents, embedding role, cache, chunking, and retrieval threshold |
| `yaml/models.local.json` | Reused model adapters and endpoint addresses |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |

For example, make speech always-on by changing this line in
`yaml/tea_making_worker.yaml`:

```yaml
voice_gate_yaml: voice_gate.always-on.yaml
```

Restart the sample after an edit. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/tea-making-sample.html#configuration)
for the workflow and path-resolution rules and to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html) for
every checked-in field and example value.

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

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
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

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="run-it"></a><a id="foreground-behavior"></a>
<a id="foreground-routing-eval"></a><a id="file-outputs"></a>
<a id="configuration"></a><a id="safety"></a>

Refer to the [sample reference](https://nvidia.github.io/xr-ai/latest/reference/tea-making-sample.html)
for architecture, configuration, output contracts, safety, foreground-routing
behavior, evaluation, and adaptation guidance.
