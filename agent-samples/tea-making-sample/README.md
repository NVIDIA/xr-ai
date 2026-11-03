<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance

This sample composes a control-only SOP runner and an SDK conversation agent.
The runner executes independent Python async identification, visual-verification
and timer flows; conversation answers from runner-supplied context. Application
YAML supplies prompts and arguments. Automatic progression is the default; use
`sop.advancement: manual` in `yaml/tea_making_worker.yaml` for explicit next.

Read the [sample architecture](https://nvidia.github.io/xr-ai/latest/reference/tea-making-sample.html)
for Python APIs, context, recording and model qualification.

## Configure

Edit the sample-owned files before starting the stack; the launcher reads them
automatically:

| File | Common changes |
|---|---|
| `yaml/tea_making_worker.yaml` | Selected configuration files, observation cadence, VAD, timeouts, artifacts, and event viewer |
| `yaml/voice_gate.yaml` | Interaction mode, conversation controls and optional wake phrases |
| `yaml/rag_service.yaml` | Documents, embedding role, cache, chunking, and retrieval threshold |
| `yaml/models.local.json` | Reused model adapters and endpoint addresses |
| `yaml/media_capture.yaml` | Opt-in session recording output, NVENC, audio, captions, and retention |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |

For example, accept wake-addressed one-off questions as well as natural
conversations by changing this line in `yaml/voice_gate.yaml`:

```yaml
mode: hybrid
```

Restart the sample after an edit. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/tea-making-sample.html#configuration)
for the workflow and path-resolution rules and to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html) for
every checked-in field and example value.

## Run

Run all commands from `agent-samples/tea-making-sample/`. Start the sample’s local
models first:

```bash
uv run --project model-servers tea_making_sample_model_servers
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

To record a demo session, use:

```bash
uv run tea_making_sample --session-recording
```

Refer to the [session recording guide](https://nvidia.github.io/xr-ai/latest/reference/tea-making-sample.html#opt-in-continuous-session-capture)
for output and encoding requirements.

To make the unauthenticated event viewer reachable from a trusted private
network, use this alternative sample command:

```bash
uv run tea_making_sample --expose-web-events
```

Open the authenticated web-client URL printed by DeviceIOHub, accept the
development certificate on first use, allow microphone and camera access, and
connect. Say “Hey agent, let's start
talking,” then speak commands without a wake phrase. Say “Hey agent, let's stop
talking” to close the conversation. The optional event viewer is available at
`http://127.0.0.1:8093` with the checked-in worker configuration. Each launch
writes an SOP bundle below `artifacts/runs/`, including consumed inference images
and step records. See [SOP run capture](https://nvidia.github.io/xr-ai/main/reference/tea-making-sample.html#sop-run-capture).

The local models remain running after the sample stops. From this directory,
stop them with `uv run --project model-servers tea_making_sample_model_servers --stop`.

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="run-it"></a><a id="foreground-behavior"></a>
<a id="foreground-routing-eval"></a><a id="file-outputs"></a>
<a id="configuration"></a><a id="safety"></a>

Refer to the [sample reference](https://nvidia.github.io/xr-ai/latest/reference/tea-making-sample.html)
for architecture, configuration, output contracts, safety, SOP control-routing
behavior, evaluation, and adaptation guidance.
