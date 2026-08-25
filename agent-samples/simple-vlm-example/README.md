<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

This sample is the smallest complete voice-and-vision application in the
repository. A participant can ask a spoken or typed question about the latest
camera frame. The response streams to Piper TTS and to the `vlm.response` data
topic.

The sample launches DeviceIOHub and its worker. It reuses Parakeet STT,
Cosmos3 Nano Reasoner, and Piper TTS from the shared model stack. Before the
agent reports ready, the worker sends a representative image request through
the VLM so the first user query does not pay the multimodal warmup cost.

## Configure

The launcher reads the checked-in configuration automatically. Edit these
files before starting the sample:

| File | Common changes |
|---|---|
| `yaml/simple_vlm_example_worker.yaml` | Frame freshness, VAD sensitivity, idle timeout, or prompt override |
| `yaml/voice_gate.yaml` | Wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | Reused model adapters and endpoint addresses |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |

For example, change `followup_grace_s` in `yaml/voice_gate.yaml` to control how
long a second utterance can omit “Hey Agent”:

```yaml
followup_grace_s: 10.0
```

Restart the sample after an edit. If you change a shared model server rather
than only its sample endpoint, stop and restart that stack separately. Refer to
the [sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/simple-vlm-example.html#configuration)
for the edit workflow and to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html) for
every checked-in field and example value.

## Run

Run all commands from `agent-samples/simple-vlm-example/`. Start the shared
models first:

```bash
uv run --project ../model-servers model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample from the same terminal:

```bash
uv sync
uv run simple_vlm_example
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

Open the authenticated web-client URL printed by DeviceIOHub, allow microphone
and camera access, and connect. Speak or type a question after the agent reports
ready.

The shared models remain running after the sample stops. From this directory,
stop them with `uv run --project ../model-servers model_servers --stop`.

<!-- Compatibility anchor for a heading consolidated into the documentation. -->
<a id="relay-visibility"></a>

Refer to the [sample reference](https://nvidia.github.io/xr-ai/latest/reference/simple-vlm-example.html)
for architecture, configuration, warmup behavior, voice gating, and Relay
output.
