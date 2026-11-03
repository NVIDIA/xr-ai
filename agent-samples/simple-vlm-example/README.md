<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

> **Deprecated:** Use the
> [conversation example](../conversation-example/README.md) as the starting
> point for new voice-and-vision applications. Simple VLM remains available
> for the browser demo and existing integrations. Refer to the
> [Simple VLM reference](https://nvidia.github.io/xr-ai/latest/reference/simple-vlm-example.html)
> for its design and operational details.

This sample is the smallest complete voice-and-vision application in the
repository. A participant can ask a spoken or typed question about the current
camera view. The worker uses an existing live video frame when the client is
streaming; otherwise it asks the client to take one picture for that turn. The
response streams to Pocket TTS and to the `vlm.response` data topic.

The sample launches DeviceIOHub and its worker. It can also launch opt-in
session recording. It reuses speaker ASR, Cosmos3 Nano Reasoner, and Pocket TTS
from the sample's local model stack. Before the agent reports ready, the worker sends a
representative image request through the VLM so the first user query does not
pay the multimodal warmup cost.

## Configure

The launcher reads the checked-in configuration automatically. Edit these
files before starting the sample:

| File | Common changes |
|---|---|
| `yaml/simple_vlm_example_worker.yaml` | Frame freshness, VAD sensitivity, idle timeout, or prompt override |
| `yaml/voice_gate.yaml` | Conversation controls, optional wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | Reused model adapters and endpoint addresses |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |
| `yaml/media_capture.yaml` | Opt-in session recording output, NVENC, audio, captions, and retention |

For example, set `mode: hybrid` in `yaml/voice_gate.yaml` to accept
wake-addressed questions while the conversation is closed:

```yaml
mode: hybrid
```

Restart the sample after an edit. If you change a local model server rather
than only its sample endpoint, stop and restart that stack separately. Refer to
the [sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/simple-vlm-example.html#configuration)
for the edit workflow and to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html) for
every checked-in field and example value.

## Run

For a coding agent to launch and verify this sample, refer to
[`run-demo`](https://nvidia.github.io/xr-ai/latest/getting_started/skills.html#run-the-browser-demo).

Run all commands from `agent-samples/simple-vlm-example/`. Start the sample’s local
models first:

```bash
uv run --project model-servers simple_vlm_example_model_servers
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

Session recording is disabled by default. To record participant video, bidirectional
audio, and data-channel traffic, opt in explicitly:

```bash
uv run simple_vlm_example --session-recording
```

Open the authenticated web-client URL printed by DeviceIOHub, allow microphone
and camera access, and connect. After the agent reports ready, say
“Hey agent, let's start talking,” then speak questions without a wake phrase.
Say “Hey agent, let's stop talking” to close the conversation. Typed questions
remain available without opening a voice conversation. With `--session-recording`, each connection is recorded under
`~/.local/share/xr-ai/captures/simple-vlm-example/` and its raw bundle is finalized when the
participant disconnects or the stack stops. For a captioned MP4, disconnect the
participant and wait for rendering to finish before stopping the stack; shutdown
finalizes active raw bundles without starting new renders. The normal command writes no
session recording.

The local models remain running after the sample stops. From this directory,
stop them with `uv run --project model-servers simple_vlm_example_model_servers --stop`.

<!-- Compatibility anchor for a heading consolidated into the documentation. -->
<a id="relay-visibility"></a>

Refer to the [sample reference](https://nvidia.github.io/xr-ai/latest/reference/simple-vlm-example.html)
for architecture, configuration, warmup behavior, voice gating, and Relay
output.
