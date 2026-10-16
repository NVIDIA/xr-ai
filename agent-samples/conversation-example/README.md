<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Conversation example

This sample is a runnable voice-and-vision application in the repository. Its
reusable `xr_ai_routing.AgentRouter` routes each
turn through Clef to general conversation and direct visual observation. It
also includes an explicitly started background visual monitor and a
foreground note-taking session that can hold focus and share participant-
authored notes with general conversation. General conversation covers text
questions across subjects, including AR/VR knowledge.
The worker uses an existing live video frame when the client is streaming;
otherwise it asks the client to take one picture for that turn.
Responses stream to Pocket TTS and the `vlm.response` data topic.

The sample launches DeviceIOHub and its worker. It can also launch opt-in
session recording. It reuses the shared speaker-conditioned multi-talker STT
service, Nemotron-3 Nano Omni for conversation, Clef for routing, Cosmos3 Nano
Reasoner for visual questions, and Pocket TTS from the sample's local model
stack. Before the agent reports ready, the worker sends a representative image
request through the VLM so the first visual query does not pay the multimodal
warmup cost.

## Configure

The launcher reads the checked-in configuration automatically. Edit these
files before starting the sample:

| File | Common changes |
|---|---|
| `yaml/conversation_example_worker.yaml` | Frame freshness, VAD sensitivity, idle timeout, optional route-aware acknowledgment delay, context topics (including shared notes), monitor window size, or prompt override |
| `yaml/voice_gate.yaml` | Conversation controls, optional wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | Reused model adapters, Clef router endpoint, and conversation generation defaults |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |
| `yaml/media_capture.yaml` | Opt-in session recording output, NVENC, audio, captions, and retention |
| `worker/conversation_example_worker/prompts/` | Separate conversation, current-view, monitor, and note-taking instructions |

For example, change `followup_grace_s` in `yaml/voice_gate.yaml` to control how
long a second utterance can omit “Hey Agent”:

```yaml
followup_grace_s: 10.0
```

Restart the sample after an edit. If you change a local model server rather
than only its sample endpoint, stop and restart that stack separately. Refer to
the [sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/conversation-example.html#configuration)
for the edit workflow and to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html) for
every checked-in field and example value.

## Run

Run all commands from `agent-samples/conversation-example/`. Start the sample’s
local models first:

```bash
uv run --project model-servers conversation_example_model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample from the same terminal:

```bash
uv sync
uv run conversation_example
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

Session recording is disabled by default. To record participant video, bidirectional
audio, and data-channel traffic, opt in explicitly:

```bash
uv run conversation_example --session-recording
```

Open the authenticated web-client URL printed by DeviceIOHub, allow microphone
and camera access, and connect. After the agent reports ready, say
“Hey agent, let's start talking,” then speak questions without a wake phrase.
Say “Hey agent, let's stop talking” to close the conversation. Typed questions
remain available without opening a voice conversation. With `--session-recording`, each connection is recorded under
`~/.local/share/xr-ai/captures/conversation-example/` and finalized when the
participant disconnects or the stack stops. The normal command writes no
session recording.

The local models remain running after the sample stops. From this directory,
stop them with:

```bash
uv run --project model-servers conversation_example_model_servers --stop
```

<!-- Compatibility anchor for a heading consolidated into the documentation. -->
<a id="relay-visibility"></a>

Refer to the [sample reference](https://nvidia.github.io/xr-ai/latest/reference/conversation-example.html)
for architecture, configuration, warmup behavior, voice gating, and Relay
output.
