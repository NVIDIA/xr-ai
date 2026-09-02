<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

The simple VLM example is the smallest complete voice-and-vision application in
the repository. It answers spoken or typed questions about each participant's
latest camera frame and streams the answer to both Piper TTS and the
`vlm.response` data topic. Refer to the {doc}`quickstart
</getting_started/quickstart>` to run the sample. This reference owns the
sample's design and operational details.

## Composition

The orchestrator starts DeviceIOHub and the worker. Passing `--capture` also
starts passive session capture; capture is disabled by default. The fixed
`yaml/models.json` profile marks Parakeet STT, Cosmos3 Nano, and Piper TTS as
reused services. Start those endpoints together with the shared model-server
stack.

`VoiceAgent` owns service readiness, hub transport, voice gating, TTS, signals,
and cleanup. It publishes accepted speech and typed text as a participant-scoped
`UserQuery`. `SimpleVlmAgent` selects the participant's current image with
`CurrentFrameTool`, passes its opaque reference to
`StreamingImageQueryTool`, and publishes response chunks to voice output.
Camera bytes remain on the hub path, image locations are redacted from VLM
telemetry, and the sample has no MCP path.

A newer participant turn cancels the superseded vision request and interrupts
its voice response. Participant departure releases the sample agent's cached
frames and tasks. This is the reference composition for a single foreground
streaming image query.

## Source map

The worker package is under
`agent-samples/simple-vlm-example/worker/simple_vlm_example_worker/`:

| File | Responsibility |
|---|---|
| `__main__.py` | Parses launcher arguments and starts the worker |
| `app.py` | Composes `VoiceAgent`, `SimpleVlmAgent`, services, and readiness |
| `agent.py` | Owns participant-scoped vision turns, cancellation, and cleanup |
| `config.py` | Resolves worker, model, voice-gate, and prompt settings |
| `prompts/system.txt` | Defines the default VLM instruction |

## Readiness and warmup

Before announcing readiness, the worker performs a streaming VLM request with
a 1280×720 JPEG and consumes the response. This exercises the production
multimodal path so the first user query does not pay its initialization cost.
Endpoint health alone is not sufficient for that warmup.

## Configuration

Run and edit the sample from `agent-samples/simple-vlm-example/`. The
orchestrator always passes `yaml/device_io_hub.yaml` to DeviceIOHub and
`yaml/simple_vlm_example_worker.yaml` to the worker. The worker resolves its
models and voice-gate files relative to the worker YAML, so the checked-in
layout works without command-line configuration arguments.

| File | Owns |
|---|---|
| `yaml/simple_vlm_example_worker.yaml` | Frame freshness and wait limits, VAD, idle timeout, and optional prompt overrides |
| `yaml/voice_gate.yaml` | Wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | Model adapters, endpoints, readiness, and reuse declarations |
| `yaml/device_io_hub.yaml` | LiveKit room and ports, web and token servers, and network behavior |
| `yaml/media_capture.yaml` | Opt-in media-hub capture, NVENC output, caption layout, and retention |
| `worker/simple_vlm_example_worker/prompts/system.txt` | Default VLM instruction |

Edit the owning file, preserve the field's YAML type, and restart
`simple_vlm_example`; configuration is loaded only at process startup. For
example, lower `silero_threshold` in the worker YAML if quieter speech is being
missed, or change `magic_phrases` in the voice-gate YAML to choose the required
wake phrases. Relative paths in the worker YAML are resolved from `yaml/`, not
from the shell's current directory.

Changing an entry in `models.json` changes only the client adapter or endpoint
that this sample uses. It does not reconfigure or restart the shared server.
For a checkpoint, port, GPU, or model-runtime change, refer to
{doc}`/guides/customizing-model-servers`, update the shared stack, stop that
persistent stack, and start it again before restarting this sample.

Refer to the generated {doc}`configuration <configuration>` reference for exact
fields, checked-in values, and adjacent YAML comments.

## Opt-in session capture

Capture is disabled by default. Run `uv run simple_vlm_example --capture` to
start `device_io_capture` immediately after DeviceIOHub and record normalized
hub media without joining the LiveKit room. Each participant connection then
creates a bundle under
`~/.local/share/xr-ai/captures/simple-vlm-example/` containing one captioned
NVENC H.264 video in a playable `.mkv` with timestamp-aligned device/agent
audio embedded, retained source H.264 and WAV tracks, exact raw audio chunks,
inbound and outbound data, and a manifest. Text returned on `vlm.response`
appears in the scrolling data panel; final STT and text sent to TTS use the
larger primary caption.

Encoding and file writes run in the separate capture process behind bounded
queues. If recording falls behind, it drops pending capture frames rather than
backpressuring the hub or worker. Omit `--capture` when a deployment must not
retain device media.

Wake phrases match at the start of a final transcript or after sentence-final
`.`, `?`, or `!` punctuation followed by whitespace or a closing quote. Text
before that boundary and the phrase itself are discarded before dispatch.

## Relay output

The worker writes `relay-events.jsonl` beside `worker.log` in the per-run log
directory printed at startup. It records runtime publications, receiving-agent
callbacks, the complete `simple-vlm.turn` lifetime, and nested vision and model
calls. Per-token model marks, incremental voice fragments, and empty stream
terminators are omitted. Voice and STT each emit one semantic scope for a
completed operation. TTS emits one ``voice.tts`` scope for each sentence sent
to synthesis. Raw audio is represented only by size, duration, and sample rate.

Image locations appear as `<redacted:image>`. Prompts, questions, responses,
participant identities, and correlation metadata remain visible and may be
sensitive. Inspect a running sample with:

```bash
tail -F /tmp/log_simple-vlm-example_*/relay-events.jsonl
```
