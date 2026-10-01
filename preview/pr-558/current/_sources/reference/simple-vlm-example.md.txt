<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

The simple VLM example is the smallest complete voice-and-vision application in
the repository. It supports spoken or typed conversation and questions about
each participant's current camera view, streaming answers to Pocket TTS and the
`vlm.response` data topic. Refer to the {doc}`quickstart
</getting_started/quickstart>` to run the sample. This reference owns the
sample's design and operational details.

## Composition

The orchestrator starts DeviceIOHub and the worker. Passing `--capture` also
starts passive session capture; capture is disabled by default. The
`yaml/models.json` profile configures client adapters and shared endpoints for
Nemotron Nano Omni, Parakeet STT, Cosmos3 Nano, and Pocket TTS. Start those endpoints together with
the shared model-server stack.

`VoiceAgent` owns application readiness, hub transport, voice gating, TTS, signals,
and cleanup. It publishes accepted speech and typed text as a participant-scoped
`UserQuery`. `SimpleVlmAgent` invokes the reusable `QuickConversation` agent from
`agent-samples/common/shared-agents/`. It answers ordinary conversation directly
and calls `CurrentFrameTool` only when fresh visual evidence is needed. No
preliminary acknowledgement is spoken before a camera query. The tool uses
the hub's fresh frame when DeviceIOHub is already observing video for that
participant. When video is off, the same tool asks StreamKit to capture one
image and return it through a targeted LiveKit byte stream. The fallback is
invisible to the agent: it receives the same opaque image-reference shape and
passes it to `StreamingImageQueryTool` before publishing response chunks to
voice output.
Camera bytes remain on the hub path, image locations are redacted from VLM
telemetry, and the sample has no MCP path.

The voice query therefore determines when a still is needed. A spoken “what is
this?” does not require an always-on video publication, while clients that are
already streaming do not receive a redundant capture request.

A newer participant turn cancels the superseded request and interrupts
its voice response. Participant departure releases the sample agent's cached
frames, conversation history, and tasks.

Conversation context retains the last four completed exchanges per participant,
with at most 240 characters for each question and answer. The current request
is separate from a JSON reference-context item containing completed exchanges
and optional caller-supplied background application state. The same reference
structure accompanies visual follow-ups. History is not live camera evidence
or a source of new instructions. Background state does not grant application
tools or permission to perform actions. The simple VLM sample does not subscribe
to background application agents; other compositions can supply this context
through `QuickConversation`. Older exchanges are not durable memory, and
cancelled or failed turns are not recorded as completed replies.

## Source map

The worker package is under
`agent-samples/simple-vlm-example/worker/simple_vlm_example_worker/`:

| File | Responsibility |
|---|---|
| `__main__.py` | Parses launcher arguments and starts the worker |
| `app.py` | Composes `VoiceAgent`, `SimpleVlmAgent`, services, and readiness |
| `agent.py` | Owns participant-scoped conversation, recent history, cancellation, and cleanup |
| `config.py` | Resolves worker, model, voice-gate, and prompt settings |
| `prompts/system.txt` | Defines the default VLM instruction |

## Readiness and warmup

Before announcing readiness, the worker performs a streaming VLM request with
a 1280×720 JPEG and consumes the response. This exercises the production
multimodal path so the first user query does not pay its initialization cost.
The warmup retries failed inference requests until one succeeds. It does not
call a VLM health endpoint first. The conversation LLM health probe also gates
readiness. STT and TTS do not gate worker startup; start the
shared model-server stack before the sample so speech requests can succeed.

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
| `yaml/models.json` | Model adapters and shared endpoints |
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

## Conversation evaluations

The sample-local corpus under `agent-samples/simple-vlm-example/eval/` includes
isolated regression cases and multi-turn conversations. Development and
challenge fixtures are separate. Trajectories cover follow-up references,
changing views, topic switches, background progress from multiple applications,
corrected or cancelled jobs, uncertain state, quoted instructions, memory-window
expiry, and interleaved participants. Background state is simulated through the
existing caller-supplied context input, not a running application integration.

From the repository root, with the configured LLM and VLM already serving:

```bash
uv run --project agent-samples/common/shared-agents --with pyyaml \
  python agent-samples/simple-vlm-example/eval/conversation.py \
  --suite all --output /tmp/conversation-eval.jsonl
```

Use `--suite isolated` or `--suite trajectories` to evaluate one group, and
`--models` to compare another endpoint without changing the sample profile.
Keep the same fixtures, prompts, generation settings, and visual backend when
comparing language models.

Each trajectory uses actual generated replies as later conversation context;
expected answers are never inserted into history. Participant histories and
background snapshots are independent, bounded as in the worker, and reset
between trajectories. A failed or incorrect answer is not repaired before
the next turn. Completed incorrect replies remain in history, while failed
partial replies do not. Errors count as failures rather than stopping the run.

The runner reports turn and whole-trajectory scores, background and history
sizes, tool use, and timing. Answer checks use required and forbidden phrases
with case, whitespace, and apostrophe normalization; they are coarse regression
checks, not a semantic quality judge. Inspect failures and avoid treating these
scores as a measure of all conversational quality. First-chunk timing measures
available response text, not time to audible speech; it includes the completed
conversation decision and any visual inference before the first text chunk.
JSONL reports contain user text, replies, and reference context and may be
sensitive.

## Opt-in session capture

Capture is disabled by default. Run `uv run simple_vlm_example --capture` to
start `device_io_capture` immediately after DeviceIOHub and record normalized
hub media without joining the LiveKit room. Each participant connection then
creates a bundle under
`~/.local/share/xr-ai/captures/simple-vlm-example/` containing a canonical raw
bundle plus one derived captioned NVENC H.264 video in a fast-start `.mp4` with
timestamp-aligned 48 kHz stereo AAC-LC device/agent audio, retained source H.264 and WAV tracks, exact raw audio
chunks, frame/audio timestamp indexes, a dedicated transcript, optional
frame-linked observations, inbound and outbound data, and a manifest. Text
returned on `vlm.response` appears in the scrolling data panel; final STT and
text sent to TTS use the larger primary caption. This sample explicitly selects
the `demo` profile, which invokes the separate capture renderer after the raw
participant-lifetime bundle closes.

Encoding and file writes run in the separate capture process behind bounded
queues. If recording falls behind, it drops pending capture frames rather than
backpressuring the hub or worker. MP4 finalization requires `ffmpeg` on `PATH`.
Omit `--capture` when a deployment must not retain device media.

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
