<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Conversation example

The conversation example is a runnable voice-and-vision application in the
repository. It uses Clef to route ordinary spoken or typed questions among
general conversation, direct visual observation, and an explicitly started
background visual monitor. General conversation answers text questions across
subjects, including AR/VR knowledge and shared-context recall. A separate
foreground note-taking session captures and reads back participant-authored
notes, then shares them with general conversation. The background monitor
continuously records a bounded window of timestamped visual observations and
publishes its text summary to configured conversation context. Conversation
destinations use the reusable
`xr_ai_conversation.Conversation` component for answer/context behavior. The
sample composes it with `xr_ai_routing.AgentRouter`, then
streams through the same voice output path to Pocket TTS and the `vlm.response`
data topic. Refer to the {doc}`quickstart
</getting_started/quickstart>` for client setup. The sample README contains the
exact run commands. This reference owns the sample's design and operational
details.

## Runtime boundary

`AgentRouter` and `VoiceAgent` are registered directly with the same
`AgentRuntime`. The sample configures the router's query, participant lifecycle,
interruption, and voice output topics. The router owns routing tasks, focus,
shared turn history, acknowledgments, and route-error responses. Each destination
publishes its own attributed `ResponseOutput` messages. Conversation and direct
observation stream chunks; note-taking and monitor controls publish complete
responses.

```text
VoiceAgent
  -- UserQuery and participant lifecycle topics -->
AgentRouter
  -- route, admit, and answer -->
Selected destination
  -- ResponseOutput topic -->
VoiceAgent
```

`_app.py` constructs and registers the router, voice agent, and configured
conversation context subscriber. It supplies participant initialization and
cleanup hooks for the sample's capability snapshots, notes, monitor, and frame
resources. The router initializes those resources before a participant's first
query, even when that query arrives before the participant-joined event.

## Composition

The orchestrator starts DeviceIOHub and the worker. Passing `--session-recording` also
starts passive session capture; capture is disabled by default. The
`yaml/models.json` profile configures client adapters and endpoints for
Nemotron-3 Nano Omni (answering), Cosmos3 Nano Reasoner (VLM), Pocket TTS, and
the sample's pinned Clef decision endpoint. `yaml/voice_gate.yaml` requires the
speaker-conditioned multi-talker STT service at `127.0.0.1:8102`; the local
model-server stack configures that service's models and GPU placement. The LLM adapter
enables Omni reasoning with a 4096-token soft reasoning hint and a 4608-token
total output limit. Start those endpoints together with
the sample's local model-server stack. The worker checks LLM, Clef, and VLM
health before opening the voice session; readiness does not send an inference
request through the VLM.

`VoiceAgent` owns application readiness, hub transport, voice gating, TTS, signals,
and cleanup. It publishes accepted speech and typed text as a participant-scoped
`UserQuery`. `xr_ai_routing.AgentRouter` subscribes directly to the configured
query topic. Clef selects general conversation,
direct visual observation, background monitoring, or foreground note-taking
from routing aids and the current request/history. Acknowledgments are optional
router output after a destination accepts; clarification can come from the
router or a route agent's admission. Admission stays with each route agent:
general conversation
uses `Conversation.admit()`, direct observation accepts its current-frame
requests, and monitoring and note-taking use their own typed control decisions.
A decline produces no spoken output and gives the
router one bounded reroute. A selected destination can ask a
task-specific question through its normal answer path; unresolved routing uses
the router's clarification. Exhausted declines fall back to general conversation
when it remains eligible, otherwise the router clarifies. General conversation
answers text questions across subjects and has no frame or VLM tools. Direct
observation sends the fresh image and current user question only to Cosmos; it
does not include completed conversation history in that request. The
background monitor is explicitly started and stopped through ordinary routed
turns; it samples sequentially without a cadence delay, keeps a bounded rolling
window, and shares timestamped text through the configured context topic. Its
routing aid exposes only active task scope, status, observation availability,
and newest source time. When that task covers a request, general conversation
can use the separate timestamped window context; a fresh-camera request goes to
direct observation. The monitor destination handles start, retarget, stop,
status, and handoff controls only and does not answer recorded-content questions.
Retarget replaces the active task and rolling window; handoff declines so another
destination can answer the same request. It does not take
router focus. A distinct note-taking destination acquires focus
on request. It stores user-authored notes verbatim, supports readback, and keeps
focus through topic changes until a semantic finish. Finishing notes consumes
that request and releases focus. A question handoff declines only that query,
so another destination can answer and the next turn returns to note capture.
Captured notes remain available to
general conversation afterward. Observation records are never converted into
notes. Scope admission is based on the request,
not whether the camera will return a usable frame. Direct visual requests must
identify their target in the current request because prior conversation is not
sent to Cosmos. The
`CurrentFrameTool` uses
the hub's fresh frame when DeviceIOHub is already observing video for that
participant. When video is off, the same tool asks StreamKit to capture one
image and return it through a targeted LiveKit byte stream. The fallback is
invisible to the agent: it receives the same opaque image-reference shape and
passes it to the appropriate Cosmos query path before publishing response
chunks to voice output.
Camera bytes remain on the hub path, image locations are redacted from VLM
telemetry, and the sample has no MCP path.

For direct observation, the current voice or text query determines when a still
is needed. The separately started monitor captures continuously while its task
is active; it does not imply that ordinary queries need always-on video.

A newer participant turn cancels superseded routing, answer, and vision work and
interrupts its voice response. Participant departure releases cached frames,
conversation destination state, AgentRouter history, and source-context
tasks. This is the reference composition for streamed conversation and
current-view visual questions.

Conversation destinations can subscribe to named runtime context topics
configured in `yaml/conversation_example_worker.yaml`. The background monitor
publishes its rolling observation window as a replaceable summary, and the
notes destination publishes its bounded note list as a replaceable summary.
Publishers send text-only `ConversationContextUpdate` values. Only the general
conversation subscribes to the note topic. Incremental sources are compacted in
the background per participant and source; the resulting summary replaces the
compacted records for that source. A summary update larger than its topic budget
is clipped at its start and end before storage. Incremental updates remain
stored while compaction runs; when too much uncompacted context is present, the
oldest updates are omitted from an answer request but retained for later
compaction. The rendered dialogue/context and compaction input budgets are
UTF-8 byte budgets, not token limits.
The sample maps `context_budget_bytes` across the general conversation's
dialogue and each configured context topic. With the three default topics, a
12,000-byte setting gives 3,000 bytes to dialogue and each topic, preserving
the sample's total context allowance. The same setting also bounds
sample-owned routing cues, notes, and monitor windows before publication.
`compaction_input_budget_bytes` separately limits one summary request; its
default is 12,000 bytes. The notes conversation has no context topics and uses
the full configured context budget for dialogue.

Conversation prompts show the current time and coarse minute-rounded timestamps
for completed exchanges and named context where available. FE-owned turns use
their completion time, runtime context uses its publication time, and visual
evidence includes the frame's source timestamp. Imported history without a time
is labeled unknown. These timestamps help the model distinguish older context
from a present observation; they do not expire or remove information.

## Source map

The worker package is under
`agent-samples/conversation-example/worker/conversation_example_worker/`:

| File | Responsibility |
|---|---|
| `__main__.py` | Parses launcher arguments and starts the worker |
| `_app.py` | Composition root for `VoiceAgent`, `AgentRouter`, route agents, services, and readiness |
| `_topics.py` | Names the sample-owned voice input and lifecycle topics |
| `_config.py` | Resolves worker, model, voice-gate, and prompt settings |
| `_note_taking.py` | Implements the sample-owned focused note-taking destination |
| `_monitoring.py` | Owns the explicitly started background monitor and rolling text window |
| `_visual_agents.py` | Implements direct current-image observation |
| `prompts/system.txt` | Defines the default VLM instruction |
| `prompts/conversation.txt` | Application instructions for the conversation destination |
| `prompts/note_taking.txt` | Instructions for the focused note-taking destination |
| `prompts/monitor_observation.txt` | Produces concise observations for the rolling monitor window |
| `eval/conversation-cases.yaml` | Single-turn Clef routing and direct-answer evaluation inputs |
| `eval/conversation-flows.yaml` | Ordered multi-turn routing, focus, and context-update cases |

## Behavior validation

The sample's deterministic tests divide the contract by layer. Run them from the
repository root with the `tests` project environment:

```bash
uv --config-file uv.toml run --project tests pytest \
  tests/test_xr_ai_conversation.py \
  tests/test_xr_ai_routing.py \
  tests/test_conversation_example_worker.py \
  tests/test_conversation_example_visual_destinations.py \
  tests/test_conversation_example_monitoring.py \
  tests/test_conversation_example_flows.py \
  tests/test_conversation_example_decision_eval.py
```

| Behavior | Deterministic contract | Corpus/model layer |
|---|---|---|
| Routing, native admission, decline/reroute, default fallback, focus, cancellation | `test_xr_ai_routing.py` | Raw choice, effective destination, disposition, and invoked admissions |
| Completed history, named context, byte budgets, compaction, acknowledgment races | `test_xr_ai_conversation.py` | Context sources and answer checks when captured |
| Current-frame-only observation and camera/tool failures | `test_conversation_example_visual_destinations.py`, `test_conversation_example_worker.py` | Current-view tool/evidence expectations in both corpora |
| Monitor task scope, hot sampling, retarget, failure, stop, participant isolation | `test_conversation_example_monitoring.py` | Monitor controls, active/stopped scope, route and evidence expectations |
| Note capture, readback, focus, release/handoff, overflow, context sharing | `test_conversation_example_worker.py`, `test_conversation_example_monitoring.py` | Note actions, focus transitions, and context-source expectations |
| Decision evaluator fixture loading, call limits, teacher-forced history, and not-scored output | `test_conversation_example_decision_eval.py` | Typed Clef route and control decisions |
| Sample wiring, readiness, Relay attribution and voice-stream boundaries | `test_conversation_example_worker.py` | Not scored unless the run captures those stages |

The ordered flow corpus separates raw router choice, effective destination,
invoked admission and control decisions, evidence/tool use, and answer checks.
A layer with no captured trace is reported as not scored.

Run the bounded decision-only pass from the repository root when the configured
Clef router is available:

```bash
uv --config-file uv.toml run --project agent-samples/conversation-example/worker \
  python agent-samples/conversation-example/eval/run_decision_eval.py \
  --output /tmp/conversation-example-decision-eval.jsonl
```

Add `--preflight-only` to validate fixture counts, required authored context,
and the decision-call cap without contacting a model. The JSONL output contains
typed decision requests/responses and per-turn raw choice, effective selection,
invoked native admissions/actions, and explicit not-scored fields. Histories
marked as authored are teacher-forced inputs; they are not model outputs. This
pass does not invoke answer generation, image tools, voice services, or
compaction, so those layers remain unscored. Per-turn `published_context_*`
fields record updates emitted by sample destinations; the decision-only runner
does not start an `AgentRuntime` subscriber and does not claim those updates were
consumed by a conversation prompt. Prior authored conversation completions seed
the same bounded private routing cue used after a completed production answer.
If a run stops after a fixture precondition fails, `--resume-from` can preserve
its raw decisions and continue untouched flows. It requires matching fixture,
model-config, and production-source hashes; the output records both evaluator
hashes when resume handling itself changed.

Flow histories are teacher-forced inputs. A seeded exchange is labeled as
authored state, not as an answer generated during the evaluation. Between turns,
focus and monitor state follow the actual native control decisions; fixture
expectations for those states are scored separately. The corpus's
answer checks are lexical smoke checks; they do not establish semantic answer
quality. Model-decision runs do not establish camera grounding unless they
include a completed frame/query result, and the deterministic image fixtures do
not reproduce a participant's live camera stream.

Router uncertainty still probes the highest-probability currently eligible
destination through its normal admission step. It produces router clarification
only when no usable eligible probability remains; a selected destination can
still ask its own clarification question during admission. The decision-only
evaluator follows the same distinction and excludes a default destination that
already declined the same request from fallback.

## Readiness

Before announcing readiness, VoiceAgent retries health probes for the configured
LLM, Clef router, and VLM. These probes do not generate an answer or warm up the
multimodal inference path. STT and TTS do not gate worker startup; start the
sample's local model-server stack before the sample so speech requests can
succeed.

## Configuration

Run and edit the sample from `agent-samples/conversation-example/`. The
orchestrator always passes `yaml/device_io_hub.yaml` to DeviceIOHub and
`yaml/conversation_example_worker.yaml` to the worker. The worker resolves its
models and voice-gate files relative to the worker YAML, so the checked-in
layout works without command-line configuration arguments.

| File | Owns |
|---|---|
| `yaml/conversation_example_worker.yaml` | Frame freshness and wait limits, VAD, idle timeout, optional context topics (including note sharing) and byte budgets, route-aware acknowledgment delay, and prompt overrides |
| `yaml/voice_gate.yaml` | Conversation controls, optional wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | LLM/VLM/STT/TTS/Clef adapters, endpoints, and conversation generation defaults |
| `yaml/device_io_hub.yaml` | LiveKit room and ports, web and token servers, and network behavior |
| `yaml/media_capture.yaml` | Opt-in media-hub capture, NVENC output, caption layout, and retention |
| `worker/conversation_example_worker/prompts/system.txt` | Default VLM instruction |

Edit the owning file, preserve the field's YAML type, and restart
`conversation_example`; configuration is loaded only at process startup. For
example, lower `silero_threshold` in the worker YAML if quieter speech is being
missed, or change `conversation.start_phrase` and `conversation.stop_phrase`
in the voice-gate YAML. Set `conversation.require_wake_phrase: true` to require
its configured `magic_phrases` during an active conversation. Set `mode: hybrid`
to also accept wake-addressed one-off questions while closed, or `mode: wake_only`
to use wake-addressed turns without a persistent conversation. Refer to
{ref}`voice-conversation-controls`. Relative paths in the worker YAML are resolved from `yaml/`, not
from the shell's current directory.

The optional `acknowledgment_delay_secs` setting enables route-aware intention
acknowledgments when set to a positive number; it is disabled when omitted.
The delay is measured from accepted destination execution until its first
nonblank answer text is ready for emission, excluding routing, admission, and
any acknowledgment-output serialization. Confident predictions of a slower
answer can trigger an acknowledgment earlier; uncertain predictions use the
configured delay as a fallback. Acknowledgments are best-effort and do not
guarantee that downstream audio has not already been queued or played. The
sample uses a separate typed client for acknowledgments, with reasoning disabled
and a 64-token output limit; the conversation model settings are unchanged.
Acknowledgment text is restricted to a short “Let me think through …” goal
restatement. Nonconforming generated text is dropped without retrying.

The default `summary_context_topics` list includes
`conversation-example.monitor-window`, `conversation-example.foreground-notes`, and
`conversation-example.observation-capability`. Each destination owns its text
and publishes an initial participant-specific snapshot when the participant
joins, then replaces it when its reported state changes. The monitor reports
whether it is active and its timestamped observation window; notes report
whether note-taking is active and saved notes; direct observation describes
its fresh-frame capability without claiming that a frame is currently being
observed or that the camera is ready. These summaries are context, not health
checks. Remove a topic from the list to disable that feed; focused note capture
and readback continue to use the note destination's local state. Use
`incremental_context_topics` for a source whose publisher sends append-only
updates that the conversation compacts.

Changing an entry in `models.json` changes only the client adapter or endpoint
that this sample uses. It does not reconfigure or restart the shared server.
For a checkpoint, port, GPU, or model-runtime change, refer to
{doc}`/guides/customizing-model-servers`, update the sample model stack, stop that
persistent stack, and start it again before restarting this sample.

Refer to the generated {doc}`configuration <configuration>` reference for exact
fields, checked-in values, and adjacent YAML comments.

After connecting, say `Hey agent, let's start talking`, then ask questions
without a wake phrase. Say `Hey agent, let's stop talking` to finish. These
controls work with the sample's existing STT endpoint, including NIM. Refer to
{ref}`voice-conversation-controls` for paused phrases and the legacy wake mode.

## Opt-in session capture

Capture is disabled by default. Run `uv run conversation_example --session-recording` to
start `device_io_capture` immediately after DeviceIOHub and record normalized
hub media without joining the LiveKit room. Each participant connection then
creates a bundle under
`~/.local/share/xr-ai/captures/conversation-example/` containing a canonical raw
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
Omit `--session-recording` when a deployment must not retain device media.

Wake phrases match at the start of a final transcript or after sentence-final
`.`, `?`, or `!` punctuation followed by whitespace or a closing quote. Text
before that boundary and the phrase itself are discarded before dispatch.

## Relay output

The worker writes `relay-events.jsonl` beside `worker.log` in the per-run log
directory printed at startup. It records runtime publications, receiving-agent
callbacks, the complete `conversation-example.turn` lifetime, and nested vision and model
calls. Per-token model marks, incremental voice fragments, and empty stream
terminators are omitted. Voice and STT each emit one semantic scope for a
completed operation. TTS emits one ``voice.tts`` scope for each sentence sent
to synthesis. Raw audio is represented only by size, duration, and sample rate.

Image locations appear as `<redacted:image>`. Prompts, questions, responses,
participant identities, and correlation metadata remain visible and may be
sensitive. Inspect a running sample with:

```bash
tail -F /tmp/log_conversation-example_*/relay-events.jsonl
```

Model HTTP diagnostics are also enabled at DEBUG. `worker.log` records the
redacted request payload, provider response, and each raw streaming event,
including reasoning fields when the endpoint returns them. The per-run file
sink captures DEBUG by default; set `XR_AI_VERBOSE=1` before launch to show
these records in the terminal as well:

```bash
rg 'llm\.trace' /tmp/log_conversation-example_*/worker.log
```

The model trace redacts credentials and inline media, but text prompts,
questions, reasoning, and generated responses remain visible in the worker
log. Treat that file as sensitive diagnostic output.
