<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Conversation example

**TL;DR:** Start here to build a voice-and-vision application that can keep a
conversation going while helping with what a participant sees. The sample
shows how specialized agents can share an interaction without losing track
of the participant's current task. It is the recommended starting point for
new applications in place of the deprecated {doc}`simple-vlm-example`.

The conversation example combines four distinct roles: general conversation,
direct questions about the current view, one-shot background visual alerts, and
foreground note-taking. An alert watches for a requested visible condition,
notifies the participant once, then stops. Its observations stay private to the
alert job. Notes capture participant-authored text and can share saved notes
with conversation.

The sample composes these roles with `xr_ai_routing.AgentRouter` and
`xr_ai_conversation.Conversation`. Responses use the same voice output path to
Pocket TTS and the `vlm.response` data topic. Refer to the {doc}`quickstart
</getting_started/quickstart>` for client setup and the sample README for run
commands.

## Runtime boundary

`AgentRouter` and `VoiceAgent` are registered directly with the same
`AgentRuntime`. The sample configures the router's query, participant lifecycle,
interruption, and voice output topics. The router owns routing tasks, focus,
shared turn history, acknowledgments, and route-error responses. Each destination
publishes its own attributed `ResponseOutput` messages. Conversation and direct
observation stream chunks; note-taking and alert controls publish complete
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

`app.py` constructs and registers the router, voice agent, and conversation
context subscriber. It supplies the router's participant cleanup hook, which
clears the sample's notes, alerts, conversation context, and frame resources.

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
health and waits for the required speaker STT service before opening the voice
session; readiness does not send an inference request through the VLM.

`VoiceAgent` owns worker readiness, hub transport, voice gating, TTS, signals,
and cleanup. It publishes accepted speech and typed text as a participant-scoped
`UserQuery`. `xr_ai_routing.AgentRouter` subscribes directly to the configured
query topic. Clef routes among general conversation, direct visual observation,
visual alert controls, and foreground note-taking. General conversation answers
text questions and recall. Direct observation sends one fresh image and the
current question to Cosmos, without dialogue history.

An alert starts only when the participant explicitly asks to be notified when a
specific visible condition occurs, stated directly (“Alert me when the left
indicator is green.”) or referring to one given earlier (“Alert me when you see
what I mentioned in my notes.”). Before a job starts, the agent compiles the
request with one LLM call that sees the request, the turn's recent exchanges, and
the general conversation's saved context, including shared notes. The result is
a standalone condition and a short relevance phrase naming what the condition is
about. The spoken confirmation states the compiled condition. When no specific
visible condition can be resolved, the agent says so, starts nothing, and leaves
any running alert in place.

Each job then checks frames in a paced loop. A cheap Clef image call screens every
frame for whether the relevance subject may be visible, and a frame it rejects
costs nothing more. Only a screened frame runs the two condition checks in
parallel: a direct Clef image decision and a Cosmos observation that Clef judges
as text. The first yes owns the triggering frame; the other check is cancelled,
and its cleanup finishes before only one complete “Alert” voice update is
published as urgent. The agent then asks Cosmos for a short
description of relevant evidence from that same frame and publishes it as one
non-urgent update once the model has finished, then stops. The description does
not cut the announcement.

Checks are paced: the next frame waits at least `alert_check_interval_secs`
(default 1) after a screen, and `alert_confirm_interval_secs` (default 3) after a
screened frame, which ran Cosmos. A failure in one condition check leaves the
other available; if both fail, or the screen or frame capture fails, the job
retires with a concise failure message. Only an explicit new alert request
replaces an existing job. “Cancel the visual alert” cancels it. Cancelling or
replacing an alert also withdraws its announcement or description if voice output
has not started speaking it. Ordinary
questions and fragments are not alert controls. Each participant owns an
independent job; departure and worker shutdown cancel its work and wait for that
cleanup to finish, even if they are cancelled themselves while waiting.

The backup judge sees Cosmos's text description rather than the image, so it
can miss evidence or repeat a description error. Neither path guarantees visual
recognition accuracy. A single still image cannot establish motion, a change
over time, or a previous state; conditions requiring that evidence do not
trigger.

Conversation's instructions describe what alerts can do; conversation never
receives a job's observations or matching state. The alert condition and current
image stay inside the background job.

Foreground notes start explicitly, capture dictated text, and retain focus until
a finish. A question handoff declines only that query so another destination can
answer; the next turn returns to note capture. Every saved note is kept until
the participant leaves and is read back in full, also after the session
finishes, without restarting it. Each one is also shared
with general conversation. The notes routing description is static, so the
router's request does not grow with saved notes.
A decline gives the router one bounded reroute; unresolved routing clarifies or
uses the eligible default conversation destination. Alerts do not take focus
and can run while a participant is taking notes or asking another agent a
question.

Notes admission classifies the current utterance and whether notes have focus.
Saved content stays available for readback and shared context; completed dialogue
is not input to this action decision. Ordinary statements are dictation while
notes are focused. Questions seeking an answer go to another agent, while an
explicitly dictated question is saved verbatim. The note agent acknowledges a
save only after storing it; it does not generate general replies that could
claim an unsaved note was recorded.

Direct visual requests must identify their target in the current question,
because dialogue history is not sent to Cosmos. Admission does not establish
that a camera frame will be available. The
`CurrentFrameTool` uses
the hub's fresh frame when DeviceIOHub is already observing video for that
participant. When video is off, the same tool asks StreamKit to capture one
image and return it through a targeted LiveKit byte stream. The fallback is
invisible to the agent: it receives the same opaque image-reference shape and
passes it to the configured visual model. Direct visual answers use Cosmos;
alerts screen each frame with Clef, run direct Clef decisions and a concurrent
Cosmos-observation-to-Clef-text path only on screened frames, and add one Cosmos
description only after a match.
Camera bytes remain on the hub path, image locations are redacted from VLM
telemetry, and the sample has no MCP path.

Direct observation captures only when a question needs an image. A separately
requested alert samples until its condition matches, it is cancelled, or an
observation fails. Neither requires the participant to publish continuous video;
the frame tool can request a still when needed.

A newer participant turn cancels superseded routing, answer, and vision work and
interrupts its voice response. Participant departure releases cached frames,
conversation destination state, AgentRouter history, and source-context
tasks. This is the reference composition for streamed conversation and
current-view visual questions.

The application owns the general conversation's `ContextManager` and clears it
when a participant leaves. It holds two sections: completed dialogue and the
`conversation-example.notes` incremental topic, on which the notes destination
publishes each saved note as one text-only `ConversationContextUpdate`. Each
section keeps recent entries verbatim and folds older ones into a rolling
summary written in the background per participant, so conversation and alert
compilation see recent notes exactly and older notes summarized. When too much
uncompacted context is present, the oldest updates are omitted from an answer
request but retained for later compaction. `context_budget_bytes` is split
evenly between the two sections; with the default 12,000 bytes, each gets 6,000.
`compaction_input_budget_bytes` separately limits one summary request; its
default is 12,000 bytes. These are UTF-8 byte budgets, not token limits.

Conversation prompts show the current time and coarse minute-rounded timestamps
for completed exchanges and named context where available. Completed
conversation turns use their completion time, runtime context uses its
publication time, and visual evidence includes the frame's source timestamp.
Imported history without a time
is labeled unknown. These timestamps help the model distinguish older context
from a present observation; they do not expire or remove information.

## Source map

The worker package is under
`agent-samples/conversation-example/worker/conversation_example_worker/`:

| File | Responsibility |
|---|---|
| `__main__.py` | Parses launcher arguments and starts the worker |
| `app.py` | Composition root for `VoiceAgent`, `AgentRouter`, route agents, services, and readiness |
| `_topics.py` | Names the sample-owned voice input and lifecycle topics |
| `config.py` | Resolves worker, model, voice-gate, and prompt settings |
| `prompts/note_control.txt` | Note action instructions for dictation, question handoff, readback, and finish |
| `_note_taking.py` | Implements the sample-owned focused note-taking destination |
| `_alert_agent.py` | Owns one-shot background visual alerts: requests, the paced watch, announcement, and job lifecycle |
| `_alert_compile.py` | Compiles an alert request and the participant's context into a watch |
| `_alert_checks.py` | Screens a frame for relevance and races the two condition checks on it |
| `_decisions.py` | Asks Clef one traced question and rejects a choice outside its criteria |
| `_tasks.py` | Waits for owned tasks without letting caller cancellation interrupt their cleanup |
| `_visual_agents.py` | Implements direct current-image observation |
| `prompts/system.txt` | Defines the default VLM instruction |
| `prompts/alert_control.txt`, `prompts/alert_compile.txt`, `prompts/alert_relevance.txt`, `prompts/alert_condition.txt`, `prompts/alert_observation.txt`, `prompts/alert_observation_judge.txt`, `prompts/alert_description.txt` | Start/cancel admission, request compilation, frame screening, direct and observation-based condition checks, and the triggered description |
| `prompts/conversation.txt` | Application instructions for the conversation destination |

The decision eval is under `agent-samples/conversation-example/eval/`:

| File | Responsibility |
|---|---|
| `run_decision_eval.py` | Runs every flow through the router against the configured Clef service and fails below a pass rate |
| `conversation-flows.yaml` | Ordered participant sessions with expected routing, admission, control actions, and focus |

## Behavior validation

The sample's deterministic tests divide the contract by layer. Run them from the
repository root with the `tests` project environment:

```bash
uv --config-file uv.toml run --project tests pytest \
  tests/test_xr_ai_conversation.py \
  tests/test_xr_ai_routing.py \
  tests/test_conversation_example_worker.py \
  tests/test_conversation_example_visual_destinations.py \
  tests/test_conversation_example_agents.py \
  tests/test_conversation_example_alerts.py \
  tests/test_conversation_example_decision_eval.py
```

| Behavior | Deterministic contract |
|---|---|
| Routing, native admission, decline/reroute, default fallback, focus, cancellation | `test_xr_ai_routing.py` |
| Completed history, named context, byte budgets, compaction, acknowledgment races | `test_xr_ai_conversation.py` |
| Current-frame-only observation and camera/tool failures | `test_conversation_example_visual_destinations.py`, `test_conversation_example_worker.py` |
| One-shot matching, failure, cancellation, participant isolation | `test_conversation_example_alerts.py` |
| Note capture, readback, focus, release/handoff, context sharing | `test_conversation_example_worker.py`, `test_conversation_example_agents.py` |
| Decision eval scoring with fixture-faithful and wrong decisions | `test_conversation_example_decision_eval.py` |
| Sample wiring, readiness, Relay attribution and voice-stream boundaries | `test_conversation_example_worker.py` |

The decision eval scores the routing and control decisions that the configured
Clef model makes. It sends every turn of `eval/conversation-flows.yaml` through
`AgentRouter.handle()` with the sample's four destinations. Answers are
teacher-forced: the conversation LLM and the observation VLM return each turn's
authored completion, which later turns see as completed history. Each turn
checks the destination that answered, the router's first choice, named
admissions and native control actions, the answer's disposition, and router
focus. A turn passes when every check it specifies passes; the run exits
nonzero when fewer than 80% of turns pass. Run it from the repository root when
the configured Clef router is available:

```bash
uv --config-file uv.toml run --project agent-samples/conversation-example/worker \
  python agent-samples/conversation-example/eval/run_decision_eval.py \
  --output /tmp/conversation-example-decision-eval.jsonl
```

`--output` writes one JSON record per turn with its checks and every decision
choice. Answer quality, alert compilation, and the alert frame checks are not
scored: alert compilation returns a fixed watch, and alert jobs never receive a
frame.

Router uncertainty still probes the highest-probability currently eligible
destination through its normal admission step. It produces router clarification
only when no usable eligible probability remains; a selected destination can
still ask its own clarification question during admission. A default
destination that already declined the same request is excluded from fallback.

## Readiness

Before announcing readiness, VoiceAgent retries health probes for the configured
LLM, Clef router, and VLM. These probes do not generate an answer or warm up the
multimodal inference path. Because `yaml/voice_gate.yaml` sets
`speaker.backend: required`, VoiceAgent also waits for the speaker STT service
before announcing readiness. The sample uses no batch STT client, and TTS
health does not gate startup; start the sample's local model-server stack
before the sample so speech requests can succeed.

## Configuration

Run and edit the sample from `agent-samples/conversation-example/`. The
orchestrator always passes `yaml/device_io_hub.yaml` to DeviceIOHub and
`yaml/conversation_example_worker.yaml` to the worker. The worker resolves its
models and voice-gate files relative to the worker YAML, so the checked-in
layout works without command-line configuration arguments.

| File | Owns |
|---|---|
| `yaml/conversation_example_worker.yaml` | Frame freshness and wait limits, VAD, idle timeout, conversation context and routing history byte budgets, route-aware acknowledgment delay, alert check and confirm intervals, and prompt overrides |
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

`routing_history_budget_bytes` (default 8,000) limits the completed history the
router sends with each routing decision. Clef accepts at most 4,096 tokens per
request, and the routing prompt and the four destination descriptions use about
800 of them, so the default leaves room for history that tokenizes less
efficiently than English. Raise it only together with the Clef service's
`max_length`.

Changing an entry in `models.json` changes only the client adapter or endpoint
that this sample uses. It does not reconfigure or restart the shared server.
For a checkpoint, port, GPU, or model-runtime change, refer to
{doc}`/guides/customizing-model-servers`, update the sample model stack, stop that
persistent stack, and start it again before restarting this sample.

Refer to the generated {doc}`configuration <configuration>` reference for exact
fields, checked-in values, and adjacent YAML comments.

After connecting, say `Hey agent, let's start talking`, then ask questions
without a wake phrase. Say `Hey agent, let's stop talking` to finish. These
controls work with the sample's configured speaker-ASR service. To switch to
native NIM speech, update both the model profile and the voice-gate backend as
described in {doc}`/guides/deploying-with-nim`. Refer to
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
