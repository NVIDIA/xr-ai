<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance — building deterministic visual workflows

## Overview

Use this reference for procedural assistants that combine spoken guidance with
visual checks of task progress. Tea making is the example: the application
controls the steps and completion rules, while models interpret observations
and answer questions. Adapt the workflow and its tools to guide other
procedures.

To run the sample, refer to the
{doc}`quickstart </getting_started/quickstart>`.

For speech deployment and explicit batch-STT selection, refer to
{doc}`/guides/customizing-model-servers`.

YAML declares the procedure, evidence criteria, tool arguments, and allowed state
updates. Omni owns visual observation and label identity extraction.
Clef makes bounded choices from those descriptions and from the current user
request. Code owns confirmation counts, arithmetic, timestamps, and guarded
state commits. The language model generates answers and summaries; it receives
no state-write tool.

The focused SOP destination owns its execution engine, managed context, request
handling, and guidance notice delivery. The tea guide is one focused destination
on its top-level `AgentRouter`, like
the conversation sample's note-taking destination. Steps are data, not agents.
There are no routers inside a step.

## What to copy

Reusable patterns include:

- a focused workflow agent with state-scoped admission choices;
- a guided procedure whose evidence comes from current visual observations;
- sparse typed state updated through one controlled commit boundary;
- automatic progression after confirmation, with a manual advancement option;
- independent, opt-in visual, transcript, and video background tasks;
- bounded background facts available to foreground questions;
- typed events consumed separately by speech, files, or a customer backend.

Tea names, temperatures, timers, and step prompts are sample-specific. The
workflow state machine, foreground routing, observation loop, background-agent
ownership, and event composition are reusable.

## Quick start

Run commands from `agent-samples/tea-making-sample/`. The tea sample reuses all
model services, so start the sample's model stack, including Pocket TTS and Clef, first:

```bash
uv run --project model-servers tea_making_sample_model_servers
```

Wait for the model launcher to report readiness and return. Then launch the
sample from the same terminal:

```bash
uv sync
uv run tea_making_sample
```

Open the authenticated web-client URL printed by DeviceIOHub, accept the
development certificate, allow camera and microphone access, and connect.
Conversation mode is the default. Say “Hey agent, let's start talking” once,
then try:

```text
Help me make tea.
What is the current step?
What do you see?
Start watching for spills.
Start recording the transcript.
```

Pass `--expose-web-events` to bind the unauthenticated live event viewer to all
IPv4 interfaces instead of loopback. Refer to the generated
{doc}`command-line <command-line>` reference for all launcher options.

Speech uses the reused TTS endpoint in `yaml/models.local.json`. Voice-gate
behavior comes from `voice_gate_yaml` in `yaml/tea_making_worker.yaml`.
The default requires `Hey agent, let's start talking` after connecting,
then accepts speech without a wake phrase until `Hey agent, let's stop talking`.
Set `mode: hybrid` to also accept wake-addressed one-off questions before a
conversation starts, or `mode: always_on` for immediate, ungated speech dispatch.
The advanced `conversation.require_wake_phrase: true` policy applies only inside an open
conversation. Backend selection is independent of these modes. Refer to
{ref}`voice-conversation-controls`.

The live event viewer is available at `http://127.0.0.1:8093` on the XR-AI
host. With `--expose-web-events`, use `http://<xr-host>:8093` from a trusted
development network and restrict TCP port 8093 accordingly. Durable JSONL
records remain under `agent-samples/tea-making-sample/artifacts/`; the viewer
is only a bounded view of the current worker process.

## Architecture

YAML declares the workflow steps and state contract. A VLM describes each
current view, and the observation LLM judges that caption and selects native
tools. Application code validates the resulting state proposal and counts any
required consecutive confirmations. The model does not choose which workflow
is active, mutate arbitrary state, or advance a step implicitly.

```text
camera frames ──> ParticipantImageAgent ──> SOP agent
                                             ├─ execution engine + WorkflowStore
speech / typed query ────────────────────────> ├─ foreground request handling
                                             ├─ ContextManager + compaction
                                             └─ guidance notice delivery
                                                        │
                  independent background observers      │
                               │                        v
                               └──────────────> typed runtime events
                                                 ├─> FileOutputAgent
                                                 ├─> TeaWebEventsAgent
                                                 └─> customer backend
```

`VoiceAgent` publishes participant-scoped queries, transcripts, and lifecycle
events. The runtime fans typed events out to peer agents. Agents call peer
tools directly, and each background agent owns its participant tasks.
The worker uses native `xr_ai_runtime` agents and `xr_ai_tools` instances; it
does not use NVIDIA Agent Toolkit, PydanticAI, MCP clients, or MCP servers.
The sample accepts speech without a wake phrase during an active conversation;
callers can enable its configured wake phrases. Every spoken response is also
sent to the connection client on `agent.response` so the client can render
accessible text alongside audio.
Streamed responses are finalized there as soon as their content completes, and
combined updates as soon as their rewrite completes. An urgent barge-in can
still stop audio after the full intended utterance has reached the client.

## Agent responsibilities

| Agent | Owns | Does not own |
|---|---|---|
| `ParticipantImageAgent` | Image registry, current-frame selection, participant cleanup | VLM prompts or workflow state |
| `SOPAgent` | Execution engine, managed context, foreground request handling, and notice delivery | Model service or image-source lifetime |
| `BackgroundContextAgent` | A bounded participant-local set of recent background facts | Background scheduling |
| `ChangeWatchAgent` | Focused visual-change task and deduplication | Foreground routing |
| `TranscriptAgent` | Opt-in transcript records and periodic summaries | Voice gating or STT transport |
| `VideoLogAgent` | Periodic captions and material deltas | Workflow state |
| `FileOutputAgent` | Participant session files and JSONL records | Agent policy |
| `TeaWebEventsAgent` | Explicit compact projections for the generic live viewer | Durable storage or file polling |
| `WebEventsAgent` | Bounded HTTP presentation grouped by participant and topic | Application event selection |

The runtime delivers events but does not absorb application state or
concurrency. Moving a task to another agent also means moving its cancellation,
locks, and participant cleanup.

## Sample-local APIs and prompt ownership

Import `SOPAgent`, `Workflow`, and `load_workflow` from `tea_making_worker`.
`SOPAgent` implements the SDK routing destination contract and subscribes to
guidance notices. The application creates one top-level SDK `AgentRouter` and
registers its `runtime_subscriber()`. The agent and its owned execution/state/tools
and run-context projection live in `agents/sop/`. The existing `sop` package keeps
the workflow-loading exports; reusable evidence flows remain independent in `flows/`.

The SDK `Agent` owns tools/state and can subscribe to runtime events. Its separate
`RoutableAgent` contract handles admission, answers and focus. `SOPAgent` fills both
roles: it is one focused destination and receives guidance notices. Its private
`_WorkflowEngine` is an ordinary helper, not a second runtime agent. Workflow steps
also remain ordinary reusable flows, rather than independent agents.

Existing watcher, transcript, journal, background-context and I/O runtime agents
live under `agents/`. Small optional routed agents in `agents/_destinations.py`
receive explicit services/tools/policy and the existing answer/output callbacks;
they do not use the SOP as a dependency container. Shared answer, semantic-choice
and compaction-recording helpers remain in `conversation/`. There is one top-level
router and no additional agent hierarchy or conversation supervisor.

`tea_making_worker.flows` exports `VisualValidationStep`, `VisualCheck`,
`IdentityResolutionStep`, `ObservationPolicy`, and `Observation`. Visual validation
accepts a question, policy, configured observer, typed services and tools. Its
`observe`, `evaluate`, and `check` methods return evidence without owning frame
acquisition, confirmation counts, state mutation, or storage. Identity resolution
accepts established fields and configured retrieval/fallback policy. Optional
identity event recording is supplied by the caller. These modules depend on typed
model/tool APIs rather than tea configuration or the SOP execution engine.

Task-specific wording remains in external `workflow.yaml`: visual questions under
`trigger.arguments.question`, extraction or judgment instructions under
`observation`, retrieval/fallback instructions under `observation.lookup`, and
step dialogue under `voice.prompt`. JSON shapes, parsing, unit normalization,
provenance verification, native transitions, and context-channel authority remain
implementation contracts.

Generic model behavior can also need tuning. Keep that tuning with its owner:

| Owner | Configuration | Tunable behavior |
|---|---|---|
| Shared conversation | Worker `foreground_prompt` | Shared response style and grounding, applied to both general and active conversation |
| Active workflow conversation | `task.foreground.active_prompt` | Workflow role, response detail, spoken style and length; appended to shared and step instructions |
| Shared camera interpretation | Worker `current_view_prompt` | Camera viewpoint, visual detail and evidence description; shared by observation and immediate camera tools |
| Focused change captions | Worker `change_watch_caption_prompt` | Detail and focus for the user-requested watcher |
| Change judgment and summary | Worker `change_watch_decision`; `change_watch_event_prompt` | What counts as a relevant transition and how accepted changes are described |
| Visual journal captions | Worker `video_caption_prompt` | Scene detail, readable identifiers and uncertainty |
| Journal judgment and summary | Worker `video_delta_decision`; `video_delta_prompt` | Material-change criteria and accepted delta wording |
| Background transcript summary | Worker `transcript_summary_prompt` | Retained facts, attribution and summary detail |

All application-level model instructions and background decision mappings are
inline in `yaml/tea_making_worker.yaml`. There is no separate prompt directory,
prompt-file option, or Python prompt fallback. Literal YAML blocks preserve the
model-facing text and line breaks. Start a custom worker configuration by copying
the complete main YAML; it must supply its prompt text and decision policies.

Without `--config`, the source worker reads that checked-in YAML. An installed
worker started from the documented sample directory reads `yaml/tea_making_worker.yaml`
there; otherwise provide its path with `--config`. Paths resolve relative to the
selected YAML. Workflow-specific questions, criteria, RAG/fallback instructions,
and voice policies remain in the external `workflow.yaml` selected by
`workflow_config`. The default spoken style is declared there as well.

Background transcript summaries are separate from dialogue/run compaction.
`ContextManager` continues using its existing SDK compaction defaults; its
recording adapter does not redefine summarization policy. Background prompts stay
at application scope rather than being copied into each SOP step.

## Source map

| File | Purpose | Typical adaptation |
|---|---|---|
| `app.py` | Constructs services and registers agents | Add a backend or another observer |
| `events.py` | Typed application topics and payloads | Define domain facts, notices, or records |
| `agents/sop/_spec.py` | Validates the workflow YAML contract | Extend declarative workflow metadata |
| `agents/sop/_state.py` | Typed sparse state, evidence, transitions, rendering | Change deterministic state policy |
| `agents/sop/_tools.py` | Current view, RAG, clock, temperature, and native workflow controls | Add domain tools |
| `agents/sop/agent.py` | Focused SOP destination, native control admission, common answer path and run context | Compose a different procedure |
| `agents/sop/_context.py` | Per-run state/evidence/event channels and historical handoff | Configure context ownership |
| `conversation/_capture.py` | Participant-scoped compaction recording | Record asynchronous context work |
| `agents/sop/_engine.py` | Internal execution engine (`_WorkflowEngine`), sessions and observation loop | Add another trigger or evidence source |
| `flows/_visual.py` | Reusable, read-only visual observation/evaluation | Configure semantic or numeric validation |
| `flows/_identity.py` | Identity-to-reference settings resolution with explicit fallback provenance | Configure another identification-and-lookup flow |
| `conversation/_answer.py` | Shared message assembly and SDK read-only tool loop | Change answer instructions |
| `agents/_destinations.py` | Explicit adapters for the existing background, camera and reference capabilities | Configure application behavior |
| `_cleanup.py` | Original-generation producer acknowledgments after SDK-owned departure | Preserve file/image cleanup ordering |
| `conversation/_routing.py` | Bounded semantic choice and exact-operation authorization | Tune configured criteria |
| `flows/_observations.py` | Typed extraction, choice evidence, unit normalization, and deterministic evaluation | Adapt observation policy |
| `agents/images.py` | Shared participant image acquisition | Add another image selector |
| `agents/background_context.py` | Recent-fact store and query tool | Change foreground and background context policy |
| `agents/change_watch.py` | Focused visual observer | Build another event-oriented monitor |
| `agents/transcript.py` | Transcript recording and summarization | Forward or classify laboratory dialogue |
| `agents/video_log.py` | Periodic caption and delta observer | Build a visual journal |
| `agents/file_output.py` | Participant JSONL artifacts | Replace files with backend persistence |
| `agents/web_events.py` | Explicit typed event-to-view projections | Select or reshape live presentation topics |
| `yaml/tea_making_worker.yaml` | Inline app-level model instructions and background decision policies | Tune generic conversation, perception and summaries |
| `yaml/workflow.yaml` | Five-step procedure and evidence contract | Define a different guided task |
| `rag-documents/` | Sample retrieval corpus | Replace with domain documents or a backend |

Refer to the generated {doc}`configuration <configuration>` and
{doc}`command-line <command-line>` references for configuration fields and
command syntax.

## SOP run capture

The worker records an SOP bundle on every launch under
`artifacts/runs/<UTC timestamp>-<run ID>/`. `capture.enabled: true` in
`yaml/tea_making_worker.yaml` enables structured attempt and image capture by
default. Set it to `false` to disable those records; the run manifest,
configuration snapshots, participant JSONL and supplemental Relay log remain.
No model-server restart is needed when changing this worker setting.

```text
runs/<run>/
  manifest.json
  config/
    worker.json
    workflow.json
    resolved_workflow.json
    models.json
    voice_gate.json
  attempts.jsonl
  images/<SHA256>.jpg
  participants/<participant>/steps/<YAML step>/<channel>/<attempt>/events.jsonl
  participants/<participant>/applications/<channel>/<attempt>/events.jsonl
  <participant-session>/guidance.jsonl
  <participant-session>/foreground.jsonl
  relay-events.jsonl
```

Each observer attempt belongs to the current YAML step ID. Participant and step
path components include a short hash to avoid collisions and unsafe filesystem
names; the original IDs remain in every record. Foreground turns belong to the
step active when the turn starts, or to `applications` when no step is active.
Independent watchers and journals also use `applications`. Attempt IDs remain
unique across reconnects and procedure resets within a worker run.

Routing attempts retain SDK destination selection, admission and exact-operation
authorization decisions. The existing router processing/idle hook opens and
closes recording inside the SDK-owned task, using its public Relay metadata for
the original request ID. This hook does not supervise or cancel work. Routing
and direct-answer attempts share that request ID; both retain explicit failure
and cancellation status.


`attempts.jsonl` is the run index: each entry contains participant, step, channel,
request and attempt IDs, plus the relative event-file path. Attempt event files
contain the initial state, revision and evidence count; original frame metadata;
visual question and system instructions; model responses; Clef inputs,
criteria and output probabilities; native trigger and commit results; final
state; and completion, error or cancellation status. Generative answers and
their input messages are recorded separately from visual judgments. Model
revisions are retained when the service reports them; a configured endpoint
alone does not prove which deployed checkpoint served a request.

Image files contain the encoded bytes passed to the vision service, saved before
dispatch. SHA256 addressing deduplicates identical bytes within a run. References
in `vision.started` identify those files, their size and hash; associated frame
metadata retains source track, sequence, capture time and dimensions. No image
bytes are inserted into Relay events or client messages. This records consumed
inference images rather than every camera frame, and does not require
`--session-recording`.

Configuration snapshots include the source workflow structure and resolved
policies/prompts. The manifest stores their hashes and worker-source hashes.
Known credential fields are redacted and request headers are omitted. The
bundle remains local and includes participant images and utterances. Runs are
never automatically deleted or overwritten; manage disk space and remove
unneeded runs explicitly.

Use the exact attempt ID when recording review notes, for example an
`annotations.jsonl` file beside the run index with entries such as
`{"attempt_id":"...","expected_evidence":"rejected","note":"empty cup"}`.
This keeps human ground truth separate from model output. Review the image and
YAML snapshot before attributing a failure to captioning, semantic judgment,
source selection or native state handling. An attempt with no finish event, or a
manifest still marked `running` after process loss, is incomplete.

Recording uses serialized file writes outside the event loop. Cancellation
drains writes already started before cleanup; streamed answers retain partial
text and an explicit incomplete flag. Normal shutdown closes attempt records
before finalizing the manifest. File errors surface through worker/model-boundary
errors rather than silently claiming successful image capture. Saving the input
adds disk latency before inference; no extra model calls or GPU models are
introduced. Relay traces remain supplemental and are not required to recover
the SOP evidence/state record.

## Opt-in continuous session capture

Continuous session capture is disabled by default. Run `uv run tea_making_sample --session-recording`
from `agent-samples/tea-making-sample/` to start `device_io_capture`
immediately after DeviceIOHub. It records participant video, device and agent
audio, and inbound and outbound data in a separate process. The option can be
combined with `--expose-web-events`.

`yaml/media_capture.yaml` selects the `demo` profile and participant-lifetime
sessions. Each connection creates a raw bundle under
`~/.local/share/xr-ai/captures/tea-making-sample/`. When the session closes,
the capture renderer produces a captioned NVENC H.264 MP4 with timestamp-aligned
AAC-LC audio. Raw media and the exact data timeline are retained. Demo encoding
requires an NVIDIA GPU with NVENC support and `ffmpeg` on `PATH`.

Capture uses bounded queues and drops pending capture frames if encoding falls
behind, so recording does not backpressure the hub or worker. Omit `--session-recording`
to disable continuous media recording. Disable `capture.enabled` as well when
inference images must not be retained. Refer to
{doc}`../components/server-runtime` for the capture bundle and renderer contract.

For a captioned MP4, disconnect the participant and wait for
`video/session.mp4` and the manifest's `renderings.captioned_mp4` entry before
stopping the stack. Shutdown finalizes active raw bundles without starting new
renders. Refer to {doc}`../components/server-runtime` for rendering a completed
raw bundle separately.

## Configuration

Run and edit the sample from `agent-samples/tea-making-sample/`. The
orchestrator reads the checked-in files on every start and materializes a
temporary worker configuration with absolute paths. Edit the files under
`yaml/`, not the temporary copy named in the logs.

| File | Owns |
|---|---|
| `yaml/tea_making_worker.yaml` | Inline app-level prompts and decision policies, subordinate configuration paths, timeouts, background cadence, VAD, output directory, and event viewer |
| `yaml/workflow.yaml` | Typed state, steps, triggers, evidence, tool access, transitions, and user messages |
| `yaml/voice_gate.yaml` | Interaction mode, conversation controls and optional wake phrases |
| `yaml/rag_service.yaml` | Document path, model configuration, cache, chunking, embedding dimensions, and score threshold |
| `yaml/models.local.json` | Model adapters and shared endpoints, including RAG embeddings |
| `yaml/device_io_hub.yaml` | LiveKit room and ports, web and token servers, and network behavior |

Set the interaction mode directly in `voice_gate.yaml`; `voice_gate_yaml` in
`tea_making_worker.yaml` selects that file. Change observation and summary frequency in that
worker file, retrieval behavior in `rag_service.yaml`, and the procedure itself
in `workflow.yaml`. Relative paths declared in each configuration file resolve
from that file's directory. Workflow state and step references are validated at
startup, so an invalid contract fails before a participant session begins.

Restart `tea_making_sample` after an edit; configuration is not hot-reloaded.
The `--expose-web-events` option intentionally overrides `web_events_host` in
the runtime copy. Changing `models.local.json` changes the endpoints used by
the worker and RAG service but not the persistent servers. Refer to
{doc}`/guides/customizing-model-servers` for a server-side model, GPU, port, or
memory change, then restart the sample's model stack.

The local embedding server uses `nvidia/Nemotron-3-Embed-1B-BF16`, served as
`embed` on port 8109. RAG retains its native 2048-dimensional vectors. The
external NIM profile uses `nvidia/nemotron-3-embed-1b` with the same dimension;
refer to {doc}`/guides/deploying-with-nim` for endpoint configuration.

Refer to the generated {doc}`configuration <configuration>` reference for exact
fields, checked-in values, and adjacent YAML comments.

<a id="deterministic-foreground-routing"></a>

## Focused foreground routing

The existing `AgentRouter` runs only at the top level. It selects the tea
workflow, ordinary conversation, a current-camera question, reference retrieval,
or one of the existing background applications. Routing descriptions come from
`task.foreground` in `workflow.yaml`.

Starting guidance returns `focus`. The workflow then receives the participant's
next request first. Its direct Clef question offers only the focused guide's
controls, conversation, or handoff. Informational questions share one conversation
path; there is no per-step source classifier or separate answer/check branch.
The language model receives managed context and the read-only tools declared in
`voice.tools`, using the standard SDK tool loop. Workflow mutation tools are
excluded from that loop.

Code maps bounded control labels to the existing native operations and their
arguments; YAML supplies semantic instructions and criteria. It does not define
an arbitrary tool-execution language. Mutating
controls also require a separate decision about the current user's authorization
of that exact operation and target. The positive authorization score must exceed
the combined negative and unknown scores; a minority positive cannot authorize
a mutation merely by winning the three-way classification. Requests about capabilities, quoted or
reported instructions, and negated commands do not authorize execution.
`WorkflowStore` still decides whether an authorized advance is ready.

A request for another application declines only the current turn. The router
hands it to the ordinary destination while retaining tea focus, just as a side
question can be answered during focused note taking. Explicit exit or completion
releases focus. Router focus does not replace workflow state or stop the
independent observation task.

The SOP agent owns a persistent dialogue `ContextManager` and run-owned managers
for live snapshots and execution history. Live and historical blocks are supplied
in separate prompt messages. It does not create a `Conversation`
agent. Each run has a unique ID. Its context contains:

| Channel | Contents | Update policy |
|---|---|---|
| `sop.execution` | Run and revision, active step, established state, native operation readings | Latest replacement snapshot |
| `sop.evidence` | Current-step evidence, with frame and observation time; cleared on step transition | Latest replacement snapshot |
| `sop.events` | Meaningful step/run events | Append and compact older events |
| Dialogue | Completed user requests and assistant responses | Append and compact older exchanges |
| `sop.handoff` | Historical context exported when a run closes | Persistent historical updates, compacted independently |

Before generating any answer, the execution engine samples the YAML-declared
native `context` projections. The timer is an `active_operation` projection
using `clock__timer`; its elapsed and remaining values are recalculated for
every question while that operation is present. This does not depend on matching
question wording. Visual observations continue on the background cadence; they
are timestamped evidence, not automatically fresh camera readings. The model
may request a fresh camera observation through its ordinary read-only tools.
Current source snapshots override historical assistant claims. Response quality
remains a model responsibility; no timer-specific answer template is used.

A run closes on explicit finish, reset or restart. Its final state and bounded
context are handed off as historical reference before its manager and compaction
tasks close. A new run receives a fresh manager, state and evidence. Dialogue
survives this rollover. The final response input explicitly identifies whether
execution is active, so a historical timer is not presented as a running one.
Participant departure clears both managers after draining producers. Detailed
run bundles remain available independently of bounded conversational context.

Replacement channels are the existing SDK's `summary_context_topics`: the name
describes retaining the latest source state, not asking an LLM to summarize live
values. History channels use `incremental_context_topics`. The configured context
budget assigns half to dialogue and divides the remainder between execution,
evidence, event history and historical handoff. No SDK clear API is added.

Every visual request sends one image through the existing image tools and Omni
adapter. Clef receives text decisions in this sample; no separate visual model
is introduced. The `llm` and `vlm` roles continue to share the Omni endpoint.

Completed response text is retained in router history for follow-up admission
and in managed context for response generation. The fresh state projection and
requested live result override compacted historical facts. Detailed frame checks
and model calls remain in the run bundle rather than flooding conversation
context. Participant departure drains response and observation producers before
clearing history; shutdown also drains context compaction.
The router owns admission, focus, cancellation, and handoff; agents own their
application state and background tasks.
The sample does not maintain another user-turn task registry or buffer destination
output for later republication. Destinations use SDK `ResponseOutputs` directly;
completed dialogue is recorded only after the reply terminator is delivered.
Initialization and participant cleanup use the router's existing hooks.

The sample's left-event adapter is deliberately separate from the router's
runtime subscriber: it calls public `router.participant_left()` to drain owned
work, then acknowledges the foreground and guidance producers using the original
left event's generation. It is the sole adapter for that event, so a second
late departure cannot reset a concurrent rejoin. The bridge retains the existing
file/image cleanup protocol; it does not supervise turns or implement routing.


## Workflow contract

`yaml/workflow.yaml` declares:

- typed state fields and initial values;
- ordered steps and explicit next-step IDs;
- which state each step may read and write;
- a periodic trigger and evidence rule;
- typed observation fields or accepted, rejected, and unknown choice criteria;
- deterministic evaluation tools and code-bound arguments;
- state patches applied only after confirmed evidence;
- current-step read-only tools, native context projections and generation instructions;
- completion conditions and skip values;
- enter, complete, and skip messages.

`WorkflowStore` enforces the contract. An observation can update only fields
listed in the active step's `writes`. Updates are atomic, validated against
their declared types, and ignored after the step is complete. Completion emits
a notice but never moves to the next step. Only explicit start, next, skip,
reset, or restart controls change the participant's position.

Models never propose an arbitrary state patch. An `observation` declares one of
three execution kinds:

- `choice`: Omni supplies a description and Clef chooses accepted, rejected, or
  unknown from YAML criteria.
- `extract`: Omni returns only declared JSON fields. Code validates their types,
  normalizes declared units, and optionally executes a native evaluation tool.
- `result`: code evaluates a typed trigger result, such as a timer's `expired`
  boolean, without calling a model.

`updates` contains literals or explicit bindings from `$observation`, `$state`,
or `$clock.now_us`. The parser rejects unwritable fields and unknown bindings.
A clock binding is evaluated by code only after confirmation. Temperature
conversion and duration conversion use code, not language-model arithmetic.

Identity must come from legible current-view evidence. `visual_fields` limits
the image request to identity fields; `retain` commits those validated fields
independently before retrieval, under the execution revision guard. Configured
unavailable labels normalize to missing evidence rather than becoming identity.
Unreadable later frames cannot erase retained identity. Restart clears it.

The identity-resolution step performs actual RAG retrieval using the configured
query. A text-only model extracts supported settings, with a source-passage
index and verbatim quote for each field. Code checks the quote and normalizes
units. The YAML can permit tea-type references when an exact product reference
is unavailable. Missing settings alone go through `fallback_instructions`;
unresolved identities are never generated. Each resolved setting has explicit
retrieved or generated provenance through `source_fields`. Supported partial
settings and provenance survive retries for unresolved fields. Generated values
are recommendations, never represented as package or retrieved facts. Ranges
without a single recommendation remain unavailable to extraction. Document
relevance and recommendation quality still require model qualification.

### Adapting an existing workflow YAML

Replace the former observation `agent.prompt` and `agent.tools` block with an
`observation` policy. Keep control semantics, `reads`, `writes`, confirmation
counts, advancement policy, skip values, and messages. Declare scoped
foreground control decisions under `task.foreground`, read-only tools under
`voice.tools`, and native source projections under `context`. `voice.prompt` now describes how to answer
from supplied facts rather than how to select tools.

An extracted field with `units` returns `{value: number, unit: label}`. Each
unit declares a scale and optional offset; code computes `(value + offset) *
scale` and validates the normalized result before it can enter state. State
continues to contain canonical scalar values.

## Visual observation loop

`SOPAgent` owns an internal execution engine with one observation task per
connected participant. While a
workflow step is active, the task:

1. Executes the step's deterministic trigger, such as `current_view` or
   `clock__timer`.
2. Runs the declared choice, extraction, or native-result policy.
3. Counts accepted evidence against the declared confirmation rule.
4. Applies the declared patch atomically only after the required confirmations.
5. Publishes typed records and completion notices.

The same source frame cannot count twice toward visual confirmation. Failed
inference can retry that frame; a valid decision consumes it once. Unknown
evidence preserves the existing confirmation count, while a clear negative
resets it. No new evidence-age timeout is imposed.

Configured choice observations return `observable` and `description`. An
explicitly unobservable view becomes unknown without a Clef call; an empty
vessel can still be observable. Clef receives the description and criteria,
without unrelated progress flags or brewing targets. Water and tea immersion
each use one semantic question; immersion requires an explicit contact
relationship. Numeric temperature comparisons and elapsed timers use code.
The heating step compares the observed unit-bearing reading with the resolved
brewing target using `temperature__verify`. Readings must reach or exceed the
target on two distinct frames before `temperature_ready` is confirmed. A reading
that is rising but below target does not complete the step. Missing readings or
units remain unknown; an explicit below-target reading resets confirmation.
No tolerance band or upper-temperature bound is inferred.

Each observation already supplies step-specific VLM context: choice steps have
visual descriptions; extraction steps have typed observed facts such as the
current temperature and unit. This is not a separate whole-scene caption call.
Live visual evidence belongs to the active step and clears when the step changes.
Completed-step observations remain in run history and historical handoffs;
confirmed workflow state persists for subsequent steps. The timer receives its
native operation context without carrying a prior step's visual evidence into
its live context.

`VisualValidationStep` accepts a question, parsed policy, shared services and
tools. `check(frame, expected=...)` returns evidence and frame identity without
mutating confirmation or workflow state. The execution engine owns cadence,
duplicate-frame rejection, confirmation and guarded commits. Its staged
observation/evaluation methods also support retaining identity before fallible
settings resolution. These flow APIs are public within the sample; they are not SDK APIs.

Unavailable frames, invalid observations, and service failures record an error
and retry on a later tick. They do not kill the participant session. A leave
event cancels the task and releases the session.

By default, a confirmed completion atomically advances to the next step and
combines the completion notice with its instruction. Set `task.advancement` to
`manual` to pause completed steps until the user says next. `manual_advance_message`
in the task YAML supplies that reminder. Explicit next remains guarded by step
readiness in both modes; skip remains an explicit override. Trigger and
model I/O run outside the participant state lock; the resulting commit is
accepted only when the captured step and revision still match. Reset, restart,
status, and advance controls therefore remain responsive during slow inference.

To use audio, sensor, or backend evidence, add another deterministic trigger
that returns a typed result. Keep observation, semantic judgment, and guarded
state mutation separate so the same workflow rules remain testable without the
live source. Regex matching may guard the internal judgment token, but should
not be the primary interpreter of natural-language captions.

## Independent background agents

Background applications expose participant-bound start, stop, and status tools
to the foreground agent:

- `ChangeWatchAgent` captions a current frame with Omni, uses Clef to select
  material changes, and generates a concise summary only for an accepted change.
- `TranscriptAgent` consumes the built-in pre-gate transcript topic and
  summarizes only unsummarized turns on its interval.
- `VideoLogAgent` records Omni captions, compares them with a five-caption
  window through Clef, and generates a delta only for an accepted transition.

Each publishes its durable record topic. Important compact results also become
`BackgroundFact` events. `BackgroundContextAgent` retains a bounded,
participant-local window and exposes `application_context__query` to the
foreground. Thus background work can inform a later question without becoming
conversation history or taking ownership of the response.

## Live inspection and durable output

The worker starts a generic loopback event viewer before it announces voice
readiness. Open `http://127.0.0.1:8093` on the XR-AI host, or pass
`--expose-web-events` to bind all IPv4 interfaces and browse
`http://<xr-host>:8093` from a trusted development network.
`TeaWebEventsAgent` explicitly subscribes to the
foreground, guidance, background, transcript, video-log, and participant
lifecycle topics and publishes compact `WebEvent` payloads. Visual panes show
only application lifecycle and material `BackgroundFact` results; every-frame
captions, no-change observations, unavailable frames, and errors remain in the
durable records and worker log. There is no runtime wildcard subscription and
no file tailing.

The viewer keeps only a bounded in-memory history for the current process. The
participant JSONL files under `artifacts/runs/<run>/` remain the durable record and include
events that may not be selected for live presentation. The listener is
read-only but has no authentication or TLS. Restrict TCP port 8093 to a trusted
development network when `--expose-web-events` is enabled, or place the viewer
behind an authenticated TLS proxy.

## Connecting a backend

Add a runtime subscriber for the stable domain events you need. Inject a typed
backend client rather than placing vendor HTTP code inside workflow or visual
agents.

```python
class JournalBackendAgent(Agent):
    def __init__(self, journal: LabJournal) -> None:
        self._journal = journal
        super().__init__()

    @subscribe(GUIDANCE_RECORD_TOPIC)
    async def guidance(
        self,
        record: GuidanceRecord,
        ctx: RuntimeContext,
    ) -> None:
        await self._journal.append_guidance(
            participant_id=ctx.metadata.participant_id,
            record=record,
        )
```

Register the subscriber beside `FileOutputAgent`. A customer system can consume
guidance records, background facts, transcripts, or video deltas independently.
If it runs in another process, expose the boundary through a typed msgpack over
ZMQ service rather than bypassing the repository's service model.

## Adapting the sample

### Replace tea making with another procedure

1. Define the new state fields and steps in `workflow.yaml`.
2. Give every step the smallest possible `reads`, `writes`, and tool set.
3. Put calculations and external facts behind deterministic tools.
4. Declare factual Omni extraction or caption prompts, bounded Clef criteria,
   native evaluations, and allowed updates in YAML. Keep domain rules out of
   the observation executor.
5. Choose `task.advancement: automatic` or `manual` for the product's progression policy.
6. Add state and routing tests before tuning prompts.

If the workflow needs a different state-machine rule, change
`agents/sop/_state.py`; do not encode the rule only in prose for the model.

### Add a domain tool

Implement a native `Tool` or `AsyncTool`, then list it only on the workflow
steps that need it. Keep participant identity in a closure or runtime metadata,
not in a model-selected request field. Execute peer-agent tools through their
public `execute()` boundary so validation and Relay tracing are preserved.

### Add another background observer

Follow the same contract as the three existing applications: an agent-owned
task per participant, idempotent start, stop, and status tools, cancellation on
leave, a typed durable record topic, and optional compact `BackgroundFact`
output. Decide explicitly whether to speak the result; do not make every
background record a voice message.

### Replace files with production persistence

Subscribe to the existing topics from a backend agent. Preserve participant and
correlation metadata from `RuntimeContext`. Define retry, buffering, ordering,
and failure behavior in that integration agent rather than blocking perception
or voice by accident.

### Change voice policy

The SOP agent delegates notice delivery to its internal `GuidanceVoiceAgent`
component. Add, suppress, batch, or redirect notices there. Background applications remain
text and event producers unless a dedicated voice subscriber chooses otherwise.
Foreground answers are replies to the user's query, and guidance notices are
unprompted updates, both published through `ResponseOutputs` to
`VOICE_OUTPUT_TOPIC`. `VoiceAgent` owns when they are spoken: it combines
updates that arrive while the participant is being spoken to, and a new query supersedes prior output. The response text reaches the connection client's Agent panel as
soon as that response or combined update is complete.

## Lifecycle invariants

- Participant join starts a fresh workflow and output session.
- Participant leave cancels foreground, observation, and background tasks
  before image resources are released.
- Record-producing agents publish cleanup-complete events after cancellation;
  `FileOutputAgent` closes the session only after every producer has finished.
- A superseding query cancels the participant's prior foreground turn.
- Each participant has independent workflow state and locks.
- Agent tasks fork the Relay context for traceable native tool execution.
- Cancellation propagates through model and tool calls.
- Background facts are bounded and cleared on participant lifecycle changes.

Preserve these rules when extracting or replacing agents. State ownership and
cleanup are one design decision, not separate implementation details.

## Safety

This sample demonstrates agent composition; it is not a safety controller.
Keep hot vessels stable, follow appliance and tea-package instructions, and do
not use visual inference as the sole protection against burns, spills, or
electrical hazards.

## Validation strategy

Test deterministic behavior separately from model quality:

1. Validate workflow YAML, field types, readable and writable sets, and step links.
2. Unit-test atomic commits, evidence counts, automatic/manual advancement, skips,
   resets, and no mutation after completion.
3. Test root-versus-active tool visibility and the absence of conversation
   history.
4. Test background lifecycle, deduplication, summary windows, and cancellation.
5. Add routing evals for prompt-controlled tool choices.
6. Run the live pipeline to evaluate visual evidence and model prompts.

The live-model eval uses the production SDK router, destinations, native SOP
controls and answer request assembly. External background effects and camera
inputs are fixtures. Setup executes the native start control to establish actual
focus; populating workflow state alone is not a focused-agent test.
The suite checks selected capabilities, fixed skip arguments, generation scope,
and response constraints. Caption fixtures check extraction, not physical-camera
accuracy. The normal acceptance floor remains 80 percent; inspect every unsafe
control miss independently of that aggregate.

Start the sample's model stack so Omni on port 8108 and Clef on port 8120 are
healthy. Run from `agent-samples/tea-making-sample/`:

```bash
uv run --project worker python eval/eval.py
uv run --project worker python eval/eval.py --cases eval/focused-cases.yaml
```

Use `--output <path>` to retain JSONL results and per-call timings. These timings
exclude microphone processing, camera acquisition, visual inference for dry-run
camera actions, TTS, and client playback. They do not measure audible-response
latency. Check a real camera session before proposing a deployment.

The standalone model launcher adds Clef to the existing speaker STT, Pocket TTS,
Omni, and embedding services. That is one additional checkpoint and service;
sharing Omni for language and vision still loads only one Omni checkpoint.
Existing KV and utilization settings are retained. Qualify peak memory and
concurrent load on the selected GPU profile rather than treating per-service
utilization as a total-memory cap.

When debugging, inspect the workflow, background, foreground, and Relay JSONL
files as separate stages. This shows whether the problem came from evidence,
tool selection, state policy, event delivery, or presentation.

<a id="what-should-become-shared"></a>

## What belongs in shared code

Reuse public SDK blocks directly when they already express the contract:
`VoiceAgent`, `AgentRuntime`, `run_tool_loop`, `CurrentFrameTool`,
`ImageQueryTool`, `RAGTools`, and typed topics. Keep the workflow schema,
evidence rules, background-fact policy, and user-facing notices sample-local
until another concrete application requires the same contract.
