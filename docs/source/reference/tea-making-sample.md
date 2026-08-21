<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance — building deterministic visual workflows

The tea-making sample is a reference for applications that combine a guided
procedure, visual evidence, foreground questions, independent background
observers, and voice output. Start with the
{doc}`quickstart </getting_started/quickstart>` to run it. This architecture
reference explains how to adapt the sample to another procedure rather than
merely changing model settings.

The defining pattern is a deterministic workflow around an agentic core. YAML
declares the steps and state contract. Application code owns transitions and
evidence. The model interprets observations and selects native tools, but it
does not choose which workflow is active, mutate arbitrary state, or advance a
step implicitly.

## What to copy

Use this sample when your application needs one or more of these patterns:

- a foreground agent that changes its tool catalog based on explicit state;
- a guided procedure whose evidence comes from current visual observations;
- sparse typed state updated through one controlled commit boundary;
- workflow completion without automatic advancement;
- independent, opt-in visual, transcript, and video background tasks;
- bounded background facts available to foreground questions;
- typed events consumed separately by speech, files, or a customer backend.

Tea names, temperatures, timers, and step prompts are sample-specific. The
workflow state machine, foreground routing, observation loop, background-agent
ownership, and event composition are reusable.

## Quick start

Run commands from the repository root. The tea sample reuses all model
services, so start the shared stack and Piper TTS first:

```bash
uv run --project agent-samples/model-servers model_servers
uv run --project services/piper-tts piper_tts_server
```

Then launch the sample:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample
```

Open `https://localhost:8080`, accept the development certificate, allow camera
and microphone access, and connect. Wake-word mode is the default; try:

```text
Agent, help me make tea.
Agent, what is the current step?
Agent, what do you see?
Agent, start watching for spills.
Agent, start recording the transcript.
```

The launcher options are:

| Option | Behavior |
|---|---|
| `--expose-web-events` | Bind the unauthenticated live event viewer to all IPv4 interfaces instead of loopback |

Speech uses the reused TTS endpoint in `yaml/models.local.json`. Voice-gate
behavior comes from `voice_gate_yaml` in `yaml/tea_making_worker.yaml`.

The live event viewer is available at `http://127.0.0.1:8092` on the XR-AI
host. With `--expose-web-events`, use `http://<xr-host>:8092` from a trusted
development network and restrict TCP port 8092 accordingly. Durable JSONL
records remain under `agent-samples/tea-making-sample/artifacts/`; the viewer
is only a bounded view of the current worker process.

## Architecture

```text
camera frames ────────────────> ParticipantImageAgent
                                      │
accepted speech or typed query         ├─> GuidanceAgent observation loop
              │                       ├─> ChangeWatchAgent
              v                       └─> VideoLogAgent
        ForegroundAgent
              │
              ├─ idle ──> root tools: start workflow, inspect, retrieve,
              │                      query facts, control background agents
              │
              └─ active > current-step tools + sparse workflow context
                                      │
                                      v
                                typed runtime events
                                      ├─> GuidanceVoiceAgent ─> voice aggregation
                                      ├─> BackgroundContextAgent
                                      ├─> FileOutputAgent ─────> JSONL
                                      ├─> TeaWebEventsAgent ───> live browser
                                      └─> customer backend agent
```

`VoiceAgent` publishes participant-scoped queries, transcripts, and lifecycle
events. The runtime fans typed events out to peer agents. Agents call peer
tools directly, and each background agent owns its participant tasks.
Wake-word gating is the sample default; callers can explicitly select
always-on speech. Every spoken response is also sent to the connection client
on `agent.response` so the client can render accessible text alongside audio.
Raw streamed responses are finalized there as soon as their content completes,
and batched responses are finalized as soon as their rewrite completes. The
voice aggregator's playback reservation affects only subsequent speech
scheduling, not when the text appears. An urgent barge-in can still stop audio
after the full intended utterance has reached the client.

## Agent responsibilities

| Agent | Owns | Does not own |
|---|---|---|
| `ParticipantImageAgent` | Image registry, current-frame selection, participant cleanup | VLM prompts or workflow state |
| `ForegroundAgent` | Deterministic root-versus-step routing and one bounded tool loop per query | Conversation history or workflow transitions |
| `GuidanceAgent` | Workflow sessions, step observations, evidence, controlled state commits | Voice transport or persistence |
| `BackgroundContextAgent` | A bounded participant-local set of recent background facts | Background scheduling |
| `ChangeWatchAgent` | Focused visual-change task and deduplication | Foreground routing |
| `TranscriptAgent` | Opt-in transcript records and periodic summaries | Voice gating or STT transport |
| `VideoLogAgent` | Periodic captions and material deltas | Workflow state |
| `GuidanceVoiceAgent` | Which guidance notices are spoken | Workflow decisions |
| `VoiceAggregationAgent` | Participant speech pacing, ordering, and coalescing | Workflow decisions |
| `FileOutputAgent` | Participant session files and JSONL records | Agent policy |
| `TeaWebEventsAgent` | Explicit compact projections for the generic live viewer | Durable storage or file polling |
| `WebEventsAgent` | Bounded HTTP presentation grouped by participant and topic | Application event selection |

The runtime delivers events but does not absorb application state or
concurrency. Moving a task to another agent also means moving its cancellation,
locks, and participant cleanup.

## Source map

| File | Purpose | Typical adaptation |
|---|---|---|
| `app.py` | Constructs services and registers agents | Add a backend or another observer |
| `events.py` | Typed application topics and payloads | Define domain facts, notices, or records |
| `spec.py` | Validates the workflow YAML contract | Extend declarative workflow metadata |
| `workflow_state.py` | Typed sparse state, evidence, transitions, rendering | Change deterministic state policy |
| `workflow_tools.py` | Current view, RAG, clock, temperature, and commit tools | Add domain tools |
| `workflow.py` | Participant sessions and visual observation loop | Add another trigger or evidence source |
| `foreground.py` | Root-versus-active tool routing and current-query loop | Change interactive behavior |
| `images.py` | Shared participant image acquisition | Add another image selector |
| `background_context.py` | Recent-fact store and query tool | Change foreground and background context policy |
| `change_watch.py` | Focused visual observer | Build another event-oriented monitor |
| `transcript.py` | Transcript recording and summarization | Forward or classify laboratory dialogue |
| `video_log.py` | Periodic caption and delta observer | Build a visual journal |
| `guidance_voice.py` | Guidance notice speech policy | Add another presentation channel |
| `file_output.py` | Participant JSONL artifacts | Replace files with backend persistence |
| `web_events.py` | Explicit typed event-to-view projections | Select or reshape live presentation topics |
| `prompts/` | Default model instructions | Tune one agent without duplicating prompts in YAML |
| `yaml/workflow.yaml` | Five-step procedure and evidence contract | Define a different guided task |
| `rag-documents/` | Sample retrieval corpus | Replace with domain documents or a backend |

Configuration and command syntax are included automatically in the generated
{doc}`configuration <configuration>` and
{doc}`command-line <command-line>` references.

## Deterministic foreground routing

Before calling the LLM, `ForegroundAgent` asks `GuidanceAgent` whether the
participant has an active step.

- With no active workflow, the model receives root tools: start/status,
  current view, RAG, recent background facts, and background controls.
- With an active workflow, it receives the current step's tools plus explicit
  workflow controls and compact sparse state.

There is exactly one model loop. The foreground agent does not ask one model to
route to another agent, and background agents never capture a foreground turn.
`current_view` is a direct-return exception within that loop: after the route is
selected, `CurrentFrameTool` and `StreamingImageQueryTool` use the same streamed
voice path as `simple-vlm-example`, without a second language-model pass over
the visual answer.
Each turn contains the system prompt, current query, and current workflow
context—no accumulated conversation history.

The checked-in Omni deployment accepts one image per request by default, and
every visual tool in this sample sends exactly one image. If an adaptation adds
a multi-image tool such as ``query_images``, configure the model server's
``--limit-mm-per-prompt`` value before exposing that tool. The explicit
``reasoning_field`` in ``models.local.json`` also keeps the inline VLM adapter
aligned with the Nemotron Omni preset.

This makes routing easy to reason about: application state decides which
capabilities exist, and the model decides how to use only those capabilities.

## Workflow contract

`yaml/workflow.yaml` declares:

- typed state fields and initial values;
- ordered steps and explicit next-step IDs;
- which state each step may read and write;
- a periodic trigger and evidence rule;
- tools available to observation and voice turns;
- completion conditions and skip values;
- enter, complete, and skip messages.

`WorkflowStore` enforces the contract. An observation can update only fields
listed in the active step's `writes`. Updates are atomic, validated against
their declared types, and ignored after the step is complete. Completion emits
a notice but never moves to the next step. Only explicit next, skip, reset, or
restart controls change the participant's position.

The model can propose state only through the commit tool. It cannot mutate the
session object directly. Deterministic clock and temperature tools perform
calculations that require deterministic arithmetic rather than language-model
arithmetic.

## Visual observation loop

`GuidanceAgent` owns one observation task per connected participant. While a
workflow step is active, the task:

1. Executes the step's deterministic trigger, such as `current_view` or
   `clock__timer`.
2. Applies the declared evidence rule.
3. Calls the bounded tool loop with only the step's observation tools.
4. Accepts state through the controlled commit tool.
5. Publishes typed records and completion notices.

Unavailable frames, invalid observations, and service failures record an error
and retry on a later tick. They do not kill the participant session. A leave
event cancels the task and releases the session.

Completed steps pause this loop until the user explicitly advances. Trigger and
model I/O run outside the participant state lock; the resulting commit is
accepted only when the captured step and revision still match. Reset, restart,
status, and advance controls therefore remain responsive during slow inference.

To use audio, sensor, or backend evidence, add another deterministic trigger
that returns a typed result. Keep evidence collection separate from state
mutation so the same workflow rules remain testable without the live source.

## Independent background agents

Background applications expose participant-bound start, stop, and status tools
to the foreground agent:

- `ChangeWatchAgent` captions a current frame, detects important focused
  changes, and suppresses duplicate observations.
- `TranscriptAgent` consumes the built-in pre-gate transcript topic and
  summarizes only unsummarized turns on its interval.
- `VideoLogAgent` records broad visual captions and produces deltas from a
  five-caption window.

Each publishes its durable record topic. Important compact results also become
`BackgroundFact` events. `BackgroundContextAgent` retains a bounded,
participant-local window and exposes `application_context__query` to the
foreground. Thus background work can inform a later question without becoming
conversation history or taking ownership of the response.

## Live inspection and durable output

The worker starts a generic loopback event viewer before it announces voice
readiness. Open `http://127.0.0.1:8092` on the XR-AI host, or pass
`--expose-web-events` to bind all IPv4 interfaces and browse
`http://<xr-host>:8092` from a trusted development network.
`TeaWebEventsAgent` explicitly subscribes to the
foreground, guidance, background, transcript, video-log, and participant
lifecycle topics and publishes compact `WebEvent` payloads. There is no runtime
wildcard subscription and no file tailing.

The viewer keeps only a bounded in-memory history for the current process. The
participant JSONL files under `artifacts/` remain the durable record and include
events that may not be selected for live presentation. The listener is
read-only but has no authentication or TLS. Restrict TCP port 8092 to a trusted
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
4. Define evidence independently of the observation prompt.
5. Keep advancement explicit unless the product explicitly requires automatic
   transitions.
6. Add state and routing tests before tuning prompts.

If the workflow needs a different state-machine rule, change
`workflow_state.py`; do not encode the rule only in prose for the model.

### Add a domain tool

Implement a native `Tool` or `AsyncTool`, then list it only on the workflow
steps that need it. Keep participant identity in a closure or runtime metadata,
not in a model-selected request field. Execute peer-agent tools through their
public `execute()` boundary so validation and Relay tracing are preserved.

### Add another background observer

Follow the same contract as the three existing applications: an agent-owned
task per participant, idempotent start, stop, and status tools, cancellation on
leave, a typed durable record topic, and optional compact `BackgroundFact`
output.
Decide explicitly whether to speak the result; do not make every background
record a voice message.

### Replace files with production persistence

Subscribe to the existing topics from a backend agent. Preserve participant and
correlation metadata from `RuntimeContext`. Define retry, buffering, ordering,
and failure behavior in that integration agent rather than blocking perception
or voice by accident.

### Change voice policy

`GuidanceVoiceAgent` is intentionally separate from `GuidanceAgent`. Add,
suppress, batch, or redirect notices there. Background applications remain
text and event producers unless a dedicated voice subscriber chooses otherwise.
Foreground responses and guidance notices publish to `VoiceAggregationAgent`,
which combines non-urgent output produced during an active utterance before it
reaches `VoiceAgent`. The response text reaches the connection client's Agent
panel as soon as that raw or rewritten response is complete; the aggregator's
open-loop playback estimate remains private scheduling state.

## Lifecycle invariants

- Participant join starts a fresh workflow and output session.
- Participant leave cancels foreground, observation, and background tasks
  before image resources are released.
- Participant leave also releases queued voice-aggregation state, so a later
  connection reusing the participant ID cannot inherit stale speech.
- Record-producing agents publish cleanup-complete events after cancellation;
  `FileOutputAgent` closes the session only after every producer has finished.
- A superseding query cancels the participant's prior foreground turn.
- Each participant has independent workflow state and locks.
- Agent tasks fork the Relay context for traceable native tool execution.
- Cancellation propagates through model and tool calls.
- Background facts are bounded and cleared on participant lifecycle changes.

Preserve these rules when extracting or replacing agents. State ownership and
cleanup are one design decision, not separate implementation details.

## Validation strategy

Test deterministic behavior separately from model quality:

1. Validate workflow YAML, field types, readable and writable sets, and step links.
2. Unit-test atomic commits, evidence counts, explicit advancement, skips,
   resets, and no mutation after completion.
3. Test root-versus-active tool visibility and the absence of conversation
   history.
4. Test background lifecycle, deduplication, summary windows, and cancellation.
5. Add routing evals for prompt-controlled tool choices.
6. Run the live pipeline to evaluate visual evidence and model prompts.

When debugging, inspect the workflow, background, foreground, and Relay JSONL
files as separate stages. This shows whether the problem came from evidence,
tool selection, state policy, event delivery, or presentation.

## What belongs in shared code

Reuse public SDK blocks directly when they already express the contract:
`VoiceAgent`, `AgentRuntime`, `run_tool_loop`, `CurrentFrameTool`,
`ImageQueryTool`, `RAGTools`, and typed topics. Keep the workflow schema,
evidence rules, background-fact policy, and user-facing notices sample-local
until another concrete application requires the same contract.
