<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance — building deterministic visual workflows

The tea-making sample is a reference for applications that combine a guided
procedure, visual evidence, foreground questions, independent background
observers, and voice output. Refer to the
{doc}`quickstart </getting_started/quickstart>` to run the sample. This
architecture reference explains how to adapt the sample to another procedure
rather than merely changing model settings.

The defining pattern is a deterministic workflow around an agentic core. YAML
declares the steps and state contract. A VLM describes each current view, and
the observation LLM judges that natural-language caption and selects native
tools. Application code validates the resulting state proposal and counts any
required consecutive confirmations. The model does not choose which workflow
is active, mutate arbitrary state, or advance a step implicitly.

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

Run commands from `agent-samples/tea-making-sample/`. The tea sample reuses all
model services, so start the shared stack, including Piper TTS, first:

```bash
uv run --project ../model-servers model_servers
```

Wait for the model launcher to report readiness and return. Then launch the
sample from the same terminal:

```bash
uv sync
uv run tea_making_sample
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
The worker uses native `xr_ai_runtime` agents and `xr_ai_tools` instances; it
does not use NVIDIA Agent Toolkit, PydanticAI, MCP clients, or MCP servers.
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
| `prompts/` | Shared model instructions and packaged route policies | Tune configurable prompts without duplicating them in YAML |
| `yaml/workflow.yaml` | Five-step procedure and evidence contract | Define a different guided task |
| `rag-documents/` | Sample retrieval corpus | Replace with domain documents or a backend |

Refer to the generated {doc}`configuration <configuration>` and
{doc}`command-line <command-line>` references for configuration fields and
command syntax.

## Configuration

Run and edit the sample from `agent-samples/tea-making-sample/`. The
orchestrator reads the checked-in files on every start and materializes a
temporary worker configuration with absolute paths. Edit the files under
`yaml/`, not the temporary copy named in the logs.

| File | Owns |
|---|---|
| `yaml/tea_making_worker.yaml` | Selected subordinate configuration files, frame and model timeouts, background cadence, VAD, output directory, and event-viewer port and history |
| `yaml/workflow.yaml` | Typed state, steps, triggers, evidence, tool access, transitions, and user messages |
| `yaml/voice_gate.yaml` and `yaml/voice_gate.always-on.yaml` | Wake-word and always-on speech presets |
| `yaml/rag_service.yaml` | Document path, model configuration, cache, chunking, embedding dimensions, and score threshold |
| `yaml/models.local.json` | Reused model adapters, endpoints, and readiness checks |
| `yaml/device_io_hub.yaml` | LiveKit room and ports, web and token servers, and network behavior |

Choose a voice preset through `voice_gate_yaml` in
`tea_making_worker.yaml`. Change observation and summary frequency in that
worker file, retrieval behavior in `rag_service.yaml`, and the procedure itself
in `workflow.yaml`. Relative paths declared in each configuration file resolve
from that file's directory. Workflow state and step references are validated at
startup, so an invalid contract fails before a participant session begins.

Restart `tea_making_sample` after an edit; configuration is not hot-reloaded.
The `--expose-web-events` option intentionally overrides `web_events_host` in
the runtime copy. Changing `models.local.json` changes the endpoints used by
the worker and RAG service but not the persistent servers. Refer to
{doc}`/guides/customizing-model-servers` for a server-side model, GPU, port, or
memory change, then restart the shared stack.

Refer to the generated {doc}`configuration <configuration>` reference for exact
fields, checked-in values, and adjacent YAML comments.

## Deterministic foreground routing

Before calling the LLM, `ForegroundAgent` asks `GuidanceAgent` whether the
participant has an active step.

- With no active workflow, the model receives root tools: start/status,
  current view, RAG, recent background facts, and background controls.
- With an active workflow, it receives the focused current-step policy and
  compact sparse state. Read-only step tools remain available unless the whole
  utterance is a procedural guide question. A state-changing workflow tool is
  exposed only for an explicit command, with advance-versus-skip arguments
  fixed before the model call.

Active turns may address the current step, guide status and controls, or the
independent background applications. For a request unrelated to all of those,
the model must call no tool, leave guide state unchanged, and briefly decline.
The wording is intentionally not fixed. The rule is scoped to the active route;
the idle root assistant can answer ordinary questions.

An inline `foreground_prompt` or `foreground_prompt_file` supplies the root
assistant and explicit background-route instructions. Active tea behavior comes
from the focused policies built from `workflow.yaml`; configuration overrides
cannot replace the active-guide scope or workflow-control safeguards.

There is exactly one model loop. The foreground agent does not ask one model to
route to another agent, and background agents never capture a foreground turn.
`current_view` is a direct-return exception within that loop: after the route is
selected, `CurrentFrameTool` and `StreamingImageQueryTool` use the same streamed
voice path as `simple-vlm-example`, without a second language-model pass over
the visual answer.
Each turn contains the system prompt, current query, and current workflow
context—no accumulated conversation history.

Every visual tool in this sample sends exactly one image. The checked-in Omni
wrapper does not expose multi-image request configuration, so this deployment
does not support adapting these tools to send multiple images. The explicit
``reasoning_field`` in ``models.local.json`` keeps the inline VLM adapter aligned
with the Nemotron Omni preset.

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
a notice but never moves to the next step. Only explicit start, next, skip,
reset, or restart controls change the participant's position.

The model can propose state only through the commit tool. It cannot mutate the
session object directly. Deterministic clock and temperature tools perform
calculations that require deterministic arithmetic rather than language-model
arithmetic.

## Visual observation loop

`GuidanceAgent` owns one observation task per connected participant. While a
workflow step is active, the task:

1. Executes the step's deterministic trigger, such as `current_view` or
   `clock__timer`.
2. Calls the bounded observation tool loop to judge the trigger result.
3. Counts a valid completion proposal against the declared confirmation rule.
4. Accepts state through the controlled commit tool only after the required
   confirmations.
5. Publishes typed records and completion notices.

Unavailable frames, invalid observations, and service failures record an error
and retry on a later tick. They do not kill the participant session. A leave
event cancels the task and releases the session.

Completed steps pause this loop until the user explicitly advances. Trigger and
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
lifecycle topics and publishes compact `WebEvent` payloads. Visual panes show
only application lifecycle and material `BackgroundFact` results; every-frame
captions, no-change observations, unavailable frames, and errors remain in the
durable records and worker log. There is no runtime wildcard subscription and
no file tailing.

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
4. Give the VLM a small factual caption prompt and the observation LLM a small
   semantic judgment prompt; use code only to validate and count proposals.
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
output. Decide explicitly whether to speak the result; do not make every
background record a voice message.

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

## Safety

This sample demonstrates agent composition; it is not a safety controller.
Keep hot vessels stable, follow appliance and tea-package instructions, and do
not use visual inference as the sole protection against burns, spills, or
electrical hazards.

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

The checked-in live-model routing eval shares the production foreground's
root-versus-active prompt and tool preparation. It validates the first model
action, bound workflow arguments, semantic active-guide refusals, and positive
active-guide routes. The suite must maintain an 80 percent pass rate so
occasional language-model variation remains visible without making the
evaluation unusable. Start `model-servers` so the configured Omni LLM endpoint
on port 8108 is healthy, then run from `agent-samples/tea-making-sample/`:

```bash
uv run --project worker python eval/eval.py
```

The command prints `PASS` or `MISS` for each case. It exits nonzero when the
overall pass rate drops below 80 percent.

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
