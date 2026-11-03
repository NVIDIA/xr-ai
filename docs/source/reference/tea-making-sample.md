<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance with independent Python flows

## Overview

This sample separates SOP controls/execution from conversation. A control-only
SOP runner coordinates ordinary Python async functions; an SDK conversation agent
answers questions using context supplied by the runner. The SDK router dispatches
between these two destinations. There is no per-step agent, foreground supervisor,
YAML execution language or separate workflow engine.

Omni observes images. Clef judges those observations against supplied completion
criteria. Python owns confirmation windows, lifecycle, timers, defaults and step
transitions. Application YAML supplies prompts/settings to Python arguments; no
reusable flow reads YAML.

## What to copy

Start with the independent async functions in `tea_making_worker.components`, then
read `application/tea_sop_agent.py::_procedure` for the explicit tea procedure. The
tea application's Pydantic identity/quantity models live in `application/tea_data_models.py`.
There is no common flow base class; a little duplication keeps the three functions
independently understandable and usable without the SOP runner.

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
How much time is left?
Skip this step.
Stop tea making.
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

```text
VoiceAgent -> SDK AgentRouter
                |-- controls -> SOP runner -> independent async flows
                |                   ^                |
                |                   +-- full state --+
                |                   |
                |                   +-- SOP context -> SDK ContextManager
                |                                            |
                +-- questions -> SDK Conversation <----------+
```

The router owns turn scheduling, interruption and participant lifecycle. Starting
an SOP launches a runner-owned procedure task that outlives the initiating turn.
A question or voice interruption cancels answer work, not that procedure.
Stop/reset cancels and drains the runner's work. Explicit skip cancels the current
step; ordinary next cannot bypass an unconfirmed step.

The runner owns latest labeled state per flow. A flow intentionally spanning
several steps keeps its own active label until its Python scope ends. Completed,
skipped and previous-run data are labeled separately from active evidence. Flows
do not import conversation, recording, the runner or application configuration.

## Sample-local APIs and prompt ownership

Import `SOPAgent` from `tea_making_worker.application.tea_sop_agent`,
`SOPConfig`/`VisualInstructions` from `application.tea_sop_configuration`, and
tea result schemas from `application.tea_data_models` under the same package.
These are application-local APIs, not SDK promotion. The package root is a marker;
importing reusable components does not load the tea application.

Import `run_vision_rag_retrieval_flow`, `run_visual_verification_flow` and
`run_timer_flow` from `tea_making_worker.components`. Supply all dependencies,
inputs, wording, limits and the optional asynchronous `on_state` callback in
the call, and await its result. Each flow owns its local loop and propagates
cancellation after its awaited operations end. Callbacks run in that same task;
a callback failure propagates to the owner.

### Vision and RAG retrieval

`run_vision_rag_retrieval_flow` accepts `visual_model` and `result_model` as
Python/Pydantic schemas, the typed image and retrieval tools, an LLM service,
a Python `retrieval_query` function, prompts and bounds. `visual_question` selects
what to extract from each image; successful schema validation makes that data
usable for retrieval. Result fields must default to `None`; application defaults
belong to the runner. Tea identification and brewing settings are one application
of this flow; other visual data and result schemas require no flow changes. The tea
schema accepts a readable name or explicitly printed `tea_type`; a brand is optional.

It observes fresh frames until visual schema validation succeeds, retrieves relevant
passages, extracts supported fields and generates only missing result fields.
`VisionRAGRetrievalState` reports `visual_data`, `resolved_data`, source provenance,
recent original observations and retrieved passages. The extraction/generation
request carries visual data under the generic `visual_data` key.
Passages carry explicit `passage_id` labels. The response schema enumerates only
those IDs; source validation requires an exact ID and exact quote without guessing
an index convention. Each proposed source also reports `applicability` as
`applicable`, `unrelated` or `unknown` in the same extraction call. Only applicable
evidence can supply a retrieved value or constrain missing-value generation.
The model judges relevance; native checks enforce the declared applicability,
source identity, quote, units and range constraints. Field validators receive `source_kind` and `source_quote`
through standard Pydantic validation context. The tea quantity models reject a
range-selected scalar as a literal retrieved value and constrain generated values
to recognized, explicitly unit-bearing cited ranges. Valid range evidence and
passages remain in missing-only generation requests; recommendations are labeled
`generated`. These checks do not prove every semantic interpretation of an image
or document is correct.

Successful retrieval and validated partial/null resolution are cached for the
current visual data. Unchanged data does not rerun retrieval/extraction/generation;
the flow waits for new visual information. Malformed responses or invalid
candidates repair against the known data with bounded feedback, without rereading
the camera or repeating retrieval. A genuinely new visual result clears the old
entity's partial values/evidence. Callback and service failures propagate to the
owner; cancellation ends the observation/repair loop.

For an explicitly observed tea category, application instructions allow conventional
generated starting values when retrieval is empty or unrelated. An unfamiliar
brand does not require another camera view. Application validation requires
nonempty `tea_type` for generated settings and discards guesses without it;
exact retrieved settings can resolve without a type. This depends on the vision
model's type reading being correct. A later unreadable frame retains the
last usable identity and partial settings within this flow; its observation is
still labeled unusable. A genuinely different usable identity invalidates them.
If settings remain unresolved, the runner publishes `messages.identify_pending`
once when the missing-setting reason changes, asking for the type/instructions
or an explicit skip. It never silently fills defaults. Camera observation can
continue at the existing cadence, without repeating resolution for unchanged data.

Validated partial results are published before subsequent calls. If identification
is skipped, the runner retains those values and defaults only missing fields,
keeping each source labeled as retrieved, generated or default.

### Visual observation loop

Visual verification owns fresh-frame acquisition, one Omni observation followed
by one Clef judgment, bounded state updates, and the confirmation window.
The supplied criteria have `positive`, `negative` and `unknown` labels.

For two positives within four decidable observations:

| Evidence | Outcome |
|---|---|
| positive, positive | Complete immediately |
| positive, negative, negative, positive | Complete |
| positive, negative, negative, negative, positive | Continue; latest four contain one positive |
| positive, unreadable, positive | Complete; unreadable evidence consumes no slot |

Duplicate/older frames on the same track do not count again. Unknown observations
are retained as context but consume no positive/negative window slot. Clear
negatives consume a slot rather than resetting every prior positive.

Water presence, tea-water contact and target temperature use the same function.
The runner constructs temperature wording from the resolved target. Current
temperature must be visibly readable with an explicit unit and distinguished from
a setpoint. Otherwise the observer answers `no visible temperature`; Clef treats
that as unknown, consuming no confirmation slot. No freezing/boiling examples are
included in the visual question. The configured criteria account for Celsius/Fahrenheit. This semantic comparison
requires model qualification and is not the former native numeric evaluator.

### Native timer

`run_timer_flow` starts timing when executed, uses monotonic elapsed time, and
publishes duration, wall-clock start/deadline, sample time, elapsed/remaining time
and completion. The update interval is configurable. Completion is not delayed by
an update interval longer than the remaining duration. There are no model calls
or timer-specific question formatters.

### Full, bounded state context

Each function returns its terminal state and publishes full snapshots while
running. The runner stores those snapshots directly; it does not rebuild flow
progress from events. Visual verification and vision/RAG retrieval states retain the last few
original observations with time/frame labels, judgments, goals and useful source
evidence. Vision/RAG retrieval also retains passages/provenance; timer state carries
native operational facts.

History count, optional age, observation text size and retrieval passage limits
are configurable. Truncation is explicit. Current goals, resolved settings and
confirmation progress remain separate from history eviction. Unknown evidence
remains useful context without becoming a confirmation.

The runner replaces one SOP context topic in the SDK ContextManager; standard SDK
conversation handles questions about tea-making guidance. Before a run starts,
that topic contains only idle status and any explicitly historical prior run,
not the procedure plan or unused default settings. During execution it supplies
instructions only for steps already entered. Partial identification stays in its
flow state; effective brewing settings appear only once identification completes
or is explicitly skipped. Capability answers remain high level, while explicit
step questions can use the relevant supplied state.

SDK Conversation owns dialogue/history/compaction. All active flow states have distinct
labels, and completed/skipped states remain clearly historical. Reset clears live
SOP state and keeps a concise labeled prior-run summary; dialogue remains owned
by Conversation. The SOP topic budget is configurable independently of dialogue.
Dialogue retains earlier assistant replies, including countdowns, inside the
SDK's historical-dialogue boundaries. The current SOP source is separately
delimited and timestamped; active flow states include a readable sample time.
The timer source holds only its latest native elapsed/remaining snapshot, while
earlier answers remain historical dialogue. There is no step-specific deletion,
timer-question matcher, canned answer or parallel conversation manager.
The checked-in SOP topic budget is 64 KB; history retains up to eight observations
with at most 1,500 characters each. These are independent configurable limits,
not a guarantee that every worst-case combination fits. Adjust them together
for unusually verbose evidence or custom parallel work; ContextManager explicitly
clips an oversized source. Current facts/progress precede historical flow context.

### Sequential and scoped parallel execution

Await a flow directly for sequential work. The runner's Python procedure can
schedule the same async functions in an `asyncio.TaskGroup` when work should
span several steps. Bind each state callback to a distinct flow label. At the
scope end, await a needed result or cancel an unneeded still-running task and
let the group drain it. Normal task-group exit waits; it does not implicitly
cancel an indefinite flow. Required task failures propagate; no scheduler or
optional-flow policy language is introduced.

## Controls and progression

The two routed destinations are `workflow` and `conversation`. Clef control
classification and exact-intent authorization run before native actions. Negated,
quoted, reported and hypothetical controls are questions/discussion, not execution
requests. The conversation agent has no state-write tools.

The runner's explicit Python sequence is identify -> water -> temperature ->
tea-water contact -> timer. `sop.advancement: automatic` is the default.
Use `manual` to wait for next after a flow completes. Next does not bypass an
incomplete step; skip is the explicit bypass. Skipping tea-water contact detection
also skips its associated timer because no steeping start was established. Stop ends execution; reset also
clears current progress. Restart creates a fresh run.

## Source map

The worker separates application-specific code from reusable components at the
package's top level. Components depend on SDK libraries and each other; they do
not import tea schemas, the runner, application configuration or the YAML adapter.
The runner remains explicitly tea-specific, rather than hiding its procedure
behind a generic engine or workflow language.

```text
tea_making_worker/
  __init__.py
  __main__.py
  application/
    __init__.py
    tea_application.py
    application_config.py
    tea_data_models.py
    tea_sop_configuration.py
    tea_sop_agent.py
    tea_events.py
    tea_event_viewer.py
  components/
    __init__.py
    visual_verification_flow.py
    vision_rag_retrieval_flow.py
    timer_flow.py
    sop_run_recording.py
```

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker without application imports |
| `__main__.py` | CLI delegates to application config/loading |
| `application/__init__.py` | Application package marker |
| `application/tea_application.py` | Compose SDK models, voice, camera, conversation/router and the tea runner |
| `application/application_config.py` | Validate main YAML and construct Python configuration |
| `application/tea_data_models.py` | Tea identity, temperature, duration and partial brewing-result schemas |
| `application/tea_sop_configuration.py` | Plain Python tea prompts, criteria, defaults and loop/bounds arguments |
| `application/tea_sop_agent.py` | Controls, explicit tea procedure, task/state ownership and SOP conversation context |
| `application/tea_events.py` | Typed tea SOP state/notices and participant lifecycle topics |
| `application/tea_event_viewer.py` | Project application events into the SDK browser viewer |
| `components/__init__.py` | Public reusable flow/state exports |
| `components/visual_verification_flow.py` | Independent Omni/Clef verification and confirmation window |
| `components/vision_rag_retrieval_flow.py` | Generic visual extraction, retrieval and missing-field resolution |
| `components/timer_flow.py` | Native timer and periodic full state |
| `components/sop_run_recording.py` | Structured state, inference and image recording from caller-supplied snapshots/source directory |

Camera acquisition uses the SDK `CurrentFrameTool` and `ImageRegistry` directly
in `application/tea_application.py`. The runner receives the frame tool and the visual query tool uses the
same registry. The SDK router serializes participant departure/rejoin; its cleanup
hook drains SOP work before releasing cached frames and image handles. There is
no sample image agent or cleanup acknowledgment protocol. Shutdown clears the
registry after owned work ends.

## SOP run capture

Every launch creates `artifacts/runs/<UTC timestamp>-<run ID>/`. With
`capture.enabled: true`, the runner records each flow's initial/final state,
full state callbacks, acquired frame metadata and actual model requests/results.
The existing typed service wrappers save exact consumed image bytes before
inference, addressed by SHA256. The flow functions have no capture dependency.

```text
runs/<run>/
  manifest.json
  config/worker.json
  config/models.json
  config/voice_gate.json
  attempts.jsonl
  images/<SHA256>.jpg
  participants/<participant>/steps/<step>/flow/<attempt>/events.jsonl
  relay-events.jsonl
```

The manifest identifies the Python procedure's steps and source/config hashes.
There is no workflow YAML or resolved-interpreter snapshot. Custom foreground
and router capture scopes are removed; SDK Relay tracing remains supplemental.
Existing recorded runs are untouched.

Known credentials are redacted and headers omitted. Inference images are local
participant data; runs are not automatically deleted. `capture.enabled: false`
disables structured attempts/images while retaining manifest/config/participant
logs. Continuous media recording is separately opt-in below.

Writes are serialized outside the event loop and drained before cancellation
propagates. Input-image write failures prevent inference; incomplete runs remain
identifiable by missing finish records or a running manifest after process loss.
Recording adds disk latency but no models. For annotations, put expected judgments
and notes beside the attempt index, keyed by attempt ID; do not replace actual
model output.

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

All application prompts and settings live in `yaml/tea_making_worker.yaml`.
The `sop` mapping supplies arguments to the runner; it does not define a graph,
state-write bindings or tool dispatch. There is no per-flow YAML, packaged
prompt directory or hidden prompt fallback. Copy the complete main YAML for a
custom configuration. Reusable flow functions can be called entirely from Python.

The launcher materializes absolute paths for models, voice gate and artifacts.
Relative paths resolve against the selected worker YAML. Source startup finds
the checked-in YAML; an installed worker started from the sample directory finds
`yaml/tea_making_worker.yaml` there. Otherwise pass an explicit `--config`.

| File | Responsibility |
|---|---|
| `yaml/tea_making_worker.yaml` | SOP prompts/criteria/defaults/bounds, conversation instructions and worker services |
| `yaml/models.local.json` | Typed model adapters/shared endpoints |
| `yaml/voice_gate.yaml` | Conversation/wake addressing |
| `yaml/rag_service.yaml` | Retrieval documents, embedding/cache/chunking/threshold |
| `yaml/device_io_hub.yaml` | Transport, room and client configuration |
| `yaml/media_capture.yaml` | Opt-in continuous media recording |

Restart the worker after configuration changes. Worker prompt/flow changes do
not require restarting model services. A server/profile change follows
{doc}`/guides/customizing-model-servers`. The local embedding role is
`nvidia/Nemotron-3-Embed-1B-BF16` on port 8109 with native 2048-dimensional vectors;
the external NIM profile uses `nvidia/nemotron-3-embed-1b`. See
{doc}`/guides/deploying-with-nim`.

## Live inspection and durable output

The worker starts a generic loopback event viewer before it announces voice
readiness. Open `http://127.0.0.1:8093` on the XR-AI host, or pass
`--expose-web-events` to bind all IPv4 interfaces and browse
`http://<xr-host>:8093` from a trusted development network.
`TeaWebEventsAgent` subscribes to SOP state, SOP notices and participant lifecycle
topics and publishes compact `WebEvent` payloads. There is no runtime wildcard
subscription or file tailing. The sample has no general change watcher,
transcript-summary service or video journal.

The viewer keeps only a bounded in-memory history for the current process. The
structured SOP bundles under `artifacts/runs/<run>/` remain the durable record and include
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

Register the subscriber in `application/tea_application.py` beside the SOP event viewer adapter. A customer system can consume
SOP state records and notices independently.
If it runs in another process, expose the boundary through a typed msgpack over
ZMQ service rather than bypassing the repository's service model.

## Safety

This sample demonstrates agent composition; it is not a safety controller.
Keep hot vessels stable, follow appliance and tea-package instructions, and do
not use visual inference as the sole protection against burns, spills, or
electrical hazards.

## Lifecycle invariants

- SDK routing owns turn tasks; the runner owns the separate SOP procedure and flow tasks.
- General questions and speech interruption do not implicitly cancel SOP execution.
- Explicit stop/reset drains all owned work; skip stops only the current step.
- Session/run/task identity guards discard late state from stopped or replaced executions.
- Participant departure closes admission before draining tasks and releasing images.
- Producer acknowledgments retain the original departure generation; reconnect gets fresh state.
- State callbacks are awaited in the flow task. No callback is detached.
- Flow errors/cancellation propagate. The runner reports participant-procedure failures explicitly.
- No model is given a workflow mutation tool.

## Validation strategy

CPU tests exercise the actual public async flows, runner procedure, SDK router,
SDK Conversation request messages, source provenance, partial skip/defaults,
window unknown/duplicate rules, timer timing, callback bounds, scoped parallel
flow state, stop supersession, departure/rejoin and exact consumed-image capture.

Tests of the deleted workflow interpreter and foreground preparation paths are
replaced with these behavior tests; retaining old import shims would preserve
the architecture being removed.

From `agent-samples/tea-making-sample/`, replay existing healthy model endpoints:

```bash
uv run --project worker python eval/eval.py
uv run --project worker python eval/eval.py --cases eval/qualification-cases.yaml --output /tmp/tea-routing.jsonl
```

The evaluator uses production runner, router and Conversation with camera
acquisition unavailable. It checks control-vs-Q&A routing and authorization, not
physical image accuracy or audible latency. The original eight held-out wordings
are retained, with expectations updated for the two-agent contract; the old
background-control routing result is not directly comparable.

Qualify the visual pipeline on real/recorded images, including below-target,
at-target, Fahrenheit, unreadable units, empty vessels and occluded tea contact.
Source/quote checks and CPU tests do not establish model accuracy.

The model launcher reuses speech, Pocket TTS, Omni and embedding roles and includes
Clef. Sharing Omni for language and vision still loads one Omni checkpoint.
Concurrent flow calls reuse those clients but may contend for inference/KV
capacity. No new GPU memory/latency qualification is claimed; measure peak load
on the chosen profile.

<a id="what-should-become-shared"></a>

## What belongs in shared code

Keep the independent Python flow APIs in the sample until concrete consumers
justify SDK promotion. Reuse SDK routing, context, conversation, typed tools and
services now. The tea procedure, defaults, prompts and connections are application
code; no generic execution framework is needed to express them.
