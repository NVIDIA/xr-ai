<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making main-integration plan

## Goal

Replace the current monolithic tea-making pull request with a sequence of small,
reviewable pull requests. General application management, typed event delivery, agent, workflow,
background-processing, deterministic-tool, voice-output, and observability
components move to shared packages. The tea sample retains only composition,
tea behavior, prompts, documents, model/profile selection, and sample-specific
evaluation.

The migration must preserve the behavior proven by the current branch:

- exactly one foreground agent is invoked for a voice turn;
- root routing is bypassed while a foreground application owns the session;
- background applications never capture foreground;
- tea steps advance only after an explicit user command;
- observation continues after a step is complete, with no further state change;
- every state mutation passes through one typed commit function;
- every model call receives only its minimum task-local context;
- new clients and restarted workers receive fresh state;
- continuous observation and transcript outputs can remain text-only;
- prompts, tool lists, transitions, and sample policies remain declarative;
- unexpected service, configuration, and programming failures propagate.

## Baseline and source-of-truth rule

Plan baseline on 2026-08-10:

- source branch: `devdeepr/tea-making-sample` at `f1dacdc`;
- target: `origin/main` at `cab8ad3`;
- merge base: `5fe9901`;
- source-only history: 29 commits;
- current net comparison: about 10,000 added lines across more than 130 files.

The current branch is a behavioral reference and test oracle, not the branch to
merge. Every implementation pull request must start in its own git worktree from
the latest `origin/main`. Port the final behavior, tests, and documentation for
one concern; do not cherry-pick the cumulative changelog, root documentation, or
mixed commits from this branch.

This matters because the branch predates changes now on `main`. A direct rebase
would mix real tea work with reverse diffs for newer CI and Video MCP changes,
duplicate already-merged vLLM work, and make infrastructure fixes inseparable
from the sample.

## Target architecture

```text
VoiceSession
  |-- accepted command -> application.request event -+
  |                                                  v
  |                                      ApplicationManager
  |                               +----------+----------+
  |                               |                     |
  |                         no foreground        foreground active
  |                               |                     |
  |                         root tool agent       direct application
  |                               |                dispatch; no root LLM
  |             +-----------------+------------------+
  |             |                 |                  |
  |        inline NAT tool   launch foreground   control background
  |                               |                  |
  |                     GuidedWorkflowApp       background registry
  |                    current step variant       |      |      |
  |                    voice or observe loop   transcript vision vision
  |
  +-- every finalized STT turn -> voice.transcript event -> active subscribers

Background NAT function -> application.fact event -> context recorder
Any NAT function -------> user.output event -------> text NAT consumer
                                                \--> serial voice NAT consumer

GuidedWorkflowApp tick
  due -> configured NAT trigger -> plain observation
      -> standard agent step -> typed commit -> deterministic completion
      -> keep observing; transition only through explicit voice advance
```

The application manager owns the foreground stack and background membership.
The coordinator owns participant lifecycle, locking, and invocation context.
Each application owns its private participant state. The manager reads foreground state
before any model call, so nested routing never becomes a cascade of LLM
classifiers.

`xr_ai_nat.events` supplies typed participant-scoped delivery between NAT
functions. It validates payloads, carries event, correlation, and causation
metadata, and invokes selected NAT subscribers. It is not an agent scheduler,
prompt engine, state machine, or replacement for NAT.

The guided-workflow library gives every step one typed NAT function contract
and provides one standard agent-backed function. The YAML loader constructs
standard function configs. A future application can supply any compatible
code-defined NAT function without adding engine branches or a parallel
step-execution framework.

## Naming contract

The shared API describes applications and ownership, not a desktop, operating
system, shell, or terminal. Those metaphors helped explain the behavior but are
too restrictive for a reusable SDK.

Use these names in new shared code:

| Replace | With |
|---|---|
| `Desktop` | `ApplicationManager` |
| `DesktopRuntime` | `ApplicationManagerRuntime` |
| `DesktopState` | `ApplicationManagerState` |
| `DesktopSpec` | `ApplicationCatalog` when a manager-level value object is needed |
| `ApplicationSpec` | `ApplicationDescriptor` for routing/ownership metadata |
| `FunctionEffect` | `InvocationEffect` |
| `desktop__status` | `application_manager__status` |
| `desktop.*` events | `application_manager.*` events |
| `desktop/` package | `applications/` package |

Prefer `foreground`, `background`, `capture`, and `release` for the actual
ownership behavior. Avoid public names containing `desktop`, `OS`, `terminal`,
`shell`, or `launcher`; the component can manage voice, text, XR, or future input
surfaces without implying a particular user interface.

The sample file `yaml/applications.yaml` may retain its name because it is a
plain composition manifest, not a desktop abstraction. The old `desktop/*`
paths appear below only as migration sources and must not survive in the final
sample or shared package.

## Final ownership

| Current source | Final owner | Decision |
|---|---|---|
| `agents/factory.py`, `agents/invoke.py` | `agent-sdk/xr-ai-nat/xr_ai_nat/agents/` | Shared tool-agent construction, bounded schema retry, and compact invocation helpers. |
| `desktop/types.py` | `xr_ai_nat.applications` | Migrate the old path into shared invocation-route metadata and inline/foreground/background effects. |
| `desktop/runtime.py`, `desktop/registry.py` | `xr_ai_nat.applications` | Migrate the old paths into the shared application manager and deterministic foreground dispatch. |
| `desktop/functions.py`, `applications/controls.py` | `xr_ai_nat.applications` | Generated NAT lifecycle and status controls. |
| `runtime/scope.py` | `xr_ai_nat.applications.context` | Participant, trace, active application, operation, and private call context. No tea session type. |
| typed event topics and delivery | `xr_ai_nat.events` | Shared participant-scoped delivery to selected NAT function subscribers. |
| periodic event sources | `xr_ai_nat.events` | Shared supervised, participant-scoped timers owned by the applications that consume them; no global polling broadcast. |
| `applications/transcript*.py` | `xr_ai_nat.applications.transcript` | Shared opt-in raw transcript recorder and periodic summary application. Persist through `xr_text_memory`, not a second transcript store. |
| `applications/change_watch.py`, `video_log*.py` | `xr_ai_nat.applications.vision_observer` | One configurable rolling-caption observer used twice with different sample policies and sinks. |
| `applications/jsonl.py` | `utils/xr-ai-logging/xr_ai_logging/jsonl.py` | Move the generic append/session helpers outside the sample. Do not use them to duplicate transcript ownership. |
| `spec.py`, `runtime/state.py`, `engine/triggers.py` | `xr_ai_nat.workflows.guidance` | Code-first typed workflow, deterministic state machine, trigger resolution, completion, and manual transitions. No YAML dependency. |
| observation parts of `agents/registry.py` and `engine/coordinator.py` | `xr_ai_nat.workflows.guidance` | Standard agent-backed step adapter and homogeneous observation loop. |
| `functions/workflow.py` | `xr_ai_nat.workflows.guidance` | Generated typed commit/start/advance/reset/restart/status functions with a configurable namespace. |
| `functions/clock.py` | `xr_ai_nat.functions.clock` | Shared deterministic `now` and timer functions. |
| `functions/temperature.py` | `xr_ai_nat.functions.measurements` | Shared explicit reading/target/unit comparison. Remove the implicit tea-state dependency. |
| `functions/vision.py` | shared scoped vision adapter | Bind participant identity from invocation context while delegating frame acquisition and VLM calls to `xr_vision_tools`. |
| `functions/rag.py` | mostly delete | Use `xr_rag` directly. Put general result bounds and telemetry in the shared capability; keep only tea route wording in sample metadata. |
| `engine/notices.py` | delete after `xr-ai-voice` extraction | Use direct serialized assistant output; no synthetic secret query tokens. |
| labeled data output | sample composition or a later hub-output SDK | Keep application labels and destination policy out of the voice transport package. |
| `tea_making_viewer/` | `utils/xr-ai-activity-viewer/` | Shared stdlib-only, read-only multi-source activity viewer. Sample keeps only its source configuration. |
| `runtime/render.py` | split | Shared workflow accepts a renderer callback. Tea keeps its `duration` and `temperature_c` formatter registration and message templates. |
| `runtime/events.py` | injected event sink | Shared runtimes emit a stable event contract through an injected sink; the sample chooses the repository logging implementation. |
| `eval/` | tea sample | Cases, prompt budgets, state matrices, and live evaluations are tea behavior. Extract only a helper already used by another sample. |
| `yaml/`, `rag-documents/`, launch CLI, prompts | tea sample | Workflow and deployment policy remain sample-owned. |

## Shared API shape

Names are illustrative, but the contracts should remain this small. NAT
`Function` is the executable composition type; the manager must not define a
parallel application execution interface.

```python
@dataclass(frozen=True, slots=True)
class RoutedFunction:
    ref: FunctionRef
    route: str
    effect: InvocationEffect = InvocationEffect.INLINE
    return_direct: bool = False

@dataclass(frozen=True, slots=True)
class ApplicationDescriptor:
    id: str
    title: str
    mode: ApplicationMode
    route: str

class ApplicationTurn(BaseModel):
    request: str

@dataclass(frozen=True, slots=True)
class ApplicationMount:
    descriptor: ApplicationDescriptor
    turn: Function[ApplicationTurn, None, str]

class ApplicationRequest(BaseModel):
    text: str

APPLICATION_REQUEST = EventTopic("application.request", ApplicationRequest)
```

The manager invokes only mounted NAT functions. Background start, stop, and
status are NAT functions too. External sources publish typed events, and the
thin dispatcher invokes registered NAT functions; it does not define another
application interface. A sample YAML loader may construct descriptors and
function configs, but shared code must not import YAML or assume an
`applications.yaml` file.

The guidance boundary is similarly code-first:

```python
@dataclass(frozen=True, slots=True)
class GuidanceStepSpec:
    id: str
    title: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    complete_when: Mapping[str, object]
    next_step: str | None
    state_on_skip: Mapping[str, object]

class StepTurn(BaseModel):
    operation: Literal["observe", "answer"]
    value: object
```

Each step references a typed `Function[StepTurn, None, str]`. The standard
agent-backed function is configured with a trigger, focus prompt, observation
prompt, voice prompt, and tool lists and is the only implementation needed by
the tea YAML. Any compatible composite NAT function can replace it, including a
router, pipeline, or application turn adapter, without changing the engine.

## Pull-request sequence

Independent infrastructure pull requests can proceed in parallel. Shared
library pull requests should merge in the order shown. Each dependent branch is
created only after its predecessor lands, avoiding a long fragile stack.

### PR 1 — expose final pre-gate transcriptions

Branch: `devdeepr/voice-transcription-observer`

Scope:

- add `transcription_observer` to `VoiceSession.run()`;
- deliver every finalized user STT turn before wake-word filtering;
- keep the existing turn observer limited to accepted user queries and agent
  responses;
- document ordering and privacy/opt-in expectations;
- add processor and session tests.

Exclude early-chime timing, filler matching, tea code, and transcript storage.
This PR is required by the reusable transcript application.

Acceptance:

- an ungated utterance reaches only `transcription_observer`;
- an accepted command reaches the pre-gate observer before normal routing;
- typed text input does not masquerade as raw microphone transcription;
- exceptions follow the existing observer policy and cannot silently corrupt
  gate state.

### PR 2 — add direct voice-session output adapters

Branch: `devdeepr/voice-application-output`

Scope:

- add a public way to enqueue an already-decided spoken response without
  feeding it back through an agent handler;
- add a public way to publish labeled text data to one participant without
  TTS;
- preserve participant isolation, queue ordering, interruption, and cleanup;
- replace the behavior currently implemented by `NoticeBridge` and
  `TextOutputBridge` in tests.

Exclude application routing and tea-specific topics or labels.

Acceptance:

- direct announcements use the normal TTS/output queue exactly once;
- labeled text never enters STT, the handler, transcript observation, or TTS;
- output after participant release is rejected or dropped with a documented
  result rather than retained indefinitely.

### PR 2a — add typed NAT-function event delivery

Branch: `devdeepr/nat-typed-events`

Scope:

- add `EventTopic`, `EventEnvelope`, and `EventDispatcher` to `xr-ai-nat`;
- validate every topic payload with its Pydantic model;
- carry participant, producer, event, correlation, causation, and timestamp
  metadata;
- deliver only to explicitly selected registered NAT functions;
- add the NAT event-handler adapter and an optional observability callback;
- document that the dispatcher owns no prompts, agents, schedules, retries,
  foreground state, or persistence.

Exclude application management, voice transport, tea topics, queues, brokers,
and durable messaging.

Acceptance:

- invalid payloads fail before any subscriber runs;
- participant and trace metadata survive delivery;
- explicit subscriber selection prevents unintended fan-out;
- subscriber order and exception propagation are deterministic;
- the executable consumers are NAT functions.

### PR 3 — add shared NAT tool-agent primitives

Branch: `devdeepr/nat-agent-primitives`

Scope:

- move the compact `ToolCallAgentWorkflowConfig` builder into
  `xr_ai_nat.agents`;
- move one-retry handling for Pydantic tool-argument failures there;
- expose routed-function metadata and catalog rendering;
- expose a task-neutral invocation context carrying participant ID, trace ID,
  selected operation, and a private per-call mapping;
- keep tool errors disabled in NAT so unexpected failures retain their full
  traceback;
- add package-level unit tests and prompt/history defaults as explicit config.

The retry must remain narrow: one corrected call only for a model-generated
schema validation error. It must not catch service, timeout, configuration, or
application exceptions.

Acceptance:

- no dependency bypasses `xr-ai-models`;
- the base package remains importable without optional agent dependencies;
- tool lists and return-direct lists are deterministic;
- repeated invalid tool arguments may be surfaced or skipped according to an
  explicit caller policy.

### PR 4 — add the foreground application manager

Branch: `devdeepr/nat-application-manager`

Depends on PR 3.

Scope:

- add code-first application descriptors;
- add per-participant foreground stack, background set, revision, and lock;
- add deterministic dispatch that calls the root agent only when no foreground
  application owns the participant;
- add foreground capture/release and generated background start/stop/status
  NAT functions;
- compose participant join/leave/reset lifecycle and background
  transcription/tick topics through the typed event dispatcher;
- inject a structured event sink;
- provide a minimal test application proving nested foreground return and a
  background application proving concurrent activity.

The public modules, types, NAT function names, and events must follow the
application-manager naming contract above. Do not preserve `desktop` aliases: this
is a new API, and carrying both vocabularies would create avoidable compatibility
surface before the first shared release.

Exclude YAML loading, tea state, voice transport, vision, RAG, and concrete
background applications.

Acceptance:

- one and only one foreground model is invoked per voice turn;
- root tools are unavailable while a foreground application is active;
- inline and background actions do not change foreground ownership;
- nested foreground release returns to its caller;
- reconnect and process restart produce fresh application state;
- lifecycle operations are serialized per participant but different
  participants remain independent.

### PR 5 — add the reusable transcript and periodic-summary application

Branch: `devdeepr/nat-transcript-application`

Depends on PRs 1, 3, and 4.

Scope:

- add an opt-in background transcript application;
- receive finalized pre-gate `VoiceTurn` values;
- persist raw utterances through `xr_text_memory__add_transcript` using a
  documented source ID convention;
- summarize only unsummarized turns at a monotonic configured interval;
- send summaries through an injected labeled-text output callback;
- configure title, route, interval, source naming, summary prompt, and output
  behavior in code;
- add lifecycle, concurrency, bounded-context, and prompt-agent tests.

Delete the sample-local transcript JSONL store when the sample migrates. The
shared text-memory capability remains the only raw transcript owner.

Acceptance:

- recording continues without a wake word only after explicit start;
- stopping/releasing flushes pending writes and prevents future capture;
- each summary sees only pending utterances, not full conversation history;
- a failed model/tool call leaves turns pending for a later interval;
- transcript text is not duplicated into standard diagnostic logs.

### PR 6 — add a reusable rolling vision-observer application

Branch: `devdeepr/nat-rolling-vision-observer`

Depends on PRs 3 and 4.

Scope:

- consolidate change watching and general video delta logging into one
  application implementation;
- accept a current-frame NAT function, caption focus, comparison policy,
  interval, history bound, optional per-session instruction, event sink, and
  optional labeled-text sink;
- perform one VLM caption call followed by one small tool-agent decision over
  only the current caption and bounded prior captions;
- add a shared structured JSONL event sink for derived observation records;
- instantiate two distinct policies in tests to prove reuse;
- keep unavailable frames and repeated repairable tool failures deferred to a
  later tick.

The library owns no tea prompts and no fixed definition of “important.” The
sample config creates:

- a focused change watcher with a two-caption history and important-only UI
  output;
- a broad activity logger with a five-caption history and no spoken output.

Acceptance:

- exactly one caption is added per successful tick;
- the comparison input contains no voice history, tea state, or unbounded
  captions;
- view, wording, and lighting-only changes can be suppressed by policy;
- file writes do not block the event loop;
- each participant has isolated focus, history, schedule, and output.

### PR 7 — add the code-first guided-workflow core

Branch: `devdeepr/nat-guidance-core`

Depends on PR 4.

Scope:

- add task-neutral state field, workflow, step, trigger, evidence, message, and
  transition value objects;
- add a deterministic participant workflow store with sparse projections,
  typed write boundaries, atomic commits, completion checks, evidence counts,
  skip patches, manual advance, restart, reset, and notice draining;
- make each step reference a typed NAT function and add renderer/event-sink
  injection points;
- add generic trigger argument resolution for participant and projected state;
- explicitly reject automatic advancement in this first contract;
- add exhaustive pure unit tests without models, hub, voice, or YAML.

Exclude YAML parsing, NAT tool-agent construction, tea fields, regexes, prompts,
and formatters.

Acceptance:

- completion never changes the current step;
- completed steps still accept homogeneous no-op observation commits;
- writes outside the active step fail deterministically;
- false completion cannot pass an unmet evidence threshold;
- reset/restart/skip/advance semantics match the current tea characterization
  tests;
- callers may supply any schema-compatible NAT function without changing the
  engine.

### PR 8 — add the standard agent-backed guidance adapter

Branch: `devdeepr/nat-guidance-agent`

Depends on PRs 3 and 7.

Scope:

- implement the standard observation and voice step function with built-in NAT
  tool-calling agents;
- add one homogeneous scheduled observation loop;
- generate commit/start/advance/reset/restart/status NAT functions with a
  configurable prefix;
- build one voice agent variant per step and select it deterministically before
  the model call;
- generate only the active step's writable-state contract;
- expose generic compact prompt fragments and make all policy prompts caller
  supplied;
- preserve event names/fields needed to diagnose trigger, evidence, agent,
  commit, readiness, and transition behavior;
- add fake-agent and NAT-build tests.

Exclude tea wording, tea tool lists, YAML, RAG descriptions, and temperature
policy.

Acceptance:

- observation input is only `observation`, prior completion status, and sparse
  step state;
- foreground input is only the user request and sparse step state;
- no conversational history is added by the adapter;
- completed observations call commit with no mutation and no notice;
- explicit next/skip is the only transition path;
- a model tool-schema error gets the shared single retry and no broader catch.

### PR 9 — add shared clock and measurement NAT functions

Branch: `devdeepr/nat-clock-measurements`

Independent of the application manager and guidance adapter.

Scope:

- add an `xr_clock` function group with current epoch microseconds and fresh
  elapsed/remaining/expired timer results;
- add a deterministic measurement comparison function with explicit reading,
  reading unit, target, and target unit;
- support Celsius/Fahrenheit normalization without reading application state;
- make the clock source injectable in tests;
- document schemas and add NAT discovery/build/execution tests.

Acceptance:

- all inputs are explicit and strictly validated;
- no function mutates workflow state;
- output remains numeric and domain-neutral; human formatting remains with the
  calling application;
- invalid units or non-finite values fail validation rather than being guessed.

### PR 10 — move the activity viewer to shared utilities

Branch: `devdeepr/activity-viewer`

Independent of model and voice work.

Scope:

- create `utils/xr-ai-activity-viewer/` as a stdlib-only package/process;
- move the HTTP server, bounded event store, incremental complete-line tailer,
  source-scoped offsets, decoders, and static UI;
- make every pane, source title, path/directory, format, filters, host, port,
  polling interval, and event limit configuration driven;
- support generic JSONL, text-memory JSONL, and repository structured event-log
  sources;
- include simultaneous responsive panes and independent follow/scroll state;
- add unit tests for truncation, partial lines, duplicate paths under different
  sources, filters, ordering, and HTTP responses;
- update `DEPENDENCIES.md` and utility documentation.

Exclude tea labels, artifact paths, topics, and event prefixes. Those stay in
the sample's viewer config.

Acceptance:

- startup baselines old records and displays only new activity;
- two configured views of the same physical log do not consume each other's
  cursor;
- the process is read-only and never enters hub IPC, agent routing, model calls,
  or TTS;
- the package has zero runtime dependencies.

### PR 11 — unify the shared Omni and Cosmos model-server stack

Branch: `devdeepr/model-servers-omni-cosmos`

Independent infrastructure PR. Rebuild it from current `main`; do not reuse the
stale model-server branch wholesale.

Scope:

- launch STT, Nemotron Omni, Cosmos, and embeddings as one persistent shared
  stack;
- keep sample choice at the model deployment profile: Omni may serve both
  roles, or Cosmos may serve vision while Omni serves agent reasoning;
- encode per-profile GPU placement and memory budgets in model-server YAML;
- remove unsupported Blackwell Triton MoE forcing;
- stop only genuinely obsolete persisted services;
- update model-server tests, AI-service docs, quickstart, and consumer examples
  in a focused follow-up if needed.

The vLLM setup-before-entrypoint fix is already on `main` as PR 340 and must not
be included again.

Acceptance:

- one model-server command supports both sample model modes without restart;
- all declared profiles have valid service names, ports, GPU assignments, and
  cache paths;
- failed cleanup prevents overcommit rather than continuing;
- Blackwell config uses a supported MoE backend;
- model-server unit tests do not require GPUs.

### PR 12 — harden early wake acknowledgement

Branch: continue/rebase `devdeepr/fix-early-wake-chime`

Independent infrastructure PR.

Scope only the existing early partial-STT probe, bounded leading speech fillers,
and prevention of a late fallback chime. Keep it separate from the pre-gate
transcription observer so reviewers can evaluate latency behavior independently.

Acceptance:

- wake acknowledgement occurs during the utterance when the partial probe
  recognizes the phrase;
- final STT waits for an already-running probe rather than racing it;
- a missed early probe never inserts a chime at response start;
- arbitrary preceding conversation still cannot activate the gate;
- voice pipeline tests cover timing, cancellation, and participant cleanup.

### PR 13 — add Magpie request-level speaking rate

Branch: `devdeepr/magpie-speaking-rate`

Independent and non-blocking for the tea sample, whose stable demo path remains
Piper.

Scope:

- add validated OpenAI-compatible `speed` handling to Magpie;
- keep the server default at native speed;
- isolate audio post-processing and test duration plus pitch preservation;
- update the Magpie dependency map and service documentation.

Do not include unfinished response streaming or sample launch changes. If the
added dependency or audio quality is not acceptable in review, drop this PR
without blocking tea integration.

### PR 14 — add the core tea-guidance sample

Branch: `devdeepr/tea-guidance-sample`

Depends on PRs 2–4, 7–9, and 11. PR 12 is desirable but not structurally
required; PR 10 is needed only by the follow-up background/viewer PR.

Scope retained in the sample:

- orchestrator and worker entry points;
- explicit `--model-mode`, `--voice-mode`, and `--tts-mode` selection;
- sample deployment profiles and process wiring;
- YAML-to-code loaders for application and guidance specs;
- tea workflow state fields, five steps, prompts, evidence rules, messages,
  transition destinations, and tool lists;
- tea RAG documents and RAG-service configuration;
- tea-specific response formatters;
- sample composition mapping shared capabilities and applications to IDs;
- tea unit tests, prompt budgets, route/state matrix, live-agent evals,
  README, root sample tour, and agentic development guide.

Do not reintroduce shared runtime modules under the worker. The intended worker
shape is roughly:

```text
tea_making_worker/
  __main__.py
  app.py                 # construct models, NAT groups, manager, events, and voice session
  config.py              # sample worker configuration
  composition.py         # bind tea IDs to shared components
  workflow_loader.py     # YAML convenience -> code-first guidance objects
  applications_loader.py # YAML convenience -> code-first app descriptors
  prompts.py             # only tea/sample policy not supplied by YAML
  formatting.py          # tea message formatter registrations
```

Use `xr_vision_tools` and `xr_rag` as the capability owners. Any alias needed
for concise model-visible names must use a shared scoped/bounded adapter rather
than a tea-specific HTTP or frame implementation.

Acceptance:

- all current deterministic worker tests are ported and passing;
- prompt sizes are no larger than the current enforced budgets;
- every prompt rule has an eval case in the same PR;
- typed and spoken next/skip/reset/restart/status routes pass in every active
  step and in root state;
- temperature readout and hot-enough questions remain separate and do not
  mutate state;
- Omni-only and Cosmos-plus-Omni modes use the same running model stack;
- reconnect and restart reset tea and foreground state;
- DCO, SPDX, dependency map, docs, and both Python-version CI lanes pass.

### PR 15 — compose background applications and the viewer in the tea sample

Branch: `devdeepr/tea-sample-background-apps`

Depends on PRs 5, 6, 10, and 14.

Scope:

- instantiate the shared transcript application, focused vision watcher, and
  broad visual activity logger from `yaml/applications.yaml`;
- configure text-memory sources and derived-observation artifact locations;
- configure the shared activity viewer's transcript, visual log, watcher, and
  agent panes;
- connect finalized STT turns to the application manager and labeled output to the
  web client;
- add background prompt evals, lifecycle tests, UI contract tests, README, and
  agent-guide updates.

This final PR should add configuration and composition, not copies of library
implementations.

Acceptance:

- background start/stop/status routing works only from root while active
  background work continues alongside a foreground tea guide;
- transcript recording does not require the wake word after explicit start;
- periodic summaries and watcher notices appear as labeled text without TTS;
- video activity and watcher outputs persist separately and appear
  simultaneously in the viewer;
- the root prompt and all background prompts remain within current budgets;
- no background data is added to foreground or tea-step model context.

### Optional PR 16 — reorder the generic web-client debug panels

Branch: `devdeepr/web-video-data-layout`

The current branch only moves the Data Channel and Received sections below the
video. If this remains desired, submit it as a one-file client PR with a visual
smoke check. It is unrelated to tea behavior and must not block the sample.

## Pull-request dependency graph

```text
PR 1 transcription observer -----> PR 5 transcript app --------+
                                                                  |
PR 2 voice outputs ------------------------------+                 |
PR 2a typed events ------------------------------+                 |
                                                 |                 |
PR 3 NAT agent primitives -> PR 4 app manager ---+--> PR 14 core -+-> PR 15 apps/viewer
                 |              |                |                 |
                 |              +-> PR 5 --------+                 |
                 |              +-> PR 6 vision observer ----------+
                 |                                                |
                 +-> PR 8 guidance agent                           |
PR 7 guidance core ---------> PR 8 ------------------------------->+
PR 9 clock/measurements ------------------------------------------>+
PR 10 activity viewer --------------------------------------------+-> PR 15
PR 11 model stack ------------------------------------------------>+

PR 12 wake timing, PR 13 Magpie speed, and PR 16 web layout are independent.
```

## Migration procedure for every PR

1. Refresh `origin/main`.
2. Create a dedicated worktree and a `devdeepr/<topic>` branch from the latest
   main. Never develop these changes in the primary checkout or reuse the
   monolithic branch as the PR base.
3. Port the smallest coherent contract and its characterization tests.
4. Keep existing public names and event fields during extraction unless the PR
   is explicitly an API-change PR. Configurable prefixes can preserve current
   `workflow__*`, `change_watch__*`, and similar tool names during migration.
5. Update package README, root documentation, `DEPENDENCIES.md`, and
   `docs/changelog.md` in the same PR when their contracts change.
6. Run focused unit tests, root contract tests, `git diff --check`, SPDX checks,
   and lock checks for every changed project.
7. Commit with DCO sign-off and push only the named `devdeepr/*` branch.
8. Merge before creating the next dependent worktree, or explicitly base a
   short-lived stacked PR on its direct predecessor and rebase it immediately
   after that predecessor lands.

## Validation gates

### Shared library PRs

- public imports work with only the declared base dependencies;
- optional `agents`, `vision`, and `voice` imports fail only when used, not at
  base package import time;
- no worker imports from `server-runtime` or `xr_ai_launcher`;
- all model HTTP goes through `xr-ai-models`;
- all agentic tools are typed NAT functions;
- multi-participant lifecycle and concurrency tests pass;
- unexpected failures retain their original traceback.

### Tea core PR

- root tea contract tests;
- worker unit and NAT build/execution tests;
- fast prompt/schema/evidence checks;
- live Omni route matrix;
- live observation cases for every step;
- one end-to-end human smoke run in Omni mode and one in Cosmos vision mode;
- log review verifies compact requests, one selected foreground, one commit per
  observation, manual transitions, and no repeated notices.

### Background/viewer PR

- transcript capture before gate and only while active;
- periodic summary retry and pending-turn preservation;
- rolling visual history bounds and independent participant state;
- text-only output topics and no TTS frames;
- viewer cursor, filtering, partial-line, truncation, and same-path/multi-source
  regressions;
- browser smoke at the configured host/port with all panes visible together.

## Changes to discard or reconstruct

- Drop commit `2b60134`; its vLLM setup-before-entrypoint fix is already on
  main through PR 340.
- Do not replay merge commit `894dc79`; use the focused wake branch and current
  main.
- Do not port reverse diffs for newer nightly CI, Video MCP startup, vLLM
  config, or other files changed on main after this branch diverged.
- Recreate changelog and root README entries per focused PR. Do not copy the
  monolithic branch's cumulative entries.
- Delete sample-local raw transcript storage when `xr_text_memory` is wired.
- Delete duplicated RAG and frame acquisition implementations; shared NAT
  capabilities remain authoritative.
- Do not carry private `__guide_notice_*` query tokens after direct response
  output exists in `xr-ai-voice`.
- Do not add conversation history while extracting the agent builders.
- Do not change prompt wording, evidence thresholds, state semantics, or event
  fields in nominally non-functional extraction PRs. Behavioral tuning belongs
  in a later focused sample PR with a new eval case.

## Completion criteria

The migration is complete when:

1. the original `devdeepr/tea-making-sample` PR is closed as superseded;
2. every general component listed in the ownership table is in a shared package
   or has been deliberately deleted in favor of an existing shared capability;
3. the tea worker contains only sample configuration, loaders, composition,
   formatting, prompts, and entry points;
4. the current route, workflow, background, prompt-budget, and observability
   tests pass against the rebuilt sample;
5. both supported model modes run against one shared model-server stack;
6. the agent guide describes the final library boundaries and human-test log
   procedure;
7. all merged commits are DCO-signed and every branch follows the
   `devdeepr/<topic>` naming rule.
