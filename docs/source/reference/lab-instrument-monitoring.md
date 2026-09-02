<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lab instrument monitoring — building marker-associated visual agents

The lab instrument sample is a reference for applications that combine
foreground questions, opt-in background work, persistent participant state,
visual identification, and event-driven notifications. Refer to the
{doc}`quickstart </getting_started/quickstart>` to run the sample. This
architecture reference focuses on reusing the sample's structure for another
application.

The central design choice is to separate perception, interpretation, state,
policy, and presentation. A marker scan identifies an instrument, a VLM reads
its display, a stateful agent decides whether the reading changed, and a
separate subscriber decides whether to speak. Each concern can therefore be
replaced without rewriting the others.

## What to copy

Use this sample when your application needs one or more of these patterns:

- a foreground agent that selects participant-bound tools;
- visual queries without exposing participant IDs or raw images to the model;
- background tasks that start and stop through ordinary agent tools;
- state that survives intermittent visibility but is cleared on disconnect;
- typed events consumed independently by storage, speech, or a backend;
- deterministic identity before VLM interpretation;
- JSONL artifacts and captured source frames for manual debugging.

The instrument-reading behavior is sample-specific. The runtime, voice,
tool-loop, image-selection, visual-query, and event patterns are reusable.

## Architecture

```text
camera frames ────────────────> ParticipantImageAgent
                                      │
accepted speech or typed query         ├─> current frame + generic VLM query
              │                       └─> marker scan
              v                                │
        ForegroundAgent                        v
              │                       LabInstrumentAgent
              ├─> direct answer                │ one-frame readings
              ├─> visual monitor controls      v
              ├─> instrument controls   InstrumentMonitorAgent
              └─> current readings             │
                                               ├─> change event ─┬─> voice aggregation
                                               ├─> lost event ───┤
                                               │                 └─> JSONL/backend
                                               └─> state event ────> JSONL/backend
```

`VoiceAgent` publishes participant-scoped queries and lifecycle events into
`AgentRuntime`. Agents call peer tools directly, while typed topics fan results
out to independent consumers. LiveKit remains inside DeviceIOHub and does not
appear in the worker's agent contracts.

The worker uses native `xr_ai_runtime` agents and `xr_ai_tools` instances. It
does not use NVIDIA Agent Toolkit, PydanticAI, MCP clients, or MCP servers.

## Agent responsibilities

| Agent | Owns | Does not own |
|---|---|---|
| `ParticipantImageAgent` | Image registry, current-frame selection, marker tracking, participant cleanup | VLM prompts or instrument state |
| `ForegroundAgent` | One bounded tool loop per current query, supersession, participant-bound tool catalog | Conversation history or background scheduling |
| `MonitorAgent` | An optional generic visual-change task per participant | Instrument identity or alerts |
| `LabInstrumentAgent` | One-frame marker-associated reads and debug source images | Long-lived state or notification policy |
| `InstrumentMonitorAgent` | Reading normalization, change detection, last-seen state, loss detection, periodic snapshots | Speech or storage |
| `InstrumentAlertAgent` | Which instrument events become participant voice output | Tracking state |
| `VoiceAggregationAgent` | Participant speech pacing, ordering, and coalescing | Instrument or foreground decisions |
| `FileOutputAgent` | Session files, JSONL records, recent visual-history tool | Agent decisions |
| `WebEventsAdapterAgent` | Explicit sample-event projection into browser topics | HTTP serving or durable history |
| `WebEventsAgent` | Bounded in-memory history and read-only HTTP page | Application topic selection or persistence |

This ownership keeps periodic tasks and mutable state out of the runtime. The
runtime delivers typed events; each agent cancels its own participant tasks and
releases its own state.

For a `current_view` selection, `ForegroundAgent` uses the same direct path as
`simple-vlm-example`: `CurrentFrameTool` selects one frame and
`StreamingImageQueryTool` publishes chunks to participant voice. The tool is a
direct return, so the completed visual answer does not incur a second language
model call. A dedicated current-view system prompt applies visible-evidence,
plain-speech, and response-length rules directly to the streaming VLM output.
The default voice gate accepts `agent` and `hey agent`, plays a listening chime,
and allows a five-second follow-up. `VoiceAgent` mirrors spoken text to the
client's `agent.response` topic so the same output can be rendered as captions.

## Source map

| File | Purpose | Typical adaptation |
|---|---|---|
| `app.py` | Constructs services and registers agents | Add or replace an event consumer |
| `events.py` | Typed application topics and payloads | Define backend-facing facts, alerts, or audit records |
| `images.py` | Shared participant image acquisition | Add another image-selection tool |
| `foreground.py` | Current-query tool loop and participant tool composition | Change the assistant's available operations |
| `instruments.py` | Marker scan, marker annotation, and one-frame VLM read | Replace the identified-object perception flow |
| `device_map.py` | Resolves marker family and ID to a domain name | Load identities from another source |
| `instrument_monitor.py` | Stateful reading tracker and event policy | Change normalization, loss, or alert thresholds |
| `instrument_alerts.py` | Converts selected events to voice | Send UI notifications or suppress speech |
| `monitor.py` | Generic opt-in visual change monitor | Build a different periodic visual task |
| `file_output.py` | JSONL persistence and recent-history access | Publish records to a customer backend |
| `web_events.py` | Explicit runtime-topic projections for the browser viewer | Select or rename live presentation topics |
| `yaml/device_map.yaml` | Sample identity data | Map real marker IDs to domain objects |
| `sample-markers/` | Five QR and five ArUco examples | Print or replace with deployment markers |

Refer to the generated {doc}`configuration <configuration>` and
{doc}`command-line <command-line>` references for configuration fields and
command syntax.

## Configuration

Run and edit the sample from `agent-samples/lab-instrument-monitoring/`. The
orchestrator reads the files below on every start and materializes a temporary
worker configuration with absolute paths. Edit the checked-in files, not the
temporary copy named in the logs.

| File | Owns |
|---|---|
| `yaml/lab_instrument_monitoring_worker.yaml` | Monitor and snapshot cadence, lost-device threshold, image freshness, VAD, output directory, and event-viewer port and history |
| `yaml/device_map.yaml` | QR payloads and ArUco IDs mapped to instrument names |
| `yaml/voice_gate.yaml` | Wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | Reused model adapters, endpoints, and readiness checks |
| `yaml/device_io_hub.yaml` | LiveKit room and ports, web and token servers, and network behavior |

For example, shorten both visual polling periods by setting
`monitor_interval_s` and `instrument_monitor_interval_s` in the worker YAML,
or replace entries under `devices` in `device_map.yaml` with the identifiers
attached to real instruments. Paths such as `device_map_yaml` and
`artifacts_dir` are resolved relative to the worker YAML.

Restart `lab_instrument_monitoring` after an edit; configuration is not
hot-reloaded. The `--expose-web-events` option intentionally overrides
`web_events_host` in the runtime copy, so use that option rather than editing
the host to expose the unauthenticated viewer. Changing `models.json` changes
only the endpoints consumed by this sample. Refer to
{doc}`/guides/customizing-model-servers` for server-side model, GPU, port, or
memory changes, then restart the persistent shared stack.

Refer to the generated {doc}`configuration <configuration>` reference for exact
fields, checked-in values, and adjacent YAML comments.

## Foreground tool loop

Each accepted query gets only the system prompt and current utterance. The
foreground agent builds a `ToolSet` whose closures already contain the
participant ID, then calls the shared bounded `run_tool_loop`. The model never
selects a participant and never receives raw image bytes.

The tool catalog contains:

- current-view visual Q&A;
- recent visual-monitor history;
- start, stop, and status controls for generic monitoring;
- a one-shot marker-associated instrument read;
- start, stop, and status controls for continuous instrument tracking.

Control operations and one-shot reads use direct-return results. They do not
require another model round to restate a result that the tool already knows.
The loop executes tools sequentially so state-changing calls cannot race.

To add a foreground capability:

1. Implement a native `Tool` or `AsyncTool` and keep its request schema limited
   to model-selectable arguments.
2. Capture participant or application context in the tool closure.
3. Call another agent's public tool through `execute()` rather than its private
   handler.
4. Add the participant-bound facade to `ForegroundAgent._participant_tools`.
5. Add a routing eval that expects the new tool without reusing a prompt
   example verbatim.

## Visual identification and reading

`LabInstrumentAgent` implements a deterministic perception pipeline:

1. Select one current participant frame.
2. Optionally save that exact source image under `artifacts/marker-scans/` when
   `capture_marker_scans` is enabled.
3. Detect every QR and ArUco marker in the frame.
4. Resolve each marker through `DeviceMap`.
5. Create one derived image that overlays every detected polygon with a unique
   temporary label.
6. Ask the VLM once for a strict JSON map from every temporary label to its own
   display reading or `UNKNOWN`.
7. Return mapped `InstrumentSighting` values independently from successful
   `InstrumentReading` values.

The marker determines identity before the VLM reads the display. This prevents
the model from guessing which instrument produced a value. Image references,
not image bytes, pass between tools; media stays in the shared image registry.

To use a different identifier, replace the marker tool and `DeviceMap` while
preserving the one-frame `LabInstrumentReadResult` boundary. Barcode, OCR label,
object detection, or backend inventory lookup can all feed the same reader and
tracker shape.

## Stateful monitoring

`InstrumentMonitorAgent` owns one task group and tracker per participant. Its
scan loop calls the one-frame reader. Its maintenance loop publishes lost-device
events and complete state snapshots.

Readings are normalized before comparison. If a later VLM response omits a
unit, the previous known unit is retained. The agent publishes:

- `InstrumentChange` when a device is first discovered;
- `InstrumentChange` when its normalized value or unit changes;
- `InstrumentLost` once after the device exceeds the last-seen timeout;
- `InstrumentStateSnapshot` periodically, including tracking status.

A mapped marker refreshes last-seen state even when its display is temporarily
unreadable, so glare does not produce a false lost-device alert. A device moving
in and out of view does not repeatedly alert unless its reading changes. This is
application policy and belongs in the tracker rather than the VLM prompt or
voice agent.

The checked-in demo scans instruments every two seconds and marks an instrument
as lost after five seconds without a sighting. Spoken monitoring changes are
batched on a five-second cadence. They are non-urgent voice contributions, so
the sample's aggregation policy waits for estimated active playback to finish
and then leaves five seconds of quiet before delivering the next monitoring
update. Foreground responses remain urgent and can interrupt that hold.

Only marker identities present in `device_map.yaml` are treated as instruments.
Unknown QR payloads and ArUco IDs are logged and ignored, preventing detector
false positives from becoming names such as `ArUco 17`. They are still given a
temporary visual label so the VLM can reason about competing housings. The
one-frame reader requires visible evidence that each labelled marker and
display share one continuous physical instrument housing. Proximity, alignment,
or being the only readable display does not establish ownership, and one
display may not be assigned to multiple markers. The reader returns `UNKNOWN`
when a housing has no readable display or an adjacent display cannot be
excluded. These fixed rules are supplied as a system prompt; decoded marker
identities remain outside the prompt and visible text is treated as evidence,
never as instructions.

## Connecting a backend

Add a runtime agent that subscribes to the typed topics you need. Inject a
typed backend client into that agent; do not add vendor HTTP calls to the
perception or tracking agents.

```python
class InstrumentBackendAgent(Agent):
    def __init__(self, backend: InstrumentBackend) -> None:
        self._backend = backend
        super().__init__()

    @subscribe(INSTRUMENT_CHANGE_TOPIC)
    async def changed(
        self,
        event: InstrumentChange,
        ctx: RuntimeContext,
    ) -> None:
        await self._backend.publish_change(
            participant_id=ctx.metadata.participant_id,
            event=event,
        )
```

Register this subscriber beside `FileOutputAgent`. Storage and voice can remain
enabled during development and be removed independently for production. Use a
typed msgpack over ZMQ service when the backend boundary is a separate process.

## Live event viewer

`WebEventsAdapterAgent` subscribes explicitly to monitor, instrument change,
tracking-loss, state, foreground, transcript, and participant lifecycle topics.
It publishes compact `WebEvent` projections to the shared `WebEventsAgent`; it
does not tail JSONL files or subscribe by wildcard. The page at
`http://127.0.0.1:8092` groups those events by participant and presentation
topic while the JSONL files remain the durable record. Pass
`--expose-web-events` to bind all IPv4 interfaces for access at
`http://<xr-host>:8092`, and restrict TCP port 8092 to a trusted network.

The viewer is read-only and bounded by `web_events_max_events`. Its listener has
no participant authentication or TLS. Use the externally reachable sample mode
only on a trusted development network, or put an authenticated TLS reverse
proxy in front of it.

## File outputs and persistence

Each non-empty final STT result delivered on `voice.transcript` is written to
`transcript.jsonl` before voice gating, including ambient speech rejected by a
configured wake phrase. Transcript-topic delivery is bounded and best effort;
overflow drops the oldest pending transcript, and shutdown discards pending
items. Wake-word gating controls dispatch to the foreground, not storage:
delivered rejected speech is still persisted. The default gate accepts `agent`
and `hey agent`, plays a listening chime, and allows one follow-up utterance for
five seconds. Accepted speech reaches the foreground as a `UserQuery`. Typed
text reaches the foreground but is not an STT transcript.

Each connection writes to a new participant-scoped directory:

```text
artifacts/
├── relay-events.jsonl
└── <participant>-<utc-session-stamp>/
    ├── monitor.jsonl
    ├── instrument-monitoring.jsonl
    ├── transcript.jsonl
    └── foreground.jsonl
```

Every per-session file begins with a `session` record and receives a
`session_end` record on participant departure once that output type has been
written. Monitor records contain a baseline, observations, unavailable-frame
notices, or errors. Foreground records include the query, response, and
model-selected tool names. Relay events contain prompts, responses, participant
metadata, and tool lifecycles; live camera bytes are redacted by the shared
vision tool. When `capture_marker_scans` is enabled, every lab-instrument
invocation saves the exact source JPEG under `marker-scans/`; the worker log
records that path and marker-family counts for debugging. Unmapped marker
payloads are represented by a bounded hash rather than logged verbatim.
Instrument monitoring records contain discrete changes, one-time
tracking-loss events, and complete state snapshots.

## Adapting the sample

### Track another measured object

- Replace `device_map.yaml` with the real identity map.
- Replace the marker assets or identifier implementation.
- Update the focused reading prompt in `LabInstrumentAgent`.
- Extend `normalize_meter_reading` only for domain-specific formats.
- Keep identity, reading, and normalized value as separate fields.

### Add an alert rule

Keep measurement state in `InstrumentMonitorAgent`. Publish a new typed event
when the rule changes state, then subscribe from a separate presentation agent.
This avoids mixing notification cooldowns, speech, or backend failures into the
tracking loop.

### Replace files with production persistence

Subscribe to the same record and instrument topics from a new agent. Maintain
participant correlation from `RuntimeContext.metadata`; do not add participant
IDs to model-visible schemas. Decide whether to block, retry, buffer, or drop
the event after a delivery failure—the runtime does not impose that application
policy.

### Build a different background visual flow

Use `MonitorAgent` as the minimal pattern: participant-bound start, stop, and
status tools, one owned task per participant, current-frame selection, a
dedicated `ImageQueryTool`, typed output events, and cancellation on participant
leave.

## Lifecycle invariants

- A participant join creates fresh output and application state.
- Participant-leave subscribers run concurrently. Each handler must tolerate
  another subscriber cancelling work or releasing image references first.
- A superseding foreground query cancels the previous participant turn.
- Agent tasks fork the Relay context so tool execution remains traceable.
- Model and tool cancellation propagates; it is not converted into a result.
- Only subscribed participants participate in worker readiness.

Preserve these rules when moving logic between agents. Most intermittent leaks
and cross-participant bugs come from changing ownership without moving cleanup.

## Validation strategy

Use three layers:

1. Unit-test deterministic state normalization, change detection, last-seen
   policy, and device-map resolution.
2. Decode every checked-in marker and verify its resolved device name.
3. Evaluate foreground routing separately from the monitor and instrument VLM
   prompt contracts in `eval/visual_eval.py`.

For manual pipeline testing, enable `capture_marker_scans` and inspect the saved
marker-scan image first. It is
the exact source frame used for the marker scan and separates camera and framing
problems from detector or VLM problems. Then inspect Relay events and the
participant JSONL files to follow the tool call, reading, state update, and
notification as distinct stages.

### Routing and visual evals

Start the shared model servers, then run these commands from
`agent-samples/lab-instrument-monitoring/`:

```bash
uv run --project worker python eval/eval.py
uv run --project worker python eval/visual_eval.py
```

`eval/cases.yaml` checks the foreground model's complete first action, exact
tool-call count, and every call's request model. It separates current-view,
recent-history, background-control, ordinary conversation, and general
knowledge requests. `eval/visual_cases.yaml` exercises generated images for
monitor baselines and changes, adversarial visible instructions, multiple
readable devices, competing markers, and exact joint label-to-reading output.

### Printable sample markers

The checked-in marker assets match `yaml/device_map.yaml`:

| Files | Family | Encoded IDs | Device names |
|---|---|---|---|
| `sample-markers/qr/*.png` | QR | `device-1` through `device-5` | `Device1` through `Device5` |
| `sample-markers/aruco/*.png` | ArUco `DICT_4X4_50` | `0` through `4` | `Device1` through `Device5` |

Keep the white border, print without interpolation or cropping, and place a
marker close enough to its instrument display for both to be clear in one
camera frame.

<a id="what-should-become-shared"></a>

## What belongs in shared code

Reuse public SDK blocks directly when they already express the contract:
`VoiceAgent`, `AgentRuntime`, `run_tool_loop`, `CurrentFrameTool`,
`ImageQueryTool`, `MarkerTrackingTool`, and typed topics. Keep device identity,
reading normalization, loss policy, and alert policy in the application until
another concrete application needs exactly the same behavior.
