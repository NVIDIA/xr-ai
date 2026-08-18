<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lab instrument monitoring — building marker-associated visual agents

The lab instrument sample is a reference for applications that combine
foreground questions, opt-in background work, persistent participant state,
visual identification, and event-driven notifications. Start with the
{doc}`quickstart </getting_started/quickstart>` to run it. This page focuses on
how to reuse its structure for another application.

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
accepted speech / typed query          ├─> current frame + generic VLM query
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
                                               └─> state event ──┴─> JSONL/backend
```

`VoiceAgent` publishes participant-scoped queries and lifecycle events into
`AgentRuntime`. Agents call peer tools directly, while typed topics fan results
out to independent consumers. LiveKit remains inside XR-Media-Hub and does not
appear in the worker's agent contracts.

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

This ownership keeps periodic tasks and mutable state out of the runtime. The
runtime delivers typed events; each agent cancels its own participant tasks and
releases its own state.

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
| `yaml/device_map.yaml` | Sample identity data | Map real marker IDs to domain objects |
| `sample-markers/` | Five QR and five ArUco examples | Print or replace with deployment markers |

Configuration and command syntax are also included automatically in the
generated {doc}`configuration <configuration>` and
{doc}`command-line <command-line>` references.

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
5. Create a derived image that marks the detected polygon.
6. Ask the VLM for the nearby display reading and unit.
7. Return typed `InstrumentReading` values.

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
- `InstrumentChange` when its normalized numeric value changes;
- `InstrumentLost` once after the device exceeds the last-seen timeout;
- `InstrumentStateSnapshot` periodically, including tracking status.

A device moving in and out of view does not repeatedly alert unless its value
changes. This is application policy and belongs in the tracker rather than the
VLM prompt or voice agent.

Only marker identities present in `device_map.yaml` are treated as instruments.
Unknown QR payloads and ArUco IDs are logged and ignored, preventing detector
false positives from becoming names such as `ArUco 17`. The one-frame reader
also instructs the VLM to use only a display on the same physical instrument
body as the highlighted marker and to return `UNKNOWN` when an adjacent display
cannot be excluded.

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
typed msgpack/ZMQ service when the backend boundary is a separate process.

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
IDs to model-visible schemas. Decide explicitly whether delivery failure should
block, retry, buffer, or drop—the runtime does not impose that application
policy.

### Build a different background visual flow

Use `MonitorAgent` as the minimal pattern: participant-bound start/stop/status
tools, one owned task per participant, current-frame selection, a dedicated
`ImageQueryTool`, typed output events, and cancellation on participant leave.

## Lifecycle invariants

- A participant join creates fresh output and application state.
- A participant leave cancels foreground and background work before image
  resources are released.
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
3. Evaluate foreground routing separately from visual-reading quality.

For manual pipeline testing, enable `capture_marker_scans` and inspect the saved
marker-scan image first. It is
the exact source frame used for the marker scan and separates camera/framing
problems from detector or VLM problems. Then inspect Relay events and the
participant JSONL files to follow the tool call, reading, state update, and
notification as distinct stages.

## What should become shared

Reuse public SDK blocks directly when they already express the contract:
`VoiceAgent`, `AgentRuntime`, `run_tool_loop`, `CurrentFrameTool`,
`ImageQueryTool`, `MarkerTrackingTool`, and typed topics. Keep device identity,
reading normalization, loss policy, and alert policy in the application until
another concrete application needs exactly the same behavior.
