<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lab instrument monitoring

For an adaptation-oriented architecture guide, see
[`docs/source/reference/lab-instrument-monitoring.md`](../../docs/source/reference/lab-instrument-monitoring.md).

This sample writes monitoring output only to files and keeps one visual monitor
available for every connected participant while a separate foreground agent answers voice or typed queries.
Monitoring stays dormant until the foreground model calls the matching
participant-scoped start tool. The model can also inspect the current frame or
read recent monitor observations. The shared connection web client is served, but there is no monitoring dashboard,
MCP adapter, NAT compatibility layer, or activity-viewer process.

The worker composes peer agents through typed runtime topics:

```text
hub participant join ──────────────────────────────────> file session
final voice STT ─────────> FileOutputAgent ─────────────> transcript.jsonl
accepted STT / typed text ─> ForegroundAgent ─┬> direct answer
                                           ├> current frame → image query
                                           ├> FileOutputAgent history tool
                                           ├> MonitorAgent start/stop/status tools
                                           ├> one-shot marker-labelled reads → VLM
                                           ├> InstrumentMonitorAgent controls
                                           ├> VoiceAggregationAgent → Piper TTS
                                           └> foreground.jsonl
MonitorAgent while active ──> current frame → image query ──> monitor.jsonl
InstrumentMonitorAgent ──> LabInstrumentAgent ──> change/lost/state topics
 change/lost topics ──> InstrumentAlertAgent ──> VoiceAggregationAgent
 change/lost/state topics ──> FileOutputAgent ──> instrument-monitoring.jsonl
 selected typed topics ──> WebEventsAdapterAgent ──> browser event viewer
```

`ParticipantImageAgent` owns the shared image registry, `CurrentFrameTool`, and
participant image cleanup. The monitor and foreground agents own their respective
`ImageQueryTool` instances and acquire frames through that image agent. The monitor
periodically queries a frame only while its participant-scoped task is active.
It also owns idempotent `start_monitoring`, `stop_monitoring`, and
`monitoring_status` tools. The foreground agent exposes its own composed
current-view tool and calls the monitor's control tools directly, binding the
participant before every operation.

`FileOutputAgent` owns structured durable outputs and the bounded recent-history
tool. `LabInstrumentAgent` performs reusable one-frame marker-associated reads.
It can optionally write source-frame snapshots used to debug marker extraction.
The shared
image agent uses `MarkerTrackingTool` for QR and ArUco markers. `device_map.yaml`
maps each marker family and raw ID to the device name used in readings, state,
logs, and voice alerts. Ready-to-print PNGs for every configured device and a
mapping table are available in [`sample-markers/`](sample-markers/).
Detections absent from the device map are logged and ignored; they never become
invented device names or voice alerts. For each mapped marker, the VLM is told
to accept only a display on the same physical instrument body and to return
`UNKNOWN` when an adjacent display cannot be ruled out.

`InstrumentMonitorAgent` owns all participant-scoped instrument state. It
normalizes numeric readings, retains a known unit when a later VLM result omits
it, and emits an event only when a device is first discovered or its numeric
reading changes. A device can leave and re-enter view without another alert when
its value is unchanged. Once a device has not been seen for the configured loss
timeout, the agent emits one lost-device event. It also emits the full tracked
state every 10 seconds. `InstrumentAlertAgent` converts change and lost-device
topics into voice notes; state snapshots are persisted without being spoken.
Foreground replies and alert notes first pass through `VoiceAggregationAgent`,
which keeps one participant response active and combines non-urgent updates
that arrive while it is being spoken. Interrupting safety messages retain the
existing immediate interruption behavior.

Each foreground turn starts with only the system prompt and current request.
Its native `run_tool_loop` integration uses a namespaced route catalog and
four-iteration limit. The model selects from one fixed tool catalog; the worker
does not apply a second lexical router or corrective prompt. It carries no
conversation across requests.
Return-direct monitor controls end the turn immediately.
When the model selects `current_view`, the sample follows the
`simple-vlm-example` path: it acquires one frame and publishes
`StreamingImageQueryTool` chunks directly to voice. The visual answer is not
sent back through the foreground LLM for rewriting. A dedicated current-view
prompt applies visible-evidence, plain-speech, and response-length rules on this
direct path.

## Run

The sample uses Nemotron Omni for foreground tool routing and Cosmos for image
inference. It reuses those services and STT from `model-servers` and manages its
own lightweight Piper TTS process.

```bash
cd agent-samples/model-servers
uv sync
uv run model_servers

cd ../lab-instrument-monitoring
uv sync
uv sync --project worker
uv run lab_instrument_monitoring
```

Connect a glasses or platform client using the authenticated LiveKit URL,
room, and token printed by the hub. Port 8080 serves the shared connection web
client together with the authenticated token and signaling routes. The separate
read-only event viewer runs at <http://127.0.0.1:8092> and groups live output by
participant and topic. JSONL files remain the durable output.

The event viewer intentionally listens on loopback and does not reuse media-hub
authentication. Use an SSH tunnel for remote development; do not expose it on
an untrusted network without an authenticated TLS reverse proxy.

Ask the foreground assistant to start a background task with an optional focus,
for example “watch for packages near the doorway.” It can report status or stop
the task on later turns without capturing the foreground. Current-view
questions and ordinary nonvisual questions work whether monitoring is active
or stopped.

Every non-empty final STT result is written to `transcript.jsonl` before voice
gating, including ambient speech rejected by a configured wake phrase. The
default gate accepts `agent` and `hey agent`, plays a listening chime, and
allows one follow-up utterance for five seconds. Accepted speech reaches the
foreground as a `UserQuery`. Typed text reaches the foreground but is not an
STT transcript. Spoken agent text is also published to the client
`agent.response` topic for captions and other accessibility presentation.

## File outputs

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
metadata, and tool lifecycles;
live camera bytes are redacted by the shared vision tool. When
`capture_marker_scans` is enabled, every lab-instrument invocation saves the
exact source JPEG under `marker-scans/`; the worker log records that path and
marker-family counts for debugging. Unmapped marker payloads are represented by
a bounded hash rather than logged verbatim. Instrument monitoring
records contain discrete changes, one-time tracking-loss events, and complete
state snapshots.

The visual and instrument monitoring cadences, 10-second instrument-state
interval, device-loss timeout, history bound, frame timeouts, prompts, VAD
settings, device-map path, output path, and event-viewer listener live in
`yaml/lab_instrument_monitoring_worker.yaml`. Marker-to-device mappings live in
`yaml/device_map.yaml`. Model
adapters, endpoints, readiness, and deployment ownership live in
`yaml/models.local.json`.

## Foreground routing eval

The eval checks whether the LLM selects current vision, monitoring history,
background controls, or no tool, and validates every tool call against its
runtime request model. It requires the configured `llm` role to be running; the
default profile provides it through `model-servers`.

```bash
cd agent-samples/lab-instrument-monitoring
uv run --project worker python eval/eval.py
uv run --project worker python eval/visual_eval.py
```
