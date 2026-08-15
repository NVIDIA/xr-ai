<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Background monitoring sample

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
                                           ├> one-shot QR-labelled reads → VLM
                                           ├> InstrumentMonitorAgent controls
                                           ├> Piper TTS
                                           └> foreground.jsonl
MonitorAgent while active ──> current frame → image query ──> monitor.jsonl
InstrumentMonitorAgent ──> QRInstrumentAgent ──> change/lost/state topics
 change/lost topics ──> InstrumentAlertAgent ──> participant voice note
 change/lost/state topics ──> FileOutputAgent ──> instrument-monitoring.jsonl
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
tool. `QRInstrumentAgent` performs reusable one-frame QR-associated reads and
writes source-frame snapshots used to debug QR extraction. The shared QR decoder
scans the whole frame at native resolution and at one enlarged resolution, then
maps every decoded corner back to the source frame.

`InstrumentMonitorAgent` owns all participant-scoped instrument state. It
normalizes numeric readings, retains a known unit when a later VLM result omits
it, and emits an event only when a device is first discovered or its numeric
reading changes. A device can leave and re-enter view without another alert when
its value is unchanged. Once a device has not been seen for the configured loss
timeout, the agent emits one lost-device event. It also emits the full tracked
state every 10 seconds. `InstrumentAlertAgent` converts change and lost-device
topics into voice notes; state snapshots are persisted without being spoken.

Each foreground turn starts with only the system prompt and current request.
Its native tool loop uses the tea-making sample's namespaced route catalog and
four-iteration limit. The model selects from one fixed tool catalog; the worker
does not apply a second lexical router or corrective prompt. It carries no
conversation across requests.
Return-direct monitor controls end the turn immediately.

## Run

The sample uses Nemotron Omni for foreground tool routing and Cosmos for image
inference. It reuses those services and STT from `model-servers` and manages its
own lightweight Piper TTS process.

```bash
cd agent-samples/model-servers
uv sync
uv run model_servers

cd ../background-monitoring-sample
uv sync
uv sync --project worker
uv run background_monitoring_sample
```

Connect a glasses or platform client using the authenticated LiveKit URL,
room, and token printed by the hub. Port 8080 serves the shared connection web client together with the authenticated
token and signaling routes. Monitoring output remains file-only; no monitoring
dashboard is included.

Ask the foreground assistant to start a background task with an optional focus,
for example “watch for packages near the doorway.” It can report status or stop
the task on later turns without capturing the foreground. Current-view
questions and ordinary nonvisual questions work whether monitoring is active
or stopped.

Every non-empty final STT result is written to `transcript.jsonl` before voice
gating, including ambient speech rejected by a configured wake phrase. The
default gate is always on, so every final STT turn also reaches the foreground
as a `UserQuery`. Typed text reaches the foreground but is not an STT transcript.

## File outputs

Each connection writes to a new participant-scoped directory:

```text
artifacts/
├── qr-scans/
│   └── <invocation>-<participant>-<frame>-<sequence>.jpg
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
live camera bytes are redacted by the shared vision tool. Every lab-instrument
invocation saves the exact source JPEG under `qr-scans/`; the worker log records
that path and the decoded QR identifiers for debugging. Instrument monitoring
records contain discrete changes, one-time tracking-loss events, and complete
state snapshots.

The visual and instrument monitoring cadences, 10-second instrument-state
interval, device-loss timeout, history bound, frame timeouts, prompts, VAD
settings, and output path live in `yaml/background_monitoring_worker.yaml`. Model
adapters, endpoints, readiness, and deployment ownership live in
`yaml/models.local.json`.

## Foreground routing eval

The eval checks whether the LLM selects current vision, monitoring history,
background controls, or no tool. It requires `agent-llm` to be running.

```bash
cd agent-samples/background-monitoring-sample
uv run --project worker python eval/eval.py
```
