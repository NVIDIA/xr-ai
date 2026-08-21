<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lab instrument monitoring

For an adaptation-oriented architecture guide, see
[`docs/source/reference/lab-instrument-monitoring.md`](../../docs/source/reference/lab-instrument-monitoring.md).

This sample writes durable monitoring output to files and serves a bounded live
event viewer while a separate foreground agent answers voice or typed queries.
See the docs page above for architecture, agent responsibilities, and how to
adapt the sample.

## Run

By default, the sample uses Nemotron Omni for foreground tool routing and Cosmos
for image inference. It reuses those services, STT, and Piper TTS; the sample
never starts or stops model services.

Start the model server stack in one terminal:

```bash
uv sync --project agent-samples/model-servers
uv run --project agent-samples/model-servers model_servers
```

Start Piper TTS in a second terminal:

```bash
uv sync --project services/piper-tts
uv run --project services/piper-tts piper_tts_server
```

Then start the sample in a third terminal:

```bash
uv sync --project agent-samples/lab-instrument-monitoring
uv sync --project agent-samples/lab-instrument-monitoring/worker
uv run --project agent-samples/lab-instrument-monitoring \
  lab_instrument_monitoring

# Allow direct event-viewer access from a trusted private network.
uv run --project agent-samples/lab-instrument-monitoring \
  lab_instrument_monitoring --expose-web-events
```

Connect a glasses or platform client using the authenticated LiveKit URL,
room, and token printed by the hub. Port 8080 serves the shared connection web
client together with the authenticated token and signaling routes. The separate
read-only event viewer defaults to `http://127.0.0.1:8092`. Its listener has no
authentication or TLS. Pass `--expose-web-events` to listen on all IPv4
interfaces, then open `http://<xr-host>:8092` only from a trusted network, or
put an authenticated TLS reverse proxy in front of it.

Ask the foreground assistant to start a background task with an optional focus,
for example “watch for packages near the doorway.” It can report status or stop
the task on later turns without capturing the foreground. Current-view
questions and ordinary nonvisual questions work whether monitoring is active
or stopped.

The default voice gate accepts `agent` and `hey agent`, plays a listening
chime, and allows one follow-up utterance for five seconds.

## File outputs

**Every finalized STT result, including speech rejected by the wake-word
gate, is written to `transcript.jsonl`.** Wake-word rejection controls
whether the foreground responds, not whether the utterance is stored. See the
docs page above for the full persistence and redaction contract.

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

Sample markers to print are in [`sample-markers/`](sample-markers/). The
visual and instrument monitoring cadences, device-loss timeout, history bound,
prompts, VAD settings, device-map path, and event-viewer listener live in
`yaml/lab_instrument_monitoring_worker.yaml`. Marker-to-device mappings live in
`yaml/device_map.yaml`. Model adapters and endpoints live in
`yaml/models.json`.

## Foreground routing eval

The eval checks whether the LLM selects current vision, monitoring history,
background controls, or no tool, and validates every tool call against its
runtime request model. It requires the configured `llm` role — and, for the
visual evals, the relevant VLM — to already be running; the fixed model
configuration provides both through `model-servers`.

```bash
cd agent-samples/lab-instrument-monitoring
uv run --project worker python eval/eval.py
uv run --project worker python eval/visual_eval.py
```
