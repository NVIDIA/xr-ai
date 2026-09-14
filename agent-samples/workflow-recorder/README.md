<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Workflow recorder prototype

This sample records a connected participant's camera on request at a
configurable sampling rate, saves final speech transcripts, and produces
periodic frame-linked visual captions with an Activity → Phase summary. A
session starts with `start recording` and is finalized with `finish recording`.
Refer to the [recording controls](https://nvidia.github.io/xr-ai/latest/reference/migrations.html#workflow-recorder-controls)
for connection, silence, and repeat-recording behavior.

Output is written to `artifacts/sessions/`. Executable SOP guides placed in
`guides/` are validated automatically and listed in
`artifacts/guide-index.json`. Invalid and draft guides are visible in the index
but cannot run.

## Configure

The launcher reads the checked-in configuration automatically. Edit these
files before starting the sample:

| File | Common changes |
|---|---|
| `yaml/workflow_recorder_worker.yaml` | Recording and caption intervals, artifact and guide directories, frame freshness, and VAD settings |
| `yaml/voice_gate.yaml` | Wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | Reused model adapters and endpoint addresses |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |

For example, change `capture_fps` in `yaml/workflow_recorder_worker.yaml` to
sample one camera frame per second:

```yaml
capture_fps: 1.0
```

Restart the sample after an edit. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for the edit workflow and the generated
[workflow recorder configuration](https://nvidia.github.io/xr-ai/latest/reference/configuration.html#config-workflow-recorder)
for the checked-in fields and values.

## Run

Run all commands from `agent-samples/workflow-recorder/`. Start the shared
models first:

```bash
uv run --project ../model-servers model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the recorder from the same terminal:

```bash
uv sync
uv run workflow_recorder
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

Open the authenticated URL printed by DeviceIOHub, allow camera and microphone
access, and connect. Say `start recording` to record and `finish recording` to
finish. The default sampling rate is 2
FPS and the default caption interval is five seconds; both are configured in
`yaml/workflow_recorder_worker.yaml`. This prototype has no retention policy,
so remove old session folders when their frame data is no longer needed.

To generate a guide manually, point Codex or another skill-aware coding agent
at `skills/recording-to-guide/` and a completed session packet. The skill writes
`*.guide.yaml` files into `guides/`, where the running demo discovers and
validates them. Generated guides are always drafts: review one and change
`task.status` to `approved` before use.

## Run an SOP

The voice control plane intentionally uses explicit, deterministic commands so
a question cannot accidentally mutate workflow state:

- `list guides`
- `start guide <guide-id>`
- `guide status`
- `next` or `continue` after the current step is visibly complete
- `skip`, `restart guide`, or `stop guide`

The engine follows the PR 459 execution model: participant-local typed state,
periodic step observations, evidence-gated and revision-checked commits, and no
automatic advancement. Each active run pins the guide's exact version and
SHA-256 content generation; catalog edits only affect future runs.

Guide validation can also be run directly:

```bash
uv run --project worker python -m workflow_recorder_worker._validate_guide guides/my-workflow.guide.yaml
```
