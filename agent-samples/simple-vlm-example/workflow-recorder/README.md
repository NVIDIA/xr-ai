<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Workflow recorder prototype

This sub-project automatically records a connected participant's camera at a
configurable sampling rate, saves final speech transcripts, and produces
periodic frame-linked visual captions with an Activity → Phase summary. A
session starts when a participant connects and is finalized when they leave.

Output is written to `artifacts/sessions/`. Executable SOP guides placed in
`guides/` are validated automatically and listed in
`artifacts/guide-index.json`. Invalid and draft guides are visible in the index
but cannot run.

## Run

From `agent-samples/simple-vlm-example/workflow-recorder/`, start the shared
models if needed:

```bash
uv run --project ../../model-servers model_servers
```

Then start the recorder:

```bash
uv sync
uv run workflow_recorder
```

Open the authenticated URL printed by DeviceIOHub, allow camera and microphone
access, and connect. Recording is automatic. The default sampling rate is 2
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
