---
name: recording-to-guide
description: Convert a workflow-recorder session packet containing synchronized frames, captions, deltas, hierarchy, and voice transcripts into a validated executable SOP guide. Use for recorded XR workflow demonstrations; do not use for ordinary video summaries.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Recording to Guide

Turn one completed recording packet into a draft SOP for the recorder's native
guide engine. The runtime never invokes this skill; a user runs it manually in a
coding agent after recording.

## Input selection

If the user names a session, use it. Otherwise select the newest
`artifacts/sessions/*/packet.json` whose `status` is `complete`. Read
[`references/packet-and-guide-schema.md`](references/packet-and-guide-schema.md)
and [`references/example.guide.yaml`](references/example.guide.yaml) completely
before writing a guide.

Read `packet.json`, `summary.md`, `captions.jsonl`, and `transcript.jsonl` when
present. Use caption `frame_id` links to inspect representative frames when a
caption is ambiguous, two observations conflict, or a visible precondition or
result matters. Do not inspect every 2 FPS frame by default.

## Synthesize the SOP

- Treat frames and captions as visual evidence and transcripts as the
  demonstrator's commentary. Neither source automatically overrides the other.
- Organize stable Activity → Phase groups into actionable steps. Merge repeated
  no-change observations and split phases only when the evidence shows a
  meaningful action, state transition, decision, or verification.
- Preserve useful spoken details that are not visible, but label uncertain
  claims or assumptions. Never manufacture hidden actions, exact values,
  safety guarantees, or successful outcomes.
- Give every step a bounded state write set and an observable completion rule.
- Prefer a `current_view` trigger whose question asks for a short closed-set
  answer. Add `evidence.commit` when a regex match can deterministically prove a
  state value. Use the observation agent only when structured interpretation is
  actually necessary.
- Require two or more consecutive visual matches for completion unless the
  evidence is intrinsically instantaneous and unambiguous.
- Keep `agent.tools` and `voice.tools` to the documented closed tool catalog.
  Never invent a function or put executable code in guide YAML.
- Use user-controlled advancement. The engine deliberately never auto-advances.

## Write and validate output

Write one authoritative `<descriptive-slug>.guide.yaml` beneath the configured
`guides/` directory. Do not overwrite an existing file unless the user asks.
Always emit `task.status: draft`; only a human reviewer may change it to
`approved`. Include the source session ID and increment `task.version` when
revising an existing guide.

Validate from the `workflow-recorder/` directory:

```bash
uv run --project worker python -m workflow_recorder_worker._validate_guide guides/<descriptive-slug>.guide.yaml
```

Fix all validation errors before reporting completion. Report the guide path,
the evidence gaps or uncertainties, and that human approval is required. The
running recorder discovers the file automatically; never edit
`guide-index.json` directly.
