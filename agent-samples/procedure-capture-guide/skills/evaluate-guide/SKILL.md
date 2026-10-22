---
name: evaluate-guide
description: Test a draft SOP against recorded camera input using the XR-AI guide runtime and local models, then report observed failures for guide revision and human review.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Evaluate Guide

Run the fixed draft through XR-AI, observe its outputs, and write
`<slug>.evaluation.md`. Do not edit the guide; the
[authoring skill](../recording-to-guide/SKILL.md) owns revisions. Do not approve
the source guide or modify the recording, runner, runtime, models, or skills to
make a test pass.

## Inputs and roles

Read the guide, original packet, and
[shared contract](../recording-to-guide/references/packet-and-guide-schema.md).
Read the [visual example](../recording-to-guide/references/example.yaml),
[timer example](../recording-to-guide/references/timer.yaml) for waits,
and [blocked example](../recording-to-guide/references/review-blocked.yaml)
for unresolved requirements. These links are relative to this skill.

The authoring LLM, including GPT Luna, is the **observer**, not the runtime VLM,
conversation model, or tool executor. It may inspect source images to establish
expected actions and assess a false positive, but its own answer to a completion
question is never a measured tool result. Interpret narration, captions,
summaries, and original images together as complementary, fallible evidence
of the intended task; none is proof that the runtime observed success.

Use the actual guide checkout, its configured local models, runtime prompts,
tool implementations, evidence thresholds, state transitions, and clock.
If these are unavailable, report runtime evaluation blocked. Do not substitute
Luna, fabricated tool responses, mock commits, or a mental walkthrough.

## Run the recorded-camera test

Refer to [the runner contract](references/runtime-replay.md) for prerequisites,
the exact command, artifacts, and limits. The sample's `eval/guide.py --packet`
mode supplies recorded pixels at the camera endpoint boundary to the real
current-frame tool at original relative timing. It runs the real
guide agent and voice aggregation without a browser or physical microphone.
Its synthetic user says "start guide", then Next after observed completion through the configured
local conversation model; it never directly advances or commits a step.
Real clock tools measure elapsed wall time. Old source timestamps are not timer
starts. The source guide stays draft; test eligibility exists only in memory.

Validate the YAML, then run into a new output directory for each fixed version.
Preserve its manifest, evaluated YAML, events, and Relay events. A changed guide
needs a new run from fresh state. A connection error is a failed test environment,
not evidence that a guide's condition is wrong.

## Review actual behavior

1. Independently map source actions and required objects, counts, orientation,
   placement, order, and durations to the guide's instructions and exact checks.
   Group repeated observations of one action without erasing distinct actions.
   Follow the linked step chain, not YAML listing order. Assess whether the
   transformation faithfully represents the intended task and is understandable
   to someone performing it afresh, not merely whether the recording passes.
   Identify consequential assumptions and questions needing human confirmation;
   ordinary organization and wording do not require individual confirmation.
2. Read the recorded frame selection and actual local VLM responses. Compare
   incomplete, in-progress, completed, and available incorrect states against
   the exact questions. Inspect their source frames when necessary. Do not
   replace a response with what you believe the model should have said.
   Identify false positives, supported completions missed, occlusion, and
   missing negative coverage. Do not weaken requirements to fit a bad demo.
3. Read executed tool requests and results, guidance state snapshots,
   completion notices, and conversation responses. Do not claim internal
   evidence-hit counts or commits that the trace does not expose.
   Distinguish a declared tool from a called tool. Direct `evidence.commit`
   bypasses the observation LLM. Timer starts belong to the runtime on entry
   to the wait step, never to model-written state or a spoken acknowledgment.
4. Check that the engine did not accept incomplete states, inherit a previous
   step's completion, advance without its configured control, or claim more
   than it verified. Check the inactive reset after the final completion,
   including terminal steps with `finish_on_complete: true`. Steps unreachable
   because an earlier step stalled are untested, not passed.
5. For timers, inspect runtime timer results and entry to the wait step.
   Report whether false-before and true-after expiration
   were actually exercised. No image proves elapsed time or continuous contact.
   Missing boundary coverage remains untested; never infer it from video length.
6. Classify failures: guide wording or state wiring, model perception, missing
   evidence, runtime defect, or unavailable infrastructure. Propose guide edits
   only when supported. Rewording cannot repair an unreachable model service or
   add a nonexistent temporal tool. Preserve unsupported checks as review blockers.
   Distinguish missing recording coverage from uncertainty about the intended
   requirement and from a limitation of runtime verification. Missing visual
   evidence does not by itself invalidate a suitable check for a future worker.

The recorded demonstration is open-loop: it cannot adapt to an instruction or
wait for the agent as a person could. If inference or a stalled earlier step
causes the relevant moment to be missed, report timing or coverage limitations,
not automatic proof of a bad completion question. Do not repeat one saved image
to manufacture consecutive evidence or silently trim the recording to success.

## Findings and revision handoff

Report the guide hash and version, runtime and prompt identities, local models,
source IDs and times, exact questions, actual results, committed state, completed
and untested steps, and trace paths. Separate observed execution from your
interpretation. List all simulated inputs and excluded layers (STT, TTS,
browser, hub IPC, and LiveKit); this is not a full audiovisual end-to-end test.

Return actionable findings to the author without changing the guide. After an
authorized revision, run the new YAML again; a previous pass cannot validate an
edit. Count a modification round only when a completed evaluation is followed
by a substantive YAML change. Repeated runs, validation-only formatting edits,
and the last verification without edits are not modification rounds.
Disclose remaining failures and missing tests for human review without treating
every recording coverage gap as a defective requirement. Flag consequential
unresolved requirements and unsupported verification as approval blockers;
the author records their impact and confirmation questions in `.review.md`.
Runtime completion alone is not proof of task fidelity, physical correctness,
or generalization. Keep the guide draft for human approval.
