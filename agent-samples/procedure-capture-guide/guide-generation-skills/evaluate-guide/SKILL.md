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
`evaluation.md` beside its `guide.yaml`. Do not edit the guide; the
[orchestration skill](../SKILL.md) owns revisions. Do not approve
the source guide or modify the recording, runner, runtime, models, or skills to
make a test pass.

## Inputs and roles

Read the guide, original packet, and
[shared contract](../draft-guide/references/packet-and-guide-schema.md).
Read the [visual example](../draft-guide/references/example.yaml),
[timer example](../draft-guide/references/timer.yaml) for waits,
and [blocked example](../draft-guide/references/review-blocked.yaml)
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

Ensure the target checkout's configured local LLM and VLM servers are running;
the evaluator never starts, stops, or substitutes model servers. Validate the
YAML, then run the target checkout's
`agent-samples/procedure-capture-guide/evaluation/guide/evaluate_recorded_packet.py`
from the repository root, replacing the packet and guide paths and choosing a
new output directory:

```bash
PYTHONPATH=agent-samples/procedure-capture-guide/worker \
uv run --project agent-samples/procedure-capture-guide/worker python \
  agent-samples/procedure-capture-guide/evaluation/guide/evaluate_recorded_packet.py \
  --packet /path/to/packet.json \
  --guide "agent-samples/procedure-capture-guide/<slug>-guide/guide.yaml" \
  --output "agent-samples/procedure-capture-guide/<slug>-guide/evaluation/run-001"
```

Use the script from the target checkout even if the skills were installed
elsewhere: it locates that checkout's sample configuration from its own path.
The runner imports the runtime's YAML parser and uses the worker's model,
frame-freshness, frame-timeout, and verification-timeout configuration. It
uses the sample's shared `evaluation/guide/_runtime_harness.py` for camera and speech boundaries,
agent setup, and traces. Packet playback and checks also belong to the sample's
`evaluation/guide/` implementation; this skill supplies instructions, not runtime
code. The evaluator needs no tea-example files or regression fixtures from `tests/`.
The source guide stays draft; test eligibility exists only in memory.
The command never generates YAML or approves a guide.

### Playback and controls

Each saved frame is supplied once at the camera endpoint boundary, in original
chronological intervals at 1x. The current-frame tool selects only frames
available at that time. The runner does not hold a frame until it passes or
refresh the final frame. Its deadline is the recording duration plus five real
seconds; `--tail-s` changes the extra time.

The simulated user says "Start guide", waits for observed completion, then
says "Next" when required through the configured local conversation model.
Terminal `finish_on_complete` steps end automatically. Checks compare delivered
announcements with detached runtime guidance state and require the final run
to be inactive. The harness never writes progress or calls transitions directly.
Narration is authoring evidence, not injected as simulated user commands.

Real clock tools measure elapsed wall time. The timer starts when "Next" enters
its wait step, not when playback starts; recorded timestamps never initialize
it. Allow `--tail-s` for the full `duration_s` plus preceding-step verification
and transition latency. For multiple waits, allow for their combined durations
and transitions.

### Saved evidence and scope

Preserve each version's output directory:

- `guide.yaml`: exact evaluated YAML, including its original draft status.
- `manifest.json`: source packet hash, source-frame IDs, paths, timestamps and
  hashes, simulated utterances and completion expectations.
- `runtime.json`: hashes of the sample worker package, prompts, configuration,
  evaluator sources, and loaded XR-AI SDK modules, with paths relative to the
  sample. It does not scan the worker's virtual environment or unrelated dependencies.
- `events.jsonl`: published and selected frames, VLM questions and responses,
  model tool requests, speech, guidance state snapshots, checks and failures.
- `relay.jsonl`: SDK tool and model telemetry, with media sanitized by the SDK.
- `result.json`: execution result and unscored checks, not a physical-quality grade.

The camera endpoint and typed user inputs are simulated. The guide agent,
execution backend, prompts, model adapters, current-frame and vision tools,
aggregation and clock run their production code. Hub IPC, LiveKit, browser,
microphone, STT, TTS audio and human compliance are excluded; this is not a
full audiovisual end-to-end test.

A changed guide needs a new run from fresh state. A connection error is a failed
test environment, not evidence that a guide's condition is wrong.

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

Return actionable findings to the orchestrator without changing the guide. After an
authorized revision, run the new YAML again; a previous pass cannot validate an
edit. Count a modification round only when a completed evaluation is followed
by a substantive YAML change. Repeated runs, validation-only formatting edits,
and the last verification without edits are not modification rounds.
Disclose remaining failures and missing tests for human review without treating
every recording coverage gap as a defective requirement. Flag consequential
unresolved requirements and unsupported verification as approval blockers;
the author records their impact and confirmation questions in `review.md`.
Runtime completion alone is not proof of task fidelity, physical correctness,
or generalization. Keep the guide draft for human approval.

## Check changes to the generation workflow

When testing changes to the skills or runner, run the integration regressions
from the repository root. These are not prerequisites for evaluating an
individual guide:

```bash
uv run --project tests python -m pytest tests/test_sop_guide_generation.py tests/test_sop_recorded_replay.py -q
```

These tests validate the examples with the runtime parser and invoke the entire
packet-runner CLI with deterministic model doubles. They check real runtime
controls, visual evidence deduplication, elapsed timers, automatic and explicit
completion, draft preservation and artifacts. They test plumbing, not local
model perception. Live-model failures must be reported separately.

`agent-samples/procedure-capture-guide/evaluation/guide/authoring_cases.yaml` holds seven text-only authoring
probes. Give the author only each case's `input` and the documented authoring
prompt, skills and examples. Keep `checks` in an independent reviewer context.
Save package hashes, prompt, raw model/tool trace, YAML, validation results,
action-coverage findings and revision counts. These probes have no images;
they cannot establish visual replay coverage. Use recorded packets with the
command above to evaluate image-based refinement.
