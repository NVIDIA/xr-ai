---
name: implement-flow
description: Implement or extend a sample-local procedure flow compatible with the procedure-capture-guide step runner, including lifecycle, evidence, configuration, and regression tests. Use for execution capability development, not for authoring a guide with existing capabilities.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Implement a procedure flow

Develop a flow that the existing step runner can execute without adding a
second polling loop or bypassing the guide's state guards. Work in the requested
xr-ai checkout. Paths below are relative to
`agent-samples/procedure-capture-guide/` unless stated otherwise.

## Inspect the actual contract

Read `worker/procedure_capture_guide_worker/guide/flow.py` and `step_runner.py`
first. Read `timer_flow.py` for deterministic observations or
`visual_verification_flow.py` for repeated evidence. Inspect `_prepare_runner`
and the result application paths in `guiding.py` before integrating a new flow.
These files, not a copied template, define the current API.

A flow implements:

```python
async def poll(self) -> FlowUpdate[Progress]:
    ...
```

`Progress` is a concrete dataclass or validated model for this capability. Each
call performs one bounded observation and returns a `FlowUpdate`:

- `pending`: evidence is available but the condition is not yet satisfied.
- `ready`: the flow's evidence requirement is satisfied.
- `unavailable`: this observation cannot establish a judgment. Distinguish this
  from a valid negative or unknown judgment in the typed progress value.
- `delay_s`: a finite positive delay before another observation. A timer must
  bound it by the remaining duration so a long interval cannot delay expiry.

The backend's scoped `apply` callback returns whether the **guide step** is
complete. Flow readiness alone does not advance a guide or authorize speech.
The runner serializes polls, owns the background task, and schedules updates.
An unhandled error ends background execution as `failed`; do not rely on an
implicit retry or swallow errors into positive results.

## Design the evidence and lifetime

1. Specify the input, progress type, ready condition, unavailable conditions,
   and cancellation behavior before implementing the flow. Decide whether
   evidence is latched, consecutive, windowed, or must remain continuously true.
   A one-time positive followed by a timer does not prove continuous truth.
2. Inject typed camera/tool/model operations. Use existing `xr_ai_tools` tools
   and typed model services; do not add ad-hoc model HTTP or imports from the
   tea sample. Keep model policy separate from scheduling. Visual changes must
   preserve both direct-image and caption-based verification paths as applicable.
3. Give each step execution a fresh flow instance. Runtime code owns timer
   starts and immutable durations. Source identity, including participant,
   track, sequence and timestamp, belongs to the observation; repeated inference
   on one cached frame must not count as additional evidence.
4. Keep evidence windows and frame deduplication inside the flow. Keep guide
   state, step order, controls and speech outside it. Await every operation;
   use a task group if genuine parallel operations are necessary. Propagate
   `CancelledError` after cleanup and clear interrupted confirmation state.
   Never create detached tasks or shield inference from the runner's stop.
5. Bound external operations. Handle expected service unavailability as an
   explicit unavailable update with retained diagnostics and retry cadence.
   Let invalid configuration and unexpected implementation errors propagate.
   Preserve successful intermediate evidence when a later stage fails.

## Integrate without bypassing execution

Register construction and typed result application in `guiding.py`. The active
runner must be the sole scheduler. Before committing results, check that the
participant, session, step and state revision still match the observation;
check under the backend's lock after asynchronous work. Foreground tools read
runtime state instead of creating or resetting execution.

Do not hold a session lock during model inference. Transitions cancel and drain
the previous runner before starting the next; flow cleanup must not require
that same session lock, or stopping under the lock will deadlock. Retain
invalidation of queued speech at transitions. Completion, skip, cancellation,
unavailable evidence and execution failure must remain distinguishable.

If a guide needs to select the new capability, extend `_workflow_spec.py`'s
validated contract and its explicit allowlists. Reject malformed or unsupported
configuration at load time. Do not make an arbitrary YAML string import Python
or execute a tool. The current schema permits one flow per step; introducing
multi-flow composition is a separate schema change, not a hidden convention.

For new guide-visible fields, update the documentation, orchestration skill,
`guide-generation-skills/draft-guide/` instructions and examples, and
`guide-generation-skills/evaluate-guide/` instructions together. The authoring
model still writes YAML directly. Keep implementation guidance out of generated
user instructions. Update packet evaluation and trace attribution so the same
flow and verifier execute in evaluation and live guidance.

## Prove compatibility

From the repository root, use the repository's tests environment to run the
new tests plus `tests/test_procedure_step_runner.py`, `tests/test_sop_guide.py`,
and `tests/test_procedure_guidance_policies.py`. Use controlled operations and
clocks for deterministic lifecycle checks. Test observable behavior:

- Pending, ready, unavailable, and failed execution; no completion from failure.
- Stop or skip during an operation drains it and rejects late progress.
- Restart and participant isolation create independent evidence/timer state.
- Repeated source frames do not satisfy multiple confirmations; missing,
  negative, unknown and expired evidence follow the declared policy.
- A foreground read uses the active flow's state without restarting it.
- The real guide backend accepts results only in their original execution
  scope and emits completion once; capture behavior remains unchanged.

Use the live-model evaluator when changing judgment prompts or model policy;
CPU tests alone do not establish perceptual accuracy. Preserve externally
managed servers and local guides. Report which checks ran and any evaluation
limitations. Do not create or publish a PR unless requested.
