<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Recorded-camera evaluation during guide generation

This guide-generation package includes both skills, the runtime validator, and
recorded-packet playback. No separate sample regression suite is required.
Start this checkout's configured local LLM and VLM servers; the evaluator never
starts, stops, or substitutes model servers.

From the repository root, using a new output directory:

```bash
PYTHONPATH=agent-samples/procedure-capture-replay/worker uv run --project agent-samples/procedure-capture-replay/worker python agent-samples/procedure-capture-replay/eval/replay.py --packet /path/to/packet.json --guide /path/to/draft.guide.yaml --output /path/to/evaluation/run-001
```

The runner imports the runtime's YAML parser and uses the worker's model,
frame-freshness, frame-timeout, and verification-timeout configuration.
Draft eligibility exists only in its in-memory test fixture; the source stays
unchanged. The authoring LLM observes outputs and revises YAML directly.
This command never generates YAML or approves a guide.

## Playback and controls

Each saved frame is supplied once, in original chronological intervals at 1x.
The current-frame tool selects only frames available at that time. It does not
hold a frame until it passes or refresh the final frame. The deadline is the
recording duration plus five real seconds (`--tail-s` changes this).
A longer timer needs an explicit sufficiently long tail; recorded timestamps
never initialize the runtime timer.

The simulated user says "Start replay", waits for completion, then says "Next"
when required, through the configured conversation model. Terminal
`finish_on_complete` steps end automatically. Checks compare delivered
announcements with detached runtime guidance state and require the final run
to be inactive. The harness does not write progress or call transitions directly.
Narration is authoring evidence, not injected as simulated user commands.

## Artifacts and limits

- `guide.yaml`: exact evaluated YAML, including its original draft status.
- `manifest.json`: source packet hash, source-frame IDs, paths, timestamps and
  hashes, simulated utterances and completion expectations.
- `runtime.json`: hashes of runtime source, prompts and configuration.
- `events.jsonl`: published and selected frames, VLM questions and responses,
  model tool requests, speech, guidance state snapshots, checks and failures.
- `relay.jsonl`: SDK tool and model telemetry, with media sanitized by the SDK.
- `result.json`: execution result and unscored checks, not a physical-quality grade.

Inspect source evidence independently of execution success. Completion can be
a perceptual false positive; a stall can be occlusion, model failure, a missing
prerequisite, or fixed-playback timing mismatch. Report missing negative states
and timer-boundary coverage as untested. Do not weaken requirements or repeat a
saved frame to manufacture a pass. Preserve every revision and its artifacts.

The camera endpoint and typed user inputs are simulated. The guide agent,
execution backend, prompts, model adapters, current-frame and vision tools,
aggregation and clock run their production code. Hub IPC, LiveKit, browser,
microphone, STT, TTS audio and human compliance are excluded. The demonstration
is open-loop and cannot respond to instructions. These limits prevent a claim
of full audiovisual coverage or generalization to other demonstrations.

## Check the generation workflow

Run the integration regressions from the repository root:

```bash
uv run --project tests python -m pytest tests/test_sop_guide_generation.py tests/test_sop_recorded_replay.py -q
```

These tests validate the examples with the runtime parser and invoke the entire
packet-runner CLI with deterministic model doubles. They check real runtime
controls, visual evidence deduplication, elapsed timers, automatic and explicit
completion, draft preservation and artifacts. They test plumbing, not local
model perception. Live-model failures must be reported separately.

`agent-samples/procedure-capture-replay/eval/guide_generation_cases.yaml` holds
seven text-only authoring probes. Give the author only each case's `input` and
the documented authoring prompt, skills and examples. Keep `checks` in an
independent reviewer context. Save package hashes, prompt, raw model/tool trace,
YAML, validation results, action-coverage findings and revision counts.
These probes have no images; they cannot establish visual replay coverage.
Use recorded packets with the command above to evaluate image-based refinement.
