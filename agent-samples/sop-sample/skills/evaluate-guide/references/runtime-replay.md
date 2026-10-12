<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Recorded-camera runtime evaluation

Use a checkout containing SOP replay and its installed `tests` environment.
The guide-generation-only branch does not contain the replay engine. Set
`--replay-root` to the integrated or replay checkout; do not copy engine code
into the skill. Start that checkout's configured local model servers first.
The runner checks local LLM and VLM health and never substitutes another model.

From the repository root, replacing the paths:

```bash
uv run --project /path/to/replay-checkout/tests python agent-samples/sop-sample/skills/evaluate-guide/scripts/replay_capture.py --replay-root /path/to/replay-checkout --packet /path/to/packet.json --guide /path/to/draft.guide.yaml --output /path/to/evaluation/run-001
```

The output directory must not exist. This command writes test artifacts, not
guide YAML. The evaluated YAML is copied verbatim for audit. Approval is enabled
only for the transient in-memory fixture; the original guide remains unchanged.
The manifest pins source frames, guide, model configuration, runtime source, and
prompts. `trace.jsonl` records selected source frames, visual questions and
responses, triggers, commits, simulated user Next commands, and agent output.
`relay.jsonl` records the SDK's tool and model events, including dynamic clock
and commit calls. `result.json` summarizes completion; it is not a quality grade.

Playback uses original frame intervals at 1x. The engine polls independently,
so a slow inference can miss part of a fixed demonstration. The runner does not
hold or replay a completed frame until it passes. At the end it waits five real
seconds by default (`--tail-s`), without refreshing stale camera frames. This
can expose unavailable-frame results. Record any explicit tail override.

The isolated hub, current-frame tool, image-query tool, local models, replay
agent, conversation tool loop, voice aggregation, and clock are real. The
recorded camera and typed Next commands simulate the user. Speech recognition,
audio synthesis, LiveKit, browser behavior, and a person's reaction to guidance
are not tested. Narration is authoring evidence, not injected as replay commands.
No future source image or reference answer enters a runtime request.

Keep an independent evidence inventory when judging the trace. A green engine
completion can still be a perceptual false positive; a stalled guide can be an
occluded result or an open-loop timing mismatch. Record both instead of forcing
all demonstrations to complete. Run revised guides from fresh state and preserve
each version's traces. Do not add source timestamps or evaluation fields to YAML.
