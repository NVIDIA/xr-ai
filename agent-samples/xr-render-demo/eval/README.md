<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo eval harness

End-to-end test of the agent LLM's tool-calling against the live model. Each
case feeds a synthetic scene and head pose into the model with the same system
prompt and native tool schemas as the live worker, executes tool
effects against deterministic fixtures, then checks the resulting scene
mutations against a per-case expectation.

| Tier | Command | Runs against | Cost |
|---|---|---|---|
| Supervisor routing | `xr_render_demo_eval_supervisor` | faked subagents that record delegations | ~15 s/case |
| Subagent components | `xr_render_demo_eval_subagents` | one real agent over faked leaf functions | ~30 s/case |
| End-to-end corpus + basics | `xr_render_demo_eval` | supervisor + agents over faked services | ~30 min full |
| Live | `xr_render_demo_live_{smoke,pose_matrix,manip,garble,explore}` | the running demo stack | minutes |

All commands run from the eval project:

```bash
cd agent-samples/xr-render-demo/eval && uv sync   # once

# Full corpus + precision cases + utterances battery
uv run xr_render_demo_eval

# The utterances battery alone: the most common utterances, their perturbation
# classes, and history-bearing variants. Run after EVERY prompt or ops
# change; full-suite variance hides single-case damage.
uv run xr_render_demo_eval utterances

# Subset by case name (space-separated; unknown names error out)
uv run xr_render_demo_eval move_left_one_meter between_two_spheres

# Routing and component tiers, optionally filtered by agent or case name
uv run xr_render_demo_eval_supervisor
uv run xr_render_demo_eval_subagents placement
```

The offline tiers need only the agent LLM (default `http://localhost:8108`);
they do not require the demo stack, capability services, or LOVR.

## Live drivers

Live drivers join the running stack (`uv run python main.py` from the sample
directory) as synthetic participants, inject typed text, set a simulated
head pose, and score real scene state. They require `allow_sim_pose: true`
in `yaml/openxr_service.yaml` (off by default; flip it for eval runs and
restart the stack). Isolation rules:

- Fresh participant id per case: transcript history otherwise bleeds between
  cases and collapses supervisor behavior.
- Clear the scene between cases through the scene RPC: leftovers make
  referents ambiguous and invite anchoring on stale objects.
- Vary prompt phrasing across cases: repeating one sentence builds a
  self-history no real user produces.
- Never filter a run's output in the run command; write the full log to a
  file and filter the file.
- Repeat runs (3x) before believing any single-run delta; near-tie decisions
  flip run to run even at temperature 0.

`xr_render_demo_live_garble` covers speech-to-text noise (homophones,
truncations, corrections, stutters) with restraint scoring: wrong mutations
fail, clarifying replies pass. `xr_render_demo_live_explore` sends novel
conversational phrasings scored by intent invariants; promote any violation
into a permanent tier case, then fix.

## Prompt-tuning law

The current agent model follows templates and contrast pairs; it ignores
prohibitions. Fix behavior with worked examples, and pair every
refuse-example with a proceed-example so it does not contaminate neighboring
behaviors.

When even worked examples fail (the model keeps resolving what it should
copy) move the resolution into code and rename the tool parameter so the
schema asks for exactly what the model does reliably. Anchor descriptors are
the precedent: renaming the parameter to `anchor_words` with a copy-verbatim
description fixed in one step what five prompt variants could not, with
`spatial_ops` resolving shape synonyms, mangled nouns, and color words
against the scene deterministically.

## Prompt-tuning loop

When iterating on `supervisor_prompt.txt` or any subagent prompt, run the
fast gate manually after each edit:

```bash
uv run xr_render_demo_eval utterances
```

25 cases, ~3 minutes. The `scenarios` and `precision` tiers catch
regressions but take longer; run them before calling a tuning round done.

## Writing a case

The end-to-end corpus lives in `xr_render_demo_eval/cases.py` (dict-shaped;
pose override, multi-turn `history`, and undo `recent_moves` are all
exemplified). Precision and utterances cases are `Case` dataclasses in
`xr_render_demo_eval/harness.py`; routing and component cases live in
`supervisor.py` and `subagents.py`. Copy the closest existing case and edit.

## Don't train on the test set

Prompt worked examples and case fixtures share the same model, so the
harness audits every worker prompt at startup (all three offline tiers run
it) and warns on:

1. A case utterance from any tier appearing verbatim in a prompt.
2. A case fixture id appearing in a prompt.
3. A quoted prompt example pairing an eval-vocabulary color
   (red/green/blue/yellow/cyan/orange/purple/white/black) with an
   eval-vocabulary shape (sphere/cube/box/ball).

Fix overlaps by changing the prompt, not the case; use colors and shapes
outside the eval vocabulary (teal / lavender / magenta / turquoise, cone /
cylinder / capsule / torus) in worked examples. A case that only passes
while the prompt contains its vocabulary is scoring recall, not skill.

## What the harness does not cover

- The live worker pipeline (VAD, STT, TTS, history bookkeeping); the live
  tier covers it.
- Real scene-service / LOVR effects (fixture-succeeded).
- Real visual queries (`look_at_current_frame`, `look_at_past_frame`): stubbed.
