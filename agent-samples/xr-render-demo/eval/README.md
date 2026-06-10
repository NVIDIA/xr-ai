<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo eval harness

End-to-end test of the agent LLM's tool-calling against the live model
and MCP servers. Each case feeds a synthetic scene + head pose into the
model with the same system prompt the live worker uses, runs a
multi-step rollout (executing safe oxr-mcp tools between turns;
render-mcp tools are fake-succeeded so the live LOVR scene is not
mutated), then checks the final scene mutations against a per-case
expectation.

## Prerequisites

The shared model servers and a render-demo stack must be running:

```bash
# weights resident in the background — start once, leave alone
uv run --project ~/hub/xr-ai/agent-samples/model-servers model_servers

# stack
uv run --project ~/hub/xr-ai/agent-samples/xr-render-demo xr_render_demo
```

The harness needs these reachable on default ports:
- agent LLM: `http://localhost:8107` (nemotron3_nano)
- oxr-mcp:   `http://localhost:8230`
- render-mcp: `http://localhost:8220` — must be reachable so the
  harness can discover its tool schemas (`add/update/remove_primitive`).
  Its mutating tools are then fake-succeeded so the live LOVR scene
  isn't actually mutated.

`vlm-mcp` / `video-mcp` may be down — they're only consulted for
tool-schema discovery and fail open if absent.

## Run

```bash
# All built-in cases against the current system.txt
agent-samples/xr-render-demo/eval/eval.py

# Subset by case name — fast iteration on a single failing cluster.
# Comma-separated; unknown names error out (mutually exclusive with the
# positional query arg below).
agent-samples/xr-render-demo/eval/eval.py --only move_left_one_meter,between_two_spheres

# Watcher-friendly equivalent: write case names (newline- or
# comma-separated; '#' comments OK) to eval/.only. Gitignored.
# Active subset is echoed at startup.

# One ad-hoc query (prints the raw LLM response)
agent-samples/xr-render-demo/eval/eval.py "Move the cube up 30 cm"

# Score a prompt file other than the live worker's system.txt — e.g.
# main's version, a draft, or a checkout from another branch.
agent-samples/xr-render-demo/eval/eval.py --prompt /tmp/alt-system.txt

# Score against a hosted model (e.g. nvidia/nemotron-3-super-120b-a12b at
# build.nvidia.com) instead of the local vLLM on 8107.  Set NVIDIA_API_KEY
# in the env first (or pass --agent-api-key).
export NVIDIA_API_KEY=nvapi-...
agent-samples/xr-render-demo/eval/eval.py \
  --agent-llm   https://integrate.api.nvidia.com/v1/chat/completions \
  --agent-model nvidia/nemotron-3-super-120b-a12b
```

The script is a self-contained `uv run --script` — no `uv sync` needed.

## Watcher

`eval_watch.sh` polls `system.txt`'s sha1 once per second (hash, not
mtime — editors and language servers re-save the file without
changing bytes when you switch focus). The user starts it **once**;
from then on it re-runs the eval on its own whenever the prompt's
**bytes** change. Any content change aborts the running eval and
starts a new one once the file has been quiet for 10 seconds; results
land in a gitignored log in the eval dir (`.eval_loop.log`).

```bash
agent-samples/xr-render-demo/eval/eval_watch.sh                     # user starts once
tail -f agent-samples/xr-render-demo/eval/.eval_loop.log            # read scores

agent-samples/xr-render-demo/eval/eval_watch.sh /path/to/alt.txt   # different prompt
```

The lock and log are repo-local (`.eval_watch.lock`, `.eval_loop.log` in
the eval dir, both gitignored), so watchers in different checkouts don't
share global state. The single-instance guard is an exclusive `flock` on
the lock file, released automatically when the watcher exits.

Iterating on the prompt as a coding agent (edit → wait → read the log;
never re-launch the eval, never `touch` to force a run) is documented in
[`../AGENTS.md`](../AGENTS.md).

Only one watcher runs at a time. A second invocation refuses to
start, exits non-zero, and prints the existing PID along with the
two ways to handle it (`tail` the log of the running watcher, or
`kill <pid>` and rerun). The script never kills processes it didn't
spawn — that decision stays with the caller, which keeps the behavior
predictable across users / sandboxes / CI runners.

`eval_watch.sh` is Linux-only. The single-instance guard reads
`/proc/<pid>/cmdline` to confirm the stored PID is the watcher (not
some unrelated process that recycled the same PID); macOS has no
`/proc`, so the script will not run there.

Score history at a glance:

```bash
grep "passed$" agent-samples/xr-render-demo/eval/.eval_loop.log | tail
```

## Robustness sweep (`--robustness`)

Off by default; with the sweep off the perturbations are never injected and a
run scores like the default suite. Turn it on to probe whether a case's result
is *robust* or just balanced on a decision boundary: each selected case is
rolled out under the clean context plus a few **semantically-irrelevant context
perturbations** a correct prompt must be invariant to (a different
reference-time value, the worker's context-block ordering, and the `--thinking`
scaffold). A case is reported `ROBUST` only when it PASSes **every** variant; a
single-variant pass is noise.

```bash
# one case across all variants
agent-samples/xr-render-demo/eval/eval.py --robustness --only stack_three_cubes
```

The sweep multiplies backend calls per case, so it **requires a case subset**:
running `--robustness` (or the directive below) with no `--only`/case names
refuses to sweep the full suite ×4 and exits non-zero.

Watcher-friendly toggle: add a bare `ROBUSTNESS` line to `.only` alongside the
case names, and the watcher reports robustness verdicts for that subset on
every prompt edit. Drop the `ROBUSTNESS` line to return to normal scoring:

```
ROBUSTNESS
stack_three_cubes
put_sphere_in_cube_is_containment
```

## Writing a case

Read `eval.py`'s `CASES` list — every shape (single-turn, pose
override, multi-turn `history`, undo `recent_moves`) is exemplified
there. Copy the closest existing case and edit. The case dict is
what the harness consumes directly; there's no case schema layer.

## Colour-fidelity cases

Cases tagged `"category": "color"` are calibrated to catch the two
colour failures that actually matter in this demo and to ignore the
noise in between:

1. **Gross colour confusion** — building a clearly different colour
   family than asked (the canonical "white when asked for magenta").
2. **Built-vs-said dishonesty** — narrating one colour while painting
   another.

What they deliberately do **not** police is slight, in-family hue
drift: a red leaning crimson, a cyan leaning teal, an orange-ish
salmon. Those shifts are noise, not bugs. This is a small, focused
suite over the basic palette the demo is observed
to confuse: magenta, white, red, green, blue, yellow, black, orange,
purple, cyan, gray, plus a recolour-to-magenta lie catcher. The
magenta-family gross-confusion and honesty guard is carried by the
white↔magenta cases (`color_make_magenta_sphere`,
`color_make_white_sphere`, `color_recolor_to_magenta`) — slight
in-family shifts like pink↔magenta are noise and are not policed.

Two checks combine:

- the created object's `r`/`g`/`b` must land in the colour's box. The
  boxes are **intentionally wide** — each comfortably admits any
  reasonable rendering of the requested colour and fails only a clearly
  wrong one (e.g. magenta = high R, *low* G, high B, so it excludes
  white/red/blue; white = all channels high, so it excludes any
  saturated hue). They are NOT tight hue boxes; a case fails only on
  gross family confusion, not a slight shift.
- `reply_colors` — the honesty allow-set. Two sub-checks run against it:
  the spoken reply may not name a colour outside the set (so "said
  magenta" for a white build fails), and **build-vs-said honesty** — the
  reply is also checked against the colour that was *actually built*.
  `_built_rgb()` extracts the as-built `r`/`g`/`b` from the latest
  colour-bearing add/update (resolving omitted channels against the
  scene — which is what makes `color_recolor_to_magenta` catch a no-op
  update that leaves the object white), and `_built_color_names()` maps
  that RGB to the names that honestly describe it (expanded via synonyms,
  within `_BUILD_NEAR`). `_reply_color_ok()` fails any spoken colour
  inconsistent with the built RGB — catching "built white, said magenta"
  even when the said colour was the one the user requested. A reply that
  names no colour passes.

Because these cases deliberately sweep the palette, the reserved-vocab
audit (check #4 below) ignores their colour *words* — those cannot
constrain the prompt's worked examples. Only the colour vocabulary is
exempt: a `category="color"` case's shapes, coordinates, and utterances
are still audited, and the restriction binds spatial cases in full. Shapes
stay covered because the spatial cases use the same
`sphere`/`cube`/`pyramid` vocabulary.

## Visual-query routing cases

Cases tagged `"category": "vision"` prove the agent routes a REAL-WORLD
question ("what am I holding?", "what colour is my mug?") to the camera
instead of answering from a VIRTUAL scene primitive — the headline bug
the `REAL-WORLD VISUAL QUERIES` prompt section fixes. A `"first_call"`
key switches `_check` into vision-routing mode:

- `first_call` — the allow-set for the rollout's FIRST tool call
  (`["look_at_current_frame"]`). A real-world query that opens
  with `add_primitive`/`update_primitive` (mutating the scene) or with
  no tool call at all (answering from scene state) fails.
- `vision` — a canned `look_at_current_frame` answer naming a KNOWN dummy
  colour. The harness mocks the brain-executed perception tool in
  `_exec_tool`: `look_at_current_frame` returns `{"answer": "<vision>"}`
  on success, or `{"error": ..., "spoken": ...}` on a perception failure.
  The rollout feeds that back to the model (same multi-step loop the live
  worker runs), so the case can then assert the model CONSUMED the
  camera result:
- `reply_names` — colour word(s) the FINAL reply must contain (the
  camera/real colour B).
- `reply_forbids` — colour word(s) the reply must NOT contain. The
  distractor cases (`vision_real_mug_vs_virtual_sphere`,
  `vision_real_shirt_vs_virtual_cone`) put a virtual primitive of
  colour A in the scene and a different camera colour B in `vision`,
  then assert the reply names B and never A — the proof the model
  ignores scene state and looks at the camera.

The dummy colours are supplied at runtime by the mock, never written
into the prompt, so they cannot be memorised (no train-on-test).

### Multi-turn re-query

A case with a `"turns"` list (instead of a single `user`) runs each turn
as its own rollout, accumulating prior turns' spoken replies into the
`[Recent conversation]` block — exactly how the worker rebuilds messages
per turn (turn N sees turn N-1's text reply, never its raw tool calls).
Each turn carries its own `first_call` / `reply_names` / `reply_forbids`
and is checked independently; the case passes only if every turn does.

`vision` may be a SEQUENCE of strings here: successive `look_at_current_frame`
calls consume the next entry (the last repeats), so the camera can "see" a
different colour on a re-query. `vision_requery_returns_fresh_colour`
uses this to prove the agent re-fetches: turn 1 the camera returns B1 and
the reply names B1; turn 2 ("and what about now?") the camera returns B2,
and the reply must route to the camera AGAIN and name B2 — never the
stale B1 from the conversation history. This guards the re-query rule in
`system.txt`'s `REAL-WORLD VISUAL QUERIES` section (every new user message
gets its own fresh look).

## Don't train on the test set

Prompt worked-examples and case fixtures share the same model. The
harness audits at startup for four kinds of overlap and prints a
warning for any it finds:

1. Verbatim user utterance from a case appearing in `system.txt`.
2. Concrete scene coordinates (formatted like `(0.50, 1.60, -1.50)`)
   from a case appearing in `system.txt`.
3. `recent_moves` coordinates from a case appearing in `system.txt`.
4. **Reserved prompt vocabulary** — any colour or shape word from the
   eval-case vocabulary (`_EVAL_VOCAB_COLORS` / `_EVAL_VOCAB_SHAPES`
   in `eval.py`) appearing inside a worked-example section of
   `system.txt`. Worked-example sections are triple-backtick blocks
   and any block starting with `WORKED EXAMPLE`, `Example:`,
   `iter N:`, or `tool_call N:`; the first blank line after the
   marker ends the block. Rule narration outside those blocks may
   still mention the eval vocabulary generically (e.g. the colour
   table, anchor-routing rules) — the restriction is only on the
   worked examples, which are the strings the model is most likely
   to memorise as a template. For cases tagged `"category": "color"`
   the colour *words* are dropped before this check (see "Colour-fidelity
   cases" above); their shapes are still audited.

Fix overlaps by changing the prompt example, not the case. For
check #4, use colours and shapes outside the eval vocabulary
(turquoise / teal / lavender / magenta / cone / cylinder / capsule)
when reaching for a fixture word in a worked example.

## What the harness does not cover

- The live worker pipeline (VAD, STT, TTS, history bookkeeping).
- Real render-mcp / LOVR effects (fake-succeeded).
- Real VLM perception. `look_at_current_frame` is mocked in
  `_exec_tool` (a canned `vision` answer), so the
  `category="vision"` cases check tool-call *routing* and that the reply
  consumes the camera result — not real image understanding.
