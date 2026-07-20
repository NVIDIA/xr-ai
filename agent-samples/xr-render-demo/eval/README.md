<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo eval harness

End-to-end test of the agent LLM's tool-calling against the live model. Each
case feeds a synthetic scene and head pose into the model with the same system
prompt and native NAT function schemas as the live worker, executes tool
effects against deterministic fixtures, then checks the resulting scene
mutations against a per-case expectation.

## Prerequisites

The agent LLM must be running:

```bash
# weights resident in the background — start once, leave alone
uv run --project ~/hub/xr-ai/agent-samples/model-servers model_servers

```

By default the harness calls the agent LLM at `http://localhost:8107`. It does
not require the render-demo stack, capability services, MCP adapters, or LOVR.

## Run

```bash
# All built-in cases against the current system.txt
uv run --project agent-samples/xr-render-demo/worker \
  python agent-samples/xr-render-demo/eval/eval.py

# Subset by case name — fast iteration on a single failing cluster.
# Comma-separated; unknown names error out (mutually exclusive with the
# positional query arg below).
uv run --project agent-samples/xr-render-demo/worker \
  python agent-samples/xr-render-demo/eval/eval.py \
  --only move_left_one_meter,between_two_spheres

# Watcher-friendly equivalent: write case names (newline- or
# comma-separated; '#' comments OK) to eval/.only. Gitignored.
# Active subset is echoed at startup.

# One ad-hoc query (prints the raw LLM response)
uv run --project agent-samples/xr-render-demo/worker \
  python agent-samples/xr-render-demo/eval/eval.py "Move the cube up 30 cm"

# Score a prompt file other than the live worker's system.txt — e.g.
# main's version, a draft, or a checkout from another branch.
uv run --project agent-samples/xr-render-demo/worker \
  python agent-samples/xr-render-demo/eval/eval.py --prompt /tmp/alt-system.txt

# Score against a hosted model (e.g. nvidia/nemotron-3-super-120b-a12b at
# build.nvidia.com) instead of the local vLLM on 8107.  Set NVIDIA_API_KEY
# in the env first (or pass --agent-api-key).
export NVIDIA_API_KEY=nvapi-...
uv run --project agent-samples/xr-render-demo/worker \
  python agent-samples/xr-render-demo/eval/eval.py \
  --agent-llm   https://integrate.api.nvidia.com/v1/chat/completions \
  --agent-model nvidia/nemotron-3-super-120b-a12b
```

Run `uv sync` in `agent-samples/xr-render-demo/worker` before the first eval.

## Watcher

`eval_watch.sh` polls `system.txt`'s sha1 once per second (hash, not
mtime — editors and language servers re-save the file without
changing bytes when you switch focus). This allows a coding agent to
iterate on the prompt and read scores out of `/tmp/eval_loop.log`
without the user re-launching `eval.py` between rounds. Any content
change aborts the running eval and starts a new one once the file
has been quiet for 10 seconds.

```bash
agent-samples/xr-render-demo/eval/eval_watch.sh
tail -f /tmp/eval_loop.log

agent-samples/xr-render-demo/eval/eval_watch.sh /path/to/alt.txt   # different prompt
kill $(cat /tmp/eval_watch.pid)                                     # stop
```

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
grep "passed$" /tmp/eval_loop.log | tail
```

## Writing a case

Read `eval.py`'s `CASES` list — every shape (single-turn, pose
override, multi-turn `history`, undo `recent_moves`) is exemplified
there. Copy the closest existing case and edit. The case dict is
what the harness consumes directly; there's no case schema layer.

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
   to memorise as a template.

Fix overlaps by changing the prompt example, not the case. For
check #4, use colours and shapes outside the eval vocabulary
(turquoise / teal / lavender / magenta / cone / cylinder / capsule)
when reaching for a fixture word in a worked example.

## What the harness does not cover

- The live worker pipeline (VAD, STT, TTS, history bookkeeping).
- Real scene-service / LOVR effects (fixture-succeeded).
- Real visual queries (`ask_image`, `get_latest_frame`) — stubbed.
