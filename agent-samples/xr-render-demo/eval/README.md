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
harness audits at startup for verbatim overlap (utterances, scene
coords, recent-moves coords) and prints a warning. Fix overlaps by
changing the prompt example, not the case.

## What the harness does not cover

- The live worker pipeline (VAD, STT, TTS, history bookkeeping).
- Real render-mcp / LOVR effects (fake-succeeded).
- Real visual queries (`ask_image`, `get_latest_frame`) — stubbed.
