<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo

Voice-driven XR scene manipulation sample. A supervisor routes natural-language
commands to five focused subagents; each subagent calls typed function groups
from `xr-ai-tools` to read and mutate the live XR scene.

## File map

```
agent-samples/xr-render-demo/
  main.py                        orchestrator entry point
  yaml/                          per-service YAML configs + model profile
  worker/
    xr_render_demo_worker/
      app.py                     wires xr-ai-tools groups + RenderAgent
      agent.py                   xr-ai-runtime Agent; routes voice turns
      supervisor.py              SceneSupervisor: top-level tool loop
      supervisor_prompt.txt      supervisor system prompt
      spatial_ops.py             typed spatial tools shared by subagents
      models.py                  SceneRequest / SceneReply / SubagentTask
      scene.py                   SceneContext: snapshot, diff, move history
      agents/
        placement/               subagent: move / add relative to objects
        appearance/              subagent: color, size
        object/                  subagent: create, remove, swap
        vision/                  subagent: current frame + historical frame
        memory/                  subagent: recall conversation history
  eval/
    xr_render_demo_eval/
      harness.py                 offline eval runner (mock tools)
      cases.py                   eval case definitions
      supervisor.py              supervisor-level offline cases
      subagents.py               per-subagent offline cases
      live_manip.py              live scene manipulation eval (13 cases)
      live_smoke.py              basic stack-is-alive check
      live_garble.py             garbled-utterance robustness eval
      live_explore.py            exploratory scene-query eval
  scene/                         xr_render_scene service (LOVR + OpenXR)
```

## Composition chain

```
voice query
  └─ RenderAgent (xr-ai-runtime Agent)
       └─ SceneSupervisor.handle()           one turn, per-participant lock
            └─ run_tool_loop (LLM + subagent tools)
                 ├─ make_placement_agent()   SceneTools + TrackingTools
                 ├─ make_appearance_agent()  SceneTools
                 ├─ make_object_agent()      SceneTools + TrackingTools
                 ├─ make_vision_agent()      CurrentFrameTool + ImageQueryTool
                 │                           + VideoMemoryTools (optional)
                 └─ make_memory_agent()      TextMemoryTools
```

`app.py` allocates each xr-ai-tools group and passes them to `SceneSupervisor`.
The supervisor exposes each subagent as a `Tool`; the LLM delegates to whichever
it needs and aggregates their results.

## How to extend

### Add a subagent

1. Create `worker/xr_render_demo_worker/agents/<name>/agent.py` with a
   `make_<name>_agent(...)` function returning a `Tool`.
2. Add a system prompt at `agents/<name>/prompt.txt` if needed.
3. Export it from `agents/__init__.py`.
4. Pass it into `subagent_tools` in `supervisor.py`'s `__init__`.
5. Add offline test cases in `eval/xr_render_demo_eval/subagents.py`.

### Add a scene tool / function group

1. Implement the group in `xr-ai-tools` (or locally in `worker/`) following
   the `xr_ai_tools` patterns (typed request/result, `execute()` method).
2. Allocate it in `app.py` alongside the other groups.
3. Pass it through to the subagent that needs it.

### Add an eval case

**Offline (no running stack):** add a `Case` to `eval/xr_render_demo_eval/cases.py`
and reference it in `harness.py` or `supervisor.py`/`subagents.py`.

**Live:** add a dict to `CASES` in the appropriate `live_*.py` file:
```python
{
    "name": "my_case",
    "fixtures": [("sphere", x, y, z, r, g, b, size)],
    "prompt": "...",
    "check": lambda ids, o: ...,   # True = PASS
}
```

### Edit a prompt

Prompts live in `agents/<subagent>/prompt.txt` and `supervisor_prompt.txt`.
Edit in place; the file is read at startup (no rebuild needed for prompt-only
changes). Offline cases in `harness.py` exercise the prompt without a live stack.

## Running

```bash
# Start model servers first (models marked `reused` in the YAML profile):
uv run --project agent-samples/model-servers model_servers

# Start the demo stack:
uv run --project agent-samples/xr-render-demo xr_render_demo

# Stop: send SIGTERM to the orchestrator python process (not individual services).

# Offline eval:
uv run --project agent-samples/xr-render-demo/eval xr_render_demo_eval

# Live eval (stack must be running):
uv run --project agent-samples/xr-render-demo/eval xr_render_demo_live_manip
```

## Tracing and debugging

Each voice turn carries a `trace_id` equal to `ctx.metadata.message_id` from
the xr-ai-runtime `RuntimeContext`. Filter logs by it:

```
grep "trace=<id>" <log-file>
```

Key log events (all at `DEBUG` level in `supervisor.py`):

| Event | Message pattern |
|-------|----------------|
| Turn received | `supervisor turn participant=... trace=... transcript=...` |
| Subagent delegated | emitted by the subagent's `run_tool_loop` |
| Tool rejected (ValueError) | `tool <name> rejected input: ...` |
| Tool failed (unexpected) | `tool <name> failed unexpectedly` (with traceback) |
| Turn failed | `xr-render turn failed for <participant>: ...` |

Expected degradation (tool error converted to model scratch, turn continues):
`ValueError` from a tool call (bad arguments, object not found).

Unexpected failures (traceback logged, turn aborted): any other exception
propagating out of `SceneSupervisor.handle()`.
