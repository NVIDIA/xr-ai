<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# glasses-agent-langchain

LangChain variant of `agent-samples/glasses-agent`. It keeps the same XR
process stack and user-facing smart-glasses behavior, but replaces the ordinary
request-time LLM/tool loop with LangChain and LangGraph primitives.

## Run

```bash
cd agent-samples/glasses-agent-langchain
uv sync
uv run glasses_agent_langchain
```

The sample starts the same services as `glasses-agent`: hub, STT, TTS, VLM,
small LLM, agent LLM, VLM MCP, video MCP, transcript MCP, and the worker.

## What Stays The Same

- Hub IPC, LiveKit transport, VAD, STT, TTS, background VLM observation, scene
  condensation, transcript persistence, demo recording, and guided playback are
  inherited from `glasses-agent`.
- Agents still talk to the hub through `xr-ai-agent`; LiveKit remains hidden
  from worker code.
- Demo recording and guidance remain application state machines, not LangChain
  graph nodes.
- The two-LLM split is preserved: the small LLM handles quick acknowledgements
  and lightweight calls, while the larger agent LLM handles tool calling and
  deeper reasoning.

## What Changes From `glasses-agent`

| Area | `glasses-agent` | `glasses-agent-langchain` |
|---|---|---|
| Request-time tool loop | Custom OpenAI-compatible tool loop in `processors.py` | LangChain `create_agent` runtime |
| MCP tools | Manual MCP schema conversion to OpenAI tool specs | `langchain-mcp-adapters` loads MCP tools as LangChain tools |
| Conversation state | Local last-4-turn `_history` list | LangGraph checkpointed per-participant messages with bounded trimming |
| Runtime XR context | Injected into a synthetic user prompt | Passed as `GlassesRuntimeContext` and injected by LangChain middleware |
| Tool safety | Manual `ask_image` path checks in custom tool routing | Wrapped LangChain `ask_image` tool validates image paths before VLM calls |
| JSON side calls | Manual JSON extraction in several paths | Quick ack, demo analysis, guidance Q&A, and scene condensation use structured LangChain output |
| Demo guidance lookup | Name lookup with first-demo fallback | Stable task numbers plus name matching; ambiguous requests ask for a task number |

## LangChain Integration

At startup, the worker connects to the VLM and video MCP servers through
`MultiServerMCPClient`:

```python
langchain_mcp = MultiServerMCPClient({
    "vlm": {"transport": "http", "url": cfg.vlm_mcp.rstrip("/") + "/mcp"},
    "video": {"transport": "http", "url": cfg.video_mcp.rstrip("/") + "/mcp"},
})
langchain_tools = await langchain_mcp.get_tools()
```

`QueryProcessor` creates a single LangChain agent graph. Thinking mode,
conversation trimming, and XR context injection are controlled through runtime
context and middleware instead of separate agent graphs or prompt string
assembly.

## XR Context And Memory

`AgentMemory` remains the source of truth for XR domain state:

- recent visual observations,
- scene summary,
- active recording,
- recorded demonstrations,
- analyzed demo steps and instructions.

For each ordinary user request, the worker builds a `GlassesRuntimeContext`
containing a read-only memory snapshot, participant ID, reference timestamp,
current frame path, fresh VLM frame description, and any active recording or
guidance state. Middleware turns that context into model-visible instructions
without saving it as chat history.

The sample still uses persistent FastMCP clients for latency-sensitive app
services such as background VLM observation, frame prefetch, guidance checks,
and transcript persistence. LangChain MCP tools are reserved for the
request-time agent graph.

## Task Number Guidance

Recorded demonstrations are numbered in recording order:

- first recording: `task 1`,
- second recording: `task 2`,
- and so on.

Users can start recordings with natural names or explicit numbering, for
example:

```text
start recording for table arrangements
start recording for task 2 table arrangements
```

The canonical task number still comes from recording order. During guidance,
users can refer to either the task number or task name:

```text
show me how to do task 2
show me how to do table arrangements
```

If the request does not match a known task, the agent asks the user to choose
from the numbered list, such as:

```text
Please tell me the task number: task 1 -- pico headset, task 2 -- table arrangements.
```

The worker keeps that selection pending until the user provides a valid task
number or matching task name.

## Notes

- `AgentMemory` and transcript MCP still own observations and demo state.
  LangGraph checkpointing is used only for the conversational agent path.
- Demo recordings are session-local unless restored by explicit code; old demos
  are not restored automatically across runs.
- The LangChain worker is intentionally a framework comparison sample. Shared
  XR behavior should remain aligned with `agent-samples/glasses-agent`.
