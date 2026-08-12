<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI native tools

`xr-ai-tools` is the toolkit-independent native tools layer for XR AI.
`Tool` gives voice, background triggers, and model-driven agents one typed
Pydantic invocation interface. NeMo Relay manages every tool execution;
model-backed tools use injected `xr-ai-models` services.

## Native tools and model tool calls

The base install supplies finite `Tool` and streaming `AsyncTool` types. Install
`xr-ai-tools[relay]` for the small model tool-call helpers:

```python
from pydantic import BaseModel
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tool_calling import handle_tool_call, tool_definitions


class LookupRequest(BaseModel):
    query: str


class LookupResult(BaseModel):
    answer: str


async def lookup(request: LookupRequest) -> LookupResult:
    return LookupResult(answer=request.query)


lookup_tool = Tool(
    "lookup",
    "Look up one answer.",
    LookupRequest,
    LookupResult,
    lookup,
)
tools = (lookup_tool,)
tool_set = ToolSet(tools)

response = await llm.chat(messages, tools=tool_definitions(tools))
for call in response.tool_calls or ():
    result = await handle_tool_call(call, tool_set)
    messages.append(result.message)
```

`tool_definitions(...)` adapts native tools to `xr-ai-models` `ToolDef` values.
`handle_tool_call(...)` validates and invokes one model-produced `ToolCall`, then
returns a tool-role `ChatMessage` plus its `return_direct` hint. The application
or agent owns prompts, model calls, conversation state, iteration policy, and
whether calls run sequentially or concurrently.

## Finite and streaming live vision tools

Install `xr-ai-tools[relay,live-vision]` for two independent current-frame
tools. `LiveVisionTool` is a finite `Tool` that returns one complete
`VisionResponse` for agentic planning. `StreamingVisionTool` is an `AsyncTool`
that yields typed `VisionChunk` values. It has no voice dependency or output
transport; applications decide how to consume its async stream.

Each tool owns its own participant frame source and calls an injected
`VLMService`. Both forward controlled Relay headers and redact inline camera
frames from events while preserving provider input. The finite path uses
Relay's managed tool and LLM boundaries; the streaming path uses a typed tool
scope around Relay's managed streaming LLM boundary.
