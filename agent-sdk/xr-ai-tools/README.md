<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI native tools

`xr-ai-tools` is the toolkit-independent native tools layer for XR AI.
`Tool` gives voice, background triggers, and model-driven agents one typed
Pydantic invocation interface. NeMo Relay manages every new tool execution;
model-backed tools use injected `xr-ai-models` services rather than exposing a
model client to an application trigger.

## Native tools and tool-driven agents

The base install supplies `Tool`, `AgentRunner`, and `as_agent_tool`. Install
`xr-ai-tools[relay]` for the bundled bounded tool-driven `Agent`:

```python
from pydantic import BaseModel
from xr_ai_tools import Tool
from xr_ai_tools.agents import Agent


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
agent = Agent(
    name="assistant",
    llm=llm,
    system_prompt="Use the available tools.",
    tools=(lookup_tool,),
)
```

`AgentRunner` is the small async turn protocol behind `as_agent_tool(...)`.
The bundled `Agent` is the basic stateless tool loop; applications can expose a
custom, Fabric-backed, or framework-backed runner through the same registered
`Tool`. That keeps voice, text, and autonomous background work on one
invocation path. Relay observes model calls inside a tool-backed runner; the
application never calls an LLM client as a separate control path.

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
