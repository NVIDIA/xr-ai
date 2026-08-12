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

## Live vision tool and direct voice responder

Install `xr-ai-tools[relay,live-vision]` for `LiveVisionTool`. The finite
`look_at_current_frame` tool acquires a participant's current frame and returns
one complete `VisionResponse` for agentic planning. `LiveVisionResponder`
shares that tool's frame source and streams only the direct voice path. Both
call an injected `VLMService` through Relay's matching managed LLM boundary,
forward controlled Relay headers, and redact the inline camera frame from
events while the provider receives the original. `LiveVisionTool.release()`
clears participant frame state.

Relay's managed tool API accepts completed JSON results, while its managed LLM
API supports streaming. Agentic vision therefore uses `Tool.execute()` and a
complete result; direct voice remains an application-owned response stream
with Relay managing the nested model call.
