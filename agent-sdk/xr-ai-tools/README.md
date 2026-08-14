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
from xr_ai_models import ChatMessage
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
messages.append(
    ChatMessage(
        role="assistant",
        content=response.content,
        tool_calls=response.tool_calls,
    )
)
for call in response.tool_calls or ():
    result = await handle_tool_call(call, tool_set)
    messages.append(result.message)
    if result.return_direct:
        final_answer = result.message.content
        break
```

`tool_definitions(...)` adapts native tools to `xr-ai-models` `ToolDef` values.
`handle_tool_call(...)` validates and invokes one model-produced `ToolCall`, then
returns a tool-role `ChatMessage` plus its `return_direct` hint. The application
or agent owns prompts, model calls, conversation state, iteration policy, and
whether calls run sequentially or concurrently. A unary side-effect tool uses
`result_model=None`, returns `None`, and produces `null` as its model-visible
result.

Tool catalogs can assign model-visible aliases without wrapping or changing the
underlying tools:

```python
tools = ToolSet({"camera_status": vision.status})
```

When composing independently named tool groups, namespace them at the workflow
boundary:

```python
tools = ToolSet.namespaced({
    "vision": vision.tools,
    "planner": planner.tools,
})
```

This exposes names such as `vision__status` and `planner__status` to the model.
Only finite `Tool` instances belong in a `ToolSet`; streaming `AsyncTool`
instances are consumed explicitly with `stream()`.

## Typed capability services

Install `xr-ai-tools[services]` for the msgpack/ZMQ RPC client/server and the
shared tracking, video-memory, text-memory, and spatial
building blocks. Applications compose these finite tools into their own
`ToolSet`; service processes use the matching RPC primitives without pulling
in an agent framework or HTTP server.

Each capability has a focused import surface: `tracking`, `video_memory`,
`text_memory`, `spatial`, and their shared request types in `types`. RPC
transport remains isolated in `rpc`.

## Image selection and VLM query tools

Install `xr-ai-tools[frames]` for live-frame selection and
`xr-ai-tools[vision]` for VLM queries. Image selection does not invoke a model:
`CurrentFrameTool` returns an `ImageFrame`, while
`VideoMemoryTools.get_frame_from_time` and `sample_recorded_video` return
recorded frames. Every result carries an `ImageReference` and retains its path
or frame metadata.

`ImageQueryTool` returns one complete `ImageQueryResult` for any image reference;
`StreamingImageQueryTool` yields `ImageQueryChunk` values. References may point
to a selected frame, local path, file URI, or HTTP(S) URL. In-memory bytes use a
bounded `ImageRegistry`, whose opaque handles keep tool results and telemetry
small. Both query tools forward controlled Relay headers and redact image
locations from VLM events while preserving provider input.

```python
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageReference, ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

images = ImageRegistry()
frames = CurrentFrameTool(endpoint=endpoint, images=images)
vision = ImageQueryTool(images=images, vlm=vlm)

frame = await frames.execute(CurrentFrameRequest(participant_id="alice"))
answer = await vision.execute(
    ImageQueryRequest(image=frame.image, query="What is on the table?")
)

# The same VLM tool accepts images that did not come from a camera tool.
other = await vision.execute(
    ImageQueryRequest(
        image=ImageReference(uri="/tmp/reference.png"),
        query="Does this match the current object?",
    )
)
```
