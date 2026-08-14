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
`VideoMemoryTools` groups recorded selection by timing:

- `latest_tools` contains `get_latest_video` and `get_latest_frames`;
  their windows end at the newest recorded timestamp and require only
  `duration_seconds`.
- `historical_tools` contains `get_historical_frame`,
  `get_historical_frames`, and `get_historical_video`; they share one
  absolute `start_us`, with video and sampling adding `duration_seconds`.

`get_current_frame` joins the latest group conceptually as the live-image
equivalent. Every selected frame carries the same `ImageReference`; timed
frames also share the `TimedImage` contract consumed by video queries.

`ImageQueryTool` returns one complete `ImageQueryResult` for any image reference;
`MultiImageQueryTool` queries an ordered image collection, and `VideoQueryTool`
queries chronological `TimedImage` frames with their timestamps and relative
offsets. `StreamingImageQueryTool` preserves the low-latency single-image path.
All four tools use one list-based inference implementation. The shipped Cosmos
deployment accepts at most four images in one `query_images` or `query_video`
call. This inference limit is separate from the video-memory sampling budget;
select the relevant subset before querying the VLM.

In-memory bytes use a bounded `ImageRegistry`, whose opaque handles keep tool
results and telemetry small. Registries accept only their own opaque references
by default. Trusted applications may opt into local paths, file URIs, and
HTTP(S) URLs with `ImageRegistry(allow_external=True)`, but must not expose an
external-enabled query tool directly to an untrusted model. Query tools forward
controlled Relay headers and redact every image location from VLM events while
preserving provider input. Timelines supplied to video inference are described
as estimates because recorded-frame timestamps are interpolated from chunk
metadata rather than persisted per-frame presentation timestamps.

```python
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageReference, ImageRegistry
from xr_ai_tools.video_memory import LatestFramesRequest
from xr_ai_tools.vision import (
    ImageQueryRequest,
    ImageQueryTool,
    VideoQueryRequest,
    VideoQueryTool,
)

images = ImageRegistry(allow_external=True)
frames = CurrentFrameTool(endpoint=endpoint, images=images)
vision = ImageQueryTool(images=images, vlm=vlm)
video = VideoQueryTool(images=images, vlm=vlm)

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

# The newest recorded window needs no caller-supplied timestamp.
sample = await video_memory.get_latest_frames.execute(
    LatestFramesRequest(
        participant_id="alice",
        duration_seconds=10,
        frame_budget=4,
    )
)
change = await video.execute(
    VideoQueryRequest(frames=sample.frames, query="What changed over time?")
)
```

## QR-code reading and extraction

Install `xr-ai-tools[qr-code]` to use `QRCodeTool`. The finite
`read_qr_codes` tool acquires a participant's current camera frame and uses
ZXing-C++ by default. Its typed result distinguishes an unavailable frame from
a valid frame with no readable code, and returns every decoded UTF-8 payload
with optional source-image corner coordinates. When one frame contains
multiple QR codes, `result.codes` contains one entry for each code.

```python
from xr_ai_tools.qr_code import QRCodeRequest, QRCodeTool

qr_codes = QRCodeTool(endpoint=processor_endpoint)
result = await qr_codes.execute(QRCodeRequest(participant_id="participant-1"))
for code in result.codes:
    print(code.data, code.corners)
```

Frame acquisition and extraction are separate. Pass any sync or async callable
as `extractor=` to replace ZXing-C++ without changing the tool or its callers.
The callable receives one RGB `PIL.Image.Image` and returns an iterable of
`DecodedQRCode` values or equivalent dictionaries; `corners` may be omitted by
backends that decode without localization.

```python
async def model_extractor(image):
    return await custom_qr_model.extract(image)


qr_codes = QRCodeTool(
    endpoint=processor_endpoint,
    extractor=model_extractor,
)
```

Call `release(participant_id)` when a participant disconnects. Applications
that expose the tool to a model should inject the active participant identity
at their workflow boundary, as they do for other participant-scoped tools.
