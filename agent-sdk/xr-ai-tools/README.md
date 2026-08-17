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

Derived images use `ImageRegistry.put_derived(...)` to inherit the source
reference's owner, so participant or workflow cleanup releases the source and
all derived pixels together.

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

## Marker tracking

Install `xr-ai-tools[marker-tracking]` to use `MarkerTrackingTool`. The finite
`track_markers` tool acquires a participant's current camera frame and detects
QR codes with ZXing-C++ and ArUco markers with OpenCV. Its typed result
distinguishes an unavailable frame from a valid frame with no marker. Every
marker has the same `marker_type`, `value`, and four-corner contract: `value`
is decoded text for QR codes and the decimal marker ID for ArUco markers.
QR detection scans the complete frame at native resolution and at one bounded
enlarged resolution so small codes remain readable without discarding context.
Results from both scans are deduplicated and use source-frame coordinates.

```python
from xr_ai_tools.marker_tracking import (
    MarkerTrackingRequest,
    MarkerTrackingTool,
    MarkerType,
)

markers = MarkerTrackingTool(endpoint=processor_endpoint)
result = await markers.execute(
    MarkerTrackingRequest(participant_id="participant-1")
)
for marker in result.markers:
    print(marker.marker_type, marker.value, marker.corners)
```

Both marker families are enabled by default. Select families only while
initializing the tool; requests and results do not change. ArUco detection uses
`DICT_4X4_50` by default and accepts any OpenCV predefined dictionary name.

```python
aruco_only = MarkerTrackingTool(
    endpoint=processor_endpoint,
    marker_types=(MarkerType.ARUCO,),
    aruco_dictionary="DICT_6X6_250",
)
```

Generate sample QR and ArUco marker PNGs with the standalone utility beside the
tool. Its inline dependency metadata lets `uv` create an isolated environment
without modifying the project environment:

```bash
uv run xr_ai_tools/utilities/generate_marker.py qr "XR AI" --output qr.png
uv run xr_ai_tools/utilities/generate_marker.py aruco 23 --output aruco.png
uv run xr_ai_tools/utilities/generate_marker.py aruco 42 \
  --dictionary DICT_6X6_250 --output aruco-42.png
```

Run these commands from `agent-sdk/xr-ai-tools/`. Both commands produce square
512-pixel PNGs by default; use `--help` to see sizing and border options.

Call `release(participant_id)` when a participant disconnects. Applications
that expose the tool to a model should inject the active participant identity
at their workflow boundary, as they do for other participant-scoped tools.

## Magenta polygon image editing

Install `xr-ai-tools[image-editing]` to fill an image-space polygon with
standard magenta (`#FF00FF`). `ImagePolygonFillTool` accepts an image reference
and at least three ordered pixel coordinates. It connects the final point to
the first, validates that the polygon is inside the image and encloses an area,
and stores a new lossless PNG without changing the source image. The returned
reference inherits the source owner and can be passed directly to an image
query tool. Stale image references and invalid polygons return
`available=False` with a recoverable message instead of raising an internal
tool error.

```python
from pathlib import Path

from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.image_polygon import (
    ImagePoint,
    ImagePolygonFillRequest,
    ImagePolygonFillTool,
)

images = ImageRegistry()
source = images.put(Path("measurement.png"))
fill_polygon = ImagePolygonFillTool(images=images)

result = await fill_polygon.execute(
    ImagePolygonFillRequest(
        image=source,
        coordinates=[
            ImagePoint(x=40, y=30),
            ImagePoint(x=140, y=30),
            ImagePoint(x=140, y=130),
            ImagePoint(x=40, y=130),
        ],
    )
)
```
