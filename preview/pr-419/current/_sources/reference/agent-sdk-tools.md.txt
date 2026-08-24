<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-tools

`xr-ai-tools` is the toolkit-independent native tool layer. Finite `Tool` and
streaming `AsyncTool` objects share one typed Pydantic invocation contract, and
every execution passes through NeMo Relay. Exact public APIs are in
{doc}`python/index`.

## Bounded model tool loops

Install the `relay` extra for `ToolSet` adapters and `run_tool_loop()`:

```python
from pydantic import BaseModel
from xr_ai_models import ChatMessage
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tool_calling import run_tool_loop

class LookupRequest(BaseModel):
    query: str

class LookupResult(BaseModel):
    answer: str

async def lookup(request: LookupRequest) -> LookupResult:
    return LookupResult(answer=request.query)

tools = ToolSet((Tool(
    "lookup", "Look up one answer.", LookupRequest, LookupResult, lookup
),))

async def call_model(messages, definitions):
    return await llm.chat(messages, tools=definitions, max_tokens=1024)

result = await run_tool_loop(
    [ChatMessage(role="user", content="look up the answer")],
    tools,
    call_model,
    max_iterations=4,
    max_tool_calls=8,
)
```

The runner is stateless and bounded. It validates tool IDs and arguments,
executes emitted calls sequentially, returns a resumable transcript and audit,
and reports unknown or invalid calls to the model for repair. Exhausted budgets,
empty or truncated responses, and unsafe mixed `return_direct` batches raise
typed errors with the partial audit. It never retries an already executed tool.
Applications retain prompts, history, model parameters, retries, participant
context, cancellation, and task ownership.

Use `ToolSet.namespaced()` when independently named groups share a model-visible
catalog. Aliasing changes only catalog names. Only finite tools belong in a
`ToolSet`; callers consume an `AsyncTool` explicitly with `stream()`.

## Capability services

The `services` extra provides typed msgpack over ZMQ RPC primitives plus tracking,
video-memory, text-memory, and spatial tool groups. Applications compose these
tools directly; the private transport is not an agent API and does not require
an HTTP or agent framework.

## Image selection and inference

Selection and inference are separate contracts:

- `CurrentFrameTool` selects a live frame.
- `VideoMemoryTools.latest_tools` selects the newest duration window.
- `VideoMemoryTools.historical_tools` selects an absolute `start_us` window.
- `ImageQueryTool`, `MultiImageQueryTool`, `StreamingImageQueryTool`, and
  `VideoQueryTool` resolve selected image references through an injected
  `VLMService`.

Every selected image uses an opaque `ImageReference`. A bounded
`ImageRegistry` keeps media bytes out of tool results and telemetry. Registries
accept only their own references unless a trusted application opts into local
paths or URLs with `allow_external=True`; do not expose an external-enabled
query tool directly to an untrusted model. Derived images inherit the source
owner so participant cleanup releases the complete image lineage.

The shipped Cosmos endpoint accepts at most four images per inference request.
That model limit is independent of the larger recorded-frame selection budget;
applications select the relevant subset before inference. Recorded timestamps
are estimates interpolated from chunk metadata.

```python
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryRequest, ImageQueryTool

images = ImageRegistry()
frames = CurrentFrameTool(endpoint=endpoint, images=images)
vision = ImageQueryTool(images=images, vlm=vlm)

frame = await frames.execute(CurrentFrameRequest(participant_id="alice"))
answer = await vision.execute(
    ImageQueryRequest(image=frame.image, query="What is on the table?")
)
```

## Marker tracking

The `marker-tracking` extra detects QR and ArUco markers in a current frame or a
previously selected image. `MarkerTrackingTool` distinguishes an unavailable
frame from a valid frame with no marker and returns a shared marker family,
value, and four-corner shape. QR values are decoded text; ArUco values are
decimal IDs. Both detectors scan native resolution and one bounded enlargement,
then deduplicate into source-frame coordinates.

Pass the same `ImageRegistry` to frame selection and marker tracking when
detection and later annotation must use exactly one frame. Release participant
state on departure. Applications inject participant identity rather than
exposing it as a model-selected argument.

Generate printable markers from `agent-sdk/xr-ai-tools/`:

```bash
uv run xr_ai_tools/utilities/generate_marker.py qr "XR AI" --output qr.png
uv run xr_ai_tools/utilities/generate_marker.py aruco 23 --output aruco.png
```

## Derived-image editing

The `image-editing` extra provides `ImagePolygonFillTool` for filling a validated
image-space polygon with standard magenta. It stores a new lossless PNG, keeps
the source unchanged, and inherits source ownership. Stale references and
invalid polygons return a recoverable unavailable result rather than an
internal tool exception.
