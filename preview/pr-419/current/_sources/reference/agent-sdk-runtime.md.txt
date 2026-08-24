<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-runtime

`xr-ai-agent-runtime` provides typed, participant-scoped publication and fan-out
for composable agents. It does not own models, planning, memory, tools, media,
application state, or agent-created tasks. Refer to {doc}`python/index` for the
exact public APIs.

## Agents and tools

An `Agent` owns private state and exposes ordinary finite `Tool` or streaming
`AsyncTool` objects from `xr-ai-tools`. Direct callers execute those tools
directly; model loops expose finite tools through a `ToolSet`. The runtime never
wraps or redispatches tool calls.

```python
from pydantic import BaseModel
from xr_ai_runtime import Agent, AgentRuntime
from xr_ai_tools import Tool

class Text(BaseModel):
    text: str

class TextAgent(Agent):
    def __init__(self) -> None:
        self.uppercase = Tool(
            "uppercase", "Uppercase text.", Text, Text, self._uppercase
        )
        super().__init__((self.uppercase,))

    async def _uppercase(self, request: Text) -> Text:
        return Text(text=request.text.upper())

runtime = AgentRuntime()
text = runtime.register("text", TextAgent())
async with runtime:
    result = await text.uppercase.execute(Text(text="hello"))
```

When one model sees tools from several agents, namespace only the model-visible
catalog with `ToolSet.namespaced(...)`. Participant identity remains a typed
tool input supplied at the application boundary.

## Publication and ownership

`publish(topic, event)` is asynchronous fan-out. It waits for all subscriber
deliveries and then propagates failures. A subscriber that performs lengthy
work must hand it to its own bounded queue and return promptly. The runtime
owns delivery tasks only; an agent owns creation, synchronization, cancellation,
and cleanup for its resources and background work.

Callbacks and tools may run concurrently. Agents protect shared mutable state
with their own lock or queue so unrelated and streaming work is not globally
serialized. Participant controls such as start, stop, and status are ordinary
application tools; agent lifetime is not a model tool.

Topics use full Relay telemetry by default. High-cardinality transport topics
may declare `telemetry="none"` when their consumer emits one semantic operation
scope instead. Runtime publication and delivery scopes carry participant,
message, correlation, parent-message, source, and subscriber metadata. A
detached agent task whose lifetime exceeds its callback starts a fresh Relay
scope stack and preserves logical correlation in metadata rather than retaining
an ended delivery scope.
