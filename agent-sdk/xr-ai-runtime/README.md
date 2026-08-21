<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-runtime

The `xr-ai-agent-runtime` distribution provides typed pub/sub for composable XR AI agents. An
`Agent` owns private state and exposes ordinary `Tool` or `AsyncTool` instances
from `xr-ai-tools`.

Tools have one invocation path everywhere:

- Direct callers and other agents use `Tool.execute()` or `AsyncTool.stream()`.
- Model loops expose unary tools with `ToolSet` and `handle_tool_call()`.
- The agent runtime never wraps or redispatches a tool call.

Unary tools return one validated Pydantic response or `None` for an
acknowledged side effect. Streaming tools yield validated chunks without
buffering the complete result.

```python
from pydantic import BaseModel
from xr_ai_runtime import Agent, AgentRuntime
from xr_ai_tools import Tool


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    text: str


class TextAgent(Agent):
    def __init__(self) -> None:
        self.echo = Tool(
            "echo",
            "Echo the supplied text.",
            TextRequest,
            TextResponse,
            self._echo,
        )
        self.uppercase = Tool(
            "uppercase",
            "Uppercase the supplied text.",
            TextRequest,
            TextResponse,
            self._uppercase,
        )
        super().__init__((self.echo, self.uppercase))

    async def _echo(self, request: TextRequest) -> TextResponse:
        return TextResponse(text=request.text)

    async def _uppercase(self, request: TextRequest) -> TextResponse:
        return TextResponse(text=request.text.upper())


runtime = AgentRuntime()
text = runtime.register("text", TextAgent())

async with runtime:
    result = await text.uppercase.execute(TextRequest(text="hello"))
```

Another agent receives the concrete tools it needs and calls them normally.
No runtime address or adapter is involved.

When tools from several agents are combined for one model, namespace them at
the workflow boundary with
`ToolSet.namespaced({"vision": vision.tools, "planner": planner.tools})`.
This remaps only model-visible catalog names; the agents and underlying tools
remain unchanged. Participant identity needed by a direct tool belongs in that
tool's typed request. Relay records each publication as a function scope
and each subscription delivery as an agent scope, carrying topic, participant,
message, correlation, parent-message, source, and subscriber metadata. Tool and
model scopes invoked by a callback nest under that delivery. Agent-owned
detached tasks must open a fresh Relay scope stack when their lifetime extends
beyond the callback, then add an agent scope and preserve logical correlation in
metadata. This prevents a detached operation from becoming the child of a scope
that has already ended.

`publish(topic, event)` is the separate asynchronous fan-out operation for
participant-scoped or global events. An agent that owns resources or background
work is responsible for controlling them, including creating, cancelling, and
awaiting its own tasks. The runtime neither knows nor controls whether an
agent's internal work is running. `publish()` waits for every fan-out delivery
to settle before propagating any subscriber failures. Subscriber callbacks
must hand off lengthy work to agent-owned bounded queues and return promptly.

Topics default to `telemetry="full"`. High-cardinality transport topics use
`"none"` when their consumer aggregates fragments and records one semantic
operation scope. Delivery and failure behavior remains unchanged. Keeping the
policy on the topic declaration gives every producer and consumer the same
cardinality behavior.

Tools and subscription callbacks may run concurrently. An agent whose mutable
state is shared between them owns the appropriate synchronization, such as an
`asyncio.Lock` or a private queue. This avoids imposing serialization and
head-of-line blocking on unrelated or streaming tools.

Domain controls such as `start_monitoring`, `stop_monitoring`, and `status` are
ordinary tools. Agent lifetime itself is not a model tool. Model loops,
planning, memory, and model clients remain agent implementations. Raw video
stays on the DeviceIOHub path. `VoiceAgent` publishes final pre-gate STT
results on `voice.transcript` for explicit subscribers. Its media session is
private, and its bounded transcript-delivery queue prevents slow subscribers
from delaying STT or command gating.
