<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI agent runtime

`xr-ai-agent-runtime` manages resource lifetimes, background tasks, and typed
pub/sub for composable XR AI agents. An `Agent` owns private state and exposes
ordinary `Tool` or `AsyncTool` instances from `xr-ai-tools`.

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
tool's typed request. Relay supplies nested execution tracing, while runtime
message metadata applies only to pub/sub.

`publish(topic, event)` is the separate asynchronous fan-out operation for
events. `ctx.start_task()` starts background work owned by the current agent's
runtime lifetime. An agent that owns resources implements the optional
`lifespan()` async context; most agents need no lifecycle code.

The runtime cancels tasks created through `ctx.start_task()` before resources
are released. An external caller is responsible for finishing or cancelling
direct tool calls before leaving the runtime scope.

If a runtime-owned background task fails, the runtime stops accepting work,
cancels its remaining owned tasks, and raises `RuntimeFailedError` from the
original failure on later operations. Shutdown surfaces the original failure.
`publish()` waits for every fan-out delivery to settle before propagating any
subscriber failures.

Tools and subscription callbacks may run concurrently. An agent whose mutable
state is shared between them owns the appropriate synchronization, such as an
`asyncio.Lock` or a private queue. This avoids imposing serialization and
head-of-line blocking on unrelated or streaming tools.

Domain controls such as `start_monitoring`, `stop_monitoring`, and `status` are
ordinary tools. Agent lifetime itself is not a model tool. Model loops,
planning, memory, and model clients remain agent implementations. Raw audio and
video stay on the XR-Media-Hub path.
