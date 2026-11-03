<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-web-events

`xr-ai-web-events` presents application-selected runtime events in a bounded,
read-only browser view. It does not tail files, persist data, or enter model,
voice, media, or DeviceIOHub authentication paths. Refer to {doc}`python/index`
for the public APIs.

## Publishing selected events

Applications explicitly project compact JSON-compatible payloads to
`WEB_EVENT_TOPIC`. Each event's serialized `payload` is limited to 16 KiB.

```python
from pydantic import BaseModel
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, Topic, subscribe
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent, WebEventsAgent

class Reading(BaseModel):
    value: float

READING_TOPIC = Topic("application.reading", Reading)

class ReadingEventsAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    @subscribe(READING_TOPIC)
    async def report_reading(
        self,
        reading: Reading,
        ctx: RuntimeContext,
    ) -> None:
        await ctx.publish(
            WEB_EVENT_TOPIC,
            WebEvent(
                topic="application.reading",
                title="Readings",
                payload=reading.model_dump(mode="json"),
            ),
        )

runtime = AgentRuntime()
viewer = runtime.register("web-events", WebEventsAgent(title="Agent events"))
runtime.register("reading-events", ReadingEventsAgent())

async with viewer:
    async with runtime:
        await runtime.publish(READING_TOPIC, Reading(value=21.5))
```

Start the viewer before worker readiness so a bind failure fails the worker.
`start()` and `stop()` are idempotent, and the agent is an async context manager.
The page groups events by participant and presentation topic. Its cursor API
reports bounded-history rollover; `/healthz` checks the listener.

## Exposure boundary

The default is `http://127.0.0.1:8092`. The listener has no application
authentication or TLS, and payloads may include transcripts or camera-derived
text. Prefer an SSH tunnel for remote development:

```bash
ssh -L 8092:127.0.0.1:8092 user@xr-host
```

Direct exposure belongs only on a trusted network or behind an authenticated
TLS reverse proxy. The proxy must preserve an allowed `Host` value or rewrite it
to the configured listener address. Listener addresses are IPv4-only.
