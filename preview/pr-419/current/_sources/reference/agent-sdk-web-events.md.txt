<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-web-events

`xr-ai-web-events` presents application-selected runtime events in a bounded,
read-only browser view. It does not tail files, persist data, or enter model,
voice, media, or DeviceIOHub authentication paths. Public APIs are in
{doc}`python/index`.

## Publishing selected events

Applications explicitly project compact JSON-compatible payloads to
`WEB_EVENT_TOPIC`. Each serialized event is limited to 16 KiB.

```python
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent, WebEventsAgent

viewer = runtime.register("web-events", WebEventsAgent(title="Agent events"))

await ctx.publish(
    WEB_EVENT_TOPIC,
    WebEvent(
        topic="application.reading",
        title="Readings",
        payload=reading.model_dump(mode="json"),
    ),
)

async with viewer:
    async with runtime:
        await run_application()
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
