<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-web-events

`xr-ai-web-events` provides a read-only browser view for selected typed runtime
events. It groups live output by participant and presentation topic without
tailing application files or entering model, voice, or media paths. The bounded
in-memory history is for observation only; durable output remains an
application responsibility.

Applications opt in explicitly by publishing compact JSON-compatible payloads
to `WEB_EVENT_TOPIC`:

```python
from xr_ai_web_events import WEB_EVENT_TOPIC, WebEvent, WebEventsAgent

viewer = runtime.register(
    "web-events",
    WebEventsAgent(title="Lab instrument events"),
)


async def report_reading(reading, ctx) -> None:
    await ctx.publish(
        WEB_EVENT_TOPIC,
        WebEvent(
            topic="instruments.readings",
            title="Instrument readings",
            payload=reading.model_dump(mode="json"),
        ),
    )


async with viewer:
    async with runtime:
        await run_application()
```

The server defaults to `http://127.0.0.1:8092`. Start it before the worker
announces readiness so a bind failure fails the worker instead of leaving a
partially ready stack. `start()` and `stop()` are idempotent, and the agent is
also an async context manager.

The page polls the same-origin read-only API and dynamically creates participant
and topic views. `/api/events?after=<sequence>` reports cursor rollover so a
browser can recover after falling behind the bounded store. `/healthz` reports
whether the HTTP listener can answer requests.

The shipped `simple-vlm-example` worker publishes participant-scoped VLM queries
and response chunks, while `xr-render-demo` publishes XR requests and spoken
agent output. Both configure and start their viewer from the sample worker YAML.

The listener has no application authentication or TLS. Its loopback default is
intentional because payloads may contain speech transcripts or camera-derived
text. For remote development, keep it on loopback and use an SSH tunnel:

```bash
ssh -L 8092:127.0.0.1:8092 user@xr-host
```

Do not bind it to `0.0.0.0` on an untrusted network. A remotely exposed viewer
needs an authenticated TLS reverse proxy chosen by the deployment owner; it
does not reuse the media hub's participant credentials. The server also rejects
unrecognized HTTP `Host` names. A reverse proxy must preserve a configured host,
use a literal listener address, or rewrite `Host` to `127.0.0.1`.
