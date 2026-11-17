<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Web events viewer (agent-sdk/xr-ai-web-events/xr_ai_web_events/_agent.py, _store.py, _server.py, _models.py, static/app.js)

## Role in a turn
Pure observability sidecar: a runtime subscriber (`web-events.event` topic, `telemetry="none"`) that stores events in a ring buffer and serves them to a browser via HTTP polling. It is **not** on the user-visible critical path of a turn except for (a) the small publish cost a producer pays if it awaits `publish` to this topic, and (b) event-loop/GIL contention from the HTTP threads. Viewer freshness is what it controls.

## Critical-path calls
Producer side: `runtime.publish(WEB_EVENT_TOPIC, WebEvent(...))` -> pydantic validate including a `json.dumps` size check of the payload (_models.py:38-52, validator `_bound_payload`) -> deep copy -> `_capture` subscriber -> `_EventStore.append` with `threading.Lock`, `model_copy(deep=True)` (_store.py:41-49). Producer awaits all of this (runtime gather).
Consumer side (browser, separate threads): `GET /api/events?after=<cursor>` -> `store.after()` under the same lock -> `json.dumps` -> write (_server.py:150-174, _store.py:51-67). Browser polls with `setTimeout(poll, 500)` after each response (static/app.js:111, 132).

## Fixed waits & timers
| Constant | Value | Where | Effect |
|---|---|---|---|
| Browser poll interval | 500 ms (after response completes) | static/app.js:132 | Display latency floor 0-500 ms + request time; average ~250 ms |
| `max_events` | 5,000 | _agent.py:29 | Ring size; also size of the scan under lock |
| Payload cap | 16 KiB serialized JSON | _models.py:11 | Max per event |
| `serve_forever(poll_interval)` | 0.1 s | _agent.py:96 | Only shutdown responsiveness |
| `ThreadingHTTPServer` | thread per request, daemon | _server.py:56-57 | No pool bound |
| Default bind | 127.0.0.1:8092 | _agent.py:27-28 | |

## Cost drivers
- Events/second x payload size (<=16 KiB): validate (`json.dumps`), deep copy, store.
- Poll responses: events since cursor; first poll or after a reset returns **all retained events** (up to 5,000 x up to 16 KiB = up to ~80 MB JSON, typically far less).
- Number of open browser tabs (each polls at 2 Hz and holds a thread).
- Lock hold time while serializing selection (`as_dict` building for each returned event).

## Estimate formula
Publish-side cost to producer: `t = pydantic validate + json.dumps(payload) + deepcopy x2 + lock` ~ **0.05-0.3 ms** for a 1 KB payload; ~**0.5-2 ms** for a 16 KiB payload (guess, **needs-measurement**).
Viewer lag: `t_view = U(0, 500 ms) + http_rtt (~1-5 ms localhost) + store.after (O(5000) scan ~0.3 ms + as_dict per returned event) + browser render`, i.e. ~**0.25-0.55 s typical, 0.5 s worst** (excluding huge reset responses).
Full-history fetch (reset): up to 5,000 `as_dict` + `json.dumps` of up to tens of MB: tens to hundreds of ms of GIL-bound work (guess).

## Hidden latency / improvement candidates
1. **500 ms polling floor** (app.js:132): use Server-Sent Events/long-poll to cut view lag to ~ms. Saves ~250 ms average on timeline freshness; no effect on agent latency.
2. **`after()` holds the lock and builds dicts for up to 5,000 events** (_store.py:51-67) while the event loop thread's `append` waits on the same `threading.Lock` (_store.py:41-49), i.e. a reset/initial page load can block the **asyncio event loop** (and thereby audio frames/TTS pacing) for the duration of the scan+dict build (guess 1-20 ms; GIL contention adds more during `json.dumps` in the HTTP thread). Fix: copy `list(self._events)` under the lock then build outside, or cap events per response (e.g. 500).
3. **No response size cap** (_server.py:150-174): reset returns the whole ring (<= ~80 MB worst case). Cap and paginate.
4. **Payload size check does a full `json.dumps` on every publish** (_models.py:38-52) and the store deep-copies again; combine or skip re-copy (events are already validated copies per subscriber).
5. **Thread per request, no connection reuse** (HTTP/1.0 default `BaseHTTPRequestHandler`): each poll opens a connection; negligible on localhost.
6. Producer-side: make publishes to this topic fire-and-forget on hot paths so a producer never awaits viewer bookkeeping (`runtime.publish` awaits subscribers).

## What to measure live
- Loop impact: 10 ms ticker task lateness while loading the viewer page with a full ring (5,000 events) and while 3 tabs poll; compare to a baseline without the viewer.
- Publish latency: time `await runtime.publish(WEB_EVENT_TOPIC, ...)` for 0.2/4/16 KiB payloads (in the profiler harness; topic is `telemetry="none"` so no relay scope - use `MessageMetadata.timestamp_us` vs return time).
- End-to-end viewer lag: event `timestamp_us` (the viewer exposes it, _store.py:30) vs browser receive time (`performance.timeOrigin + performance.now()`).
- Response size and time: `curl -w '%{time_total} %{size_download}' /api/events?after=0` at fill levels 100/1,000/5,000.
- Note `GET /healthz` is a cheap liveness probe (_server.py:125).

## Confidence
Constants and structure: **code-derived** (high). ms costs: **guess** pending the microbenchmarks above. The loop-blocking claim (item 2) is a code-derived hypothesis: the lock is a `threading.Lock` shared between the event-loop thread and HTTP threads, but actual stall depends on GIL scheduling.
