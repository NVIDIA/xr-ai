<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Agent runtime pub/sub (agent-sdk/xr-ai-runtime/xr_ai_runtime/runtime.py, events.py, agent.py)

## Role in a turn
In-process typed event bus between agents (voice agent -> application agent -> voice output, web-events, aggregation). Every hop between agents in a turn is one `publish`. It is on the critical path for the query hand-off (`UserQuery`), every streamed response chunk (`voice.output`, per-token when streaming), and optional UI/trace events. It adds no network/IPC hop; cost is Python overhead plus whatever the subscribers do, because **publish awaits all subscribers**.

## Critical-path calls
Per `runtime.publish` / `ctx.publish` (runtime.py:145-160, 184-238):
1. `_ensure_running`, `_register_topic` (dict lookups).
2. `topic.validate(message)`: pydantic `model_validate` (events.py:38-41). Serial.
3. `_metadata`: `uuid4().hex`, `time.time_ns()` (runtime.py:306-326).
4. If `topic.telemetry == "full"` (default): open nemo_relay scope `publish:<topic>` and `value.model_dump(mode="json")` for the scope input (runtime.py:203-209); each delivery also opens scope `agent:<name>` with another `model_dump` (runtime.py:253-263). Topics declared `telemetry="none"` skip this (voice.output, voice.contribution, web-events.event).
5. For each subscriber: `value.model_copy(deep=True)` + `asyncio.create_task(_deliver, context=nemo_relay.fork_asyncio_context())` (runtime.py:216-231). Subscribers run **in parallel** with each other.
6. `await asyncio.gather(*deliveries, return_exceptions=True)` (runtime.py:233): publish returns only when **every** subscriber coroutine returned; errors aggregate into `BaseExceptionGroup`. A subscriber that runs a whole LLM turn inline makes the publisher wait for the entire turn.
`ctx.publish` inside a subscriber nests another full publish; `parent_message_id`/`correlation_id` propagate (runtime.py:66-82).

## Fixed waits & timers
None: no sleeps, polling, queues, timeouts, or batching in the runtime. Bounded structures: none (deliveries are tracked in a per-agent set; unbounded concurrent tasks). `start()` only flips a flag (runtime.py:136-143). `stop()` cancels in-flight deliveries and awaits them (runtime.py:162-175). Topic message sizes are unbounded (pydantic only).

## Cost drivers
- Message size: validate + deep copy per subscriber + (telemetry full) two `model_dump(mode="json")`s. Large payloads (images as base64, long text) scale linearly per subscriber.
- #subscribers per topic: one task + one deep copy each.
- Publish frequency: streaming outputs publish per chunk/token (VoiceAgent chunks), so overhead is multiplied by token rate; telemetry `full` on high-rate topics would add a nemo_relay scope per publish and per delivery.
- Subscriber duration: directly added to the publisher's await.
- Concurrency: nothing serializes or limits concurrent deliveries to the same subscriber; per-participant ordering must be provided by the subscriber (as `VoiceAggregationAgent` does with a per-pid queue).

## Estimate formula
`t_publish = validate(msg) + n_sub * (deepcopy(msg) + task_spawn) + [telemetry: 1 + n_sub scopes + dumps] + max_i(t_sub_i)`

Overhead excluding subscriber work, small message (<1 KB), 1 subscriber, `telemetry="none"`: ~**50-150 us** (guess; pydantic v2 validate ~5-20 us, deepcopy ~10-30 us, uuid4 ~2 us, task create ~10-20 us, gather ~10-20 us). With `telemetry="full"`: add ~50-300 us (scope enter/exit + dumps; depends on nemo_relay, **needs measurement**). Large (100 KB) payload: +0.1-1 ms per subscriber (copy + dump).
Per-turn count: typically 2-5 publishes on the critical path (UserQuery, 1..N response chunks, optional interrupted/participant events) => **< 1-2 ms** total overhead unless a subscriber blocks.

## Hidden latency / improvement candidates
1. **Publish awaits subscriber completion** (runtime.py:233): publishing `UserQuery` blocks the caller (`_run_query` in io.py:203) for the entire application turn; for per-token output publishing, a slow subscriber (e.g. `VoiceAgent.output` blocked on `_ResponseStream.send` with a full 32-chunk queue, runtime.py:85-102) back-pressures the LLM streaming loop. Designing apps with fire-and-forget (`create_task(publish)`) or a bounded per-subscriber queue would decouple producer speed from consumer speed. Saving: up to the slow subscriber's tail latency per chunk.
2. **Deep copy per subscriber + double model_dump with telemetry full** (runtime.py:208, 222, 257): for large image/text messages and several subscribers this is O(size x subscribers). Use `telemetry="none"` on high-rate or large topics (as voice topics do), or share frozen models. Saving: ~0.1-1 ms per publish for big payloads.
3. **No ordering guarantee between consecutive publishes with concurrent subscribers**: two publishes from the same producer run their subscribers as independent tasks; ordering depends on the producer awaiting publish (it does, so sequential per producer). Not a latency issue unless the producer fires without awaiting.
4. **Unbounded task fan-out** (runtime.py:217-231): bursts (e.g. 100 tok/s x N subscribers) create tasks without limit; each task copies context (`fork_asyncio_context`). CPU only.
5. `inspect.getmembers` subscription discovery at registration only (startup).

## What to measure live
- `MessageMetadata.timestamp_us` on every delivery (runtime.py:325) = publish time; `message_id`, `parent_message_id`, `correlation_id` chain a whole turn (runtime.py:318-324). A profiler agent subscribing to every topic gets the entire event timeline for free (note topics with `telemetry="none"` still deliver to subscribers).
- Subscriber duration: nemo_relay scopes `publish:<topic>` and `agent:<name>` (runtime.py:204-263) give publish duration (= slowest subscriber) and each subscriber's duration, for topics with `telemetry="full"`.
- Pure runtime overhead: microbenchmark `publish` with a no-op subscriber (n_sub = 1,3; payload 0.2 KB / 20 KB; telemetry full/none) in the profiler harness - no external services needed.
- Wrap `AgentRuntime._publish` in the harness for per-topic latency histograms when a scope is unavailable.

## Confidence
Behavior and absence of waits: **code-derived** (high). Microsecond overheads: **guess** until the microbenchmark is run; nemo_relay scope cost: **needs-measurement**.
