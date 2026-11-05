<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay wrapping overhead (agent-sdk/xr-ai-tools/xr_ai_tools/tools.py, async_tools.py, vision.py, _vision.py, _relay.py; agent-sdk/xr-ai-conversation/xr_ai_conversation/_context.py:261,398)

## Role in a turn
Every tool execution and every VLM call passes through `nemo_relay` (Rust-backed native lifecycle/scope/telemetry layer). It is a fixed per-call tax added on the critical path of each tool hop; middleware/subscribers attached at runtime (not in this package) can make it much larger.

## Critical-path calls
Per tool call (`Tool.execute`, `tools.py:75-82`): `typed.tool_execute(name, request, _execute_handler, request_codec, result_codec)`:
1. `PydanticCodec.to_json(request)` (serialize request for the Relay event), scope push (tool scope), start-event emission to subscribers.
2. `_execute_handler` -> handler (`tools.py:108-112`), wrapped into `ToolExecutionResult`.
3. `PydanticCodec(result_model)` serialize result for the end event; scope pop; subscribers.
Plus `Tool.invoke`: `json.loads` + `model_validate` (`tools.py:87-96`) and result render `model_dump_json` (`:185-188`). The request model is thus validated once on invoke and then (probably) again by the codec on the Relay side (not visible in this repo: native code).
Per VLM call (`vision.py:128-148`): `image_sanitizer()` register/deregister (2 native calls) `_vision.py:30-50`; `llm.execute` with `OpenAIChatCodec` on request and response (encode `relay_request` dict -> LLMRequest -> codec; response decoded) ; `_sanitize_images` `copy.deepcopy(request.content)` (URIs only, no bytes, so small) `_vision.py:191`; LLM start/end events.
Per streaming tool chunk (`async_tools.py:41-107`): producer task created with `nemo_relay.fork_asyncio_context()` (`:57`) and a Tool scope (`:95`); per chunk `model_validate(chunk)` + `queue.put` + consumer `create_task(queue.get())` + `asyncio.wait` (`:61-66`). VLM stream additionally runs two Python callbacks per chunk (`vision.py:182`, `:189`).
Conversation compaction tasks: `asyncio.create_task(..., context=nemo_relay.fork_asyncio_context())` (`_context.py:261, :398`) - context copy per job, trivial.

## Fixed waits & timers
None in this package: no sleeps, no timeouts, no batching intervals around Relay. (If a Relay exporter/subscriber is configured by the application it may flush on its own timer - outside the scanned code.)

## Cost drivers
#tool calls per turn; size of request/result JSON (RAG results with 20 chunks, 256 frame refs in `SampleFramesResult` are serialized twice: codec for events and `model_dump_json` for the model); VLM stream chunk count; active Relay subscribers/exporters (unknown); sanitizer deepcopy size (small since images are URIs).

## Estimate formula
`t_relay_tool = t_codec_req + t_scope_push/pop + t_events + t_codec_res`: code-only guess 50-300 us per tool call with no subscribers (pydantic dump/validate 10-50 us each + native calls). `t_relay_vlm = 2 * t_sanitizer_native + t_deepcopy(small) + t_codec_req/resp` ~0.1-0.5 ms. Streaming: ~50-200 us per chunk. Needs measurement; with exporters attached could be 1-10 ms (guess).

## Hidden latency / improvement candidates
* Large tool results (SampleFramesResult up to 256 frames, RAG top_k 20) pay JSON serialization twice (event + model-facing); bound `frame_budget` / `top_k`. Saving: <1 ms typical, a few ms for 256 frames.
* Per-chunk AsyncTool machinery (task creation per chunk) can be replaced with a persistent getter; saving ~50 us/chunk = ~1 ms per 20 tokens: negligible against decode time.
* Image sanitizer registered per call; could be registered once per scope: saves two native calls (~10-50 us).
* No evidence of blocking IO in the wrapper itself; the risk is subscribers (unknown).

## What to measure live
Baseline vs wrapped: for a trivial tool (e.g. `TrackingTools` spatial or `ListDocuments`), time `tool.handler(request)` directly vs `await tool.execute(request)` vs `await tool.invoke(json)` (all public) to isolate Relay + validation cost; repeat with and without any Relay subscriber/exporter configured by the host app; for streaming, time per-chunk of `AsyncTool.stream` vs the raw handler iterator.

## Confidence
Which calls are wrapped: code-derived. Cost of the native layer: not readable from this repo, needs measurement; microsecond figures are guesses.
