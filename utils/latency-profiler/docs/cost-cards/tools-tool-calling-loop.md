<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tool-calling loop (agent-sdk/xr-ai-tools/xr_ai_tools/tool_calling.py, tools.py, async_tools.py)

## Role in a turn
`run_tool_loop` is the agent-turn driver for native tools: LLM call -> execute tool calls -> LLM call ... until a final answer. Every hop is on the critical path; total latency = sum of LLM calls + tool executions, strictly serial.

## Critical-path calls
1. `tool_definitions(tools)` once per loop: `model_json_schema()` per tool (`tool_calling.py:113-130`, called `:175`). Pydantic schema generation is cached per class after first use? (pydantic caches core schema, but `model_json_schema()` regenerates JSON each call: ~50-300 us per tool, guess).
2. Loop up to `max_iterations` (default 4): `response = await call_model(tuple(transcript), definitions)` `:180` (caller-owned: model params, retries, stream vs chat; the SDK samples call `llm.chat(transcript, tools=..., max_tokens=2048, temperature=0.0)` non-streaming, e.g. `agent-samples/xr-render-demo/worker/.../agents/vision/agent.py:124`).
3. Safety checks (finish_reason length/tool_calls, unique IDs, direct-return rules, budget) `:188-248` - microseconds.
4. `for call in calls: result = await handle_tool_call(call, tools)` `:271-272` - tool calls in one model batch run SEQUENTIALLY in emitted order (docstring `:166`), not `asyncio.gather`.
5. `handle_tool_call` -> `tool.invoke(arguments)` (`tool_calling.py:141`, `tools.py:84-100`): `json.loads`, pydantic `model_validate`, `typed.tool_execute` (Relay), render result (`model_dump_json`).
6. Transcript grows by assistant message + one tool message per call; next `call_model` re-sends the entire transcript and tool definitions.
7. `AsyncTool.stream` (streaming tools) `async_tools.py:41-88`: producer task + `Queue(maxsize=1)`; per chunk a new `create_task(queue.get())` and `asyncio.wait`; first chunk waits on producer start.

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| `max_iterations` | 4 (render-demo samples use 12: `supervisor.py:244`) | `tool_calling.py:159` |
| `max_tool_calls` | 16 total per turn | `tool_calling.py:160` |
| Model `max_tokens` / `temperature` | decided by the callback; sample = 2048 / 0.0 | sample agents |
| Per-tool timeout | none in loop; each tool's own (RPC 10/30 s, VLM 60 s) | - |
| Retry | none ("The callback owns ... retry policy" `:164`) | - |
| `AsyncTool` queue size | 1 (lockstep backpressure) | `async_tools.py:53` |
| Iteration exhaustion | raises `ToolLoopIterationLimitError` | `:290-295` |

## Cost drivers
#iterations (each = full LLM request with growing transcript), tool definitions size (re-sent every iteration), tool latency (serial), tool-result size in tokens (images are refs, but VLM text answers and RAG chunks are text), batches with several calls (serial, not parallel).

## Estimate formula
`t_turn = sum_{i=1..K} [ t_llm(P_i, n_out_i) ] + sum_{calls} t_tool_j + K * t_loop_overhead(<0.5 ms)`, with `P_i = P_0 + sum of prior assistant/tool message tokens`, K = iterations (final-answer hop included). For a typical vision turn K=3: LLM(tool call) -> `get_current_frame` -> LLM(tool call) -> `query_image` -> LLM(answer). So `t = 3*t_llm + t_frame + t_vlm`. Plug in: `t_llm` per hop (prefill of P_i + decode of the short tool-call args ~30-60 tokens), tool times from other cards. Needs measurement; loop overhead itself is code-derived negligible.

## Hidden latency / improvement candidates
* Sequential execution of independent calls in one batch (`:271-272`): `asyncio.gather` would save min(sum, ...) - e.g. two parallel RAG+tracking calls save one RPC time (tens of ms-s depending on tools); caveat: ordering/direct-return semantics.
* Non-streaming `chat` per hop in samples: tool-call hops get no benefit from streaming (tool calls held to [DONE] anyway, see streaming doc), but the FINAL answer hop could stream to TTS: saves (n_out/decode_tps) of perceived latency, e.g. 100-300 tokens -> 1.5-6 s at 50 tps. The loop contract (callback returns `ChatResponse`) forces whole-response return, so the final hop is not streamable through `run_tool_loop`.
* Hop fusion: tool chains like `get_current_frame` -> `query_image` cost 2 LLM hops because the LLM must pass the `xr-image://` reference. A composed tool (`capture_and_ask`) or pre-seeding the current frame removes one LLM call (saves ~0.5-2 s per avoided hop).
* Tool definitions + transcript are re-prefilled each hop; server prefix caching (identical prefix) mitigates only if the system prompt+tools block and earlier messages stay byte-identical (the loop appends only, so prefix is stable; good).
* `return_direct` (`tools.py:43`) skips the final LLM hop; the vision tools do not set it.
* `max_tokens=2048` with thinking-off is a non-binding cap normally; with runaway output it bounds worst case at 2048/decode_tps.

## What to measure live
Per-iteration: wrap the `call_model` callback and `handle_tool_call` with timers (both public); record `ToolLoopResult.iterations`, `tool_calls`. Compare `run_tool_loop` total vs sum of parts to get loop overhead (expected ~0). Sweep tool count (5/20/40 definitions) to fit `t_prefill(tool_defs_tokens)`.

## Confidence
Control flow: code-derived. Hop times: needs measurement. Schema-generation microseconds: guess.
