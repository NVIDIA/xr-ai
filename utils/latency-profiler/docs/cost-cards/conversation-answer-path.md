<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Conversation answer path (agent-sdk/xr-ai-conversation/xr_ai_conversation/_conversation.py:142-215, :252-355, :567-590, _topics.py)

## Role in a turn
The text-LLM answer stage of the voice/agent pipeline: build prompt from context, stream tokens from the LLM, publish each chunk to the output topic (typically feeding TTS). It determines time-to-first-text (TTFT) of the spoken response.

## Critical-path calls
1. Router calls `answer(turn, started=...)` (`:142`). `_stream_response` generator created.
2. `reconcile_transcript` + `_schedule_dialogue_compaction` (sync) `:289-297` (see compaction card).
3. `_answer_messages` -> 3 messages: system (core prompt + instructions), user (context blob), user (request) `:329-355, :586-590`.
4. `self._llm.stream_events(messages)` (`:300`): ONE streaming request, no `tools`, no `max_tokens`, no `temperature`, no `enable_thinking` (endpoint default: thinking off for Nemotron presets). Only `kind == "text"` events are forwarded `:302-305`; reasoning events ignored.
5. First nonblank chunk: `await started()` (router hook, may interrupt older audio) `:171-174` BEFORE the first publish; then `await runtime.publish(ResponseOutput(...))` per chunk, serially `:180-188` (the consumer loop is lockstep with the LLM stream: next SSE line is not read until publish returns).
6. After stream end: final empty frame publish `:201-209`, `_commit_exchange` `:213`, compaction rescheduled `:327`.

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| LLM request timeout | 60 s (read timeout resets per chunk in streaming) | LLM client |
| `max_tokens` | none | `:300` |
| Budgets (dialogue 8000 B, topic 4000 B) | see compaction card | `:50-52` |
| No sleeps, polling or debounce in answer path | - | - |
| Cancellation checks | `task.cancelling()` per chunk | `:177-178` |

## Cost drivers
Prompt bytes (context budgets), LLM TTFT and decode rate, per-chunk publish cost (runtime bus), `started()` latency (router), number of topics.

## Estimate formula
`t_ttft_text = t_prompt_build(<1 ms) + t_started(measure) + t_llm_ttft(P_tokens) + t_publish_first(measure)`; `t_answer = t_ttft_text + (n_chunks-1) * max(1/decode_tps, t_publish_chunk) + t_final_publish`. Plug in `t_llm_ttft` from models-openai-compat-llm.md, `t_publish_chunk` (runtime publish; ~50-300 us guess). Best/typical/worst client-only overhead: 1 / 3 / 10 ms.

## Hidden latency / improvement candidates
* Per-chunk `await publish` in the stream loop couples LLM read rate to bus latency; at 50 tok/s (20 ms/token) 0.3 ms publish is 1.5 %, negligible unless publish backs up (bounded queues). Buffering whole-clause chunks for TTS is the voice package's job.
* `started()` awaited on the first chunk; if it takes tens of ms (router state/interrupt handling) it directly adds to TTFT; measure.
* The history/context blob is rebuilt and sent every turn (no server-side session): prefix caching only covers the stable part (system prompt + earlier blocks), and the dialogue summary changes after each compaction (cache miss for the following turn).
* The system prompt says "No camera or visual-inspection tool is available" (`prompts/core.txt`), so this path never routes to vision by itself; vision turns go through the tool loop (tools-tool-calling-loop.md).
* `stream_events` is used instead of `stream`, so the tool-call hold logic and strict `[DONE]`/finish_reason checks apply: a connection that drops before `[DONE]` raises after partial audio was already published.

## What to measure live
TTFT from `answer()` call to first `runtime.publish` (public `Conversation` API with a stub runtime), split into prompt build, `started()`, first LLM `text` event; chunk interval distribution (check publish does not stall the stream); vary context size (0/2/8/12 KB) to get `d(t_ttft)/d(prompt_bytes)`.

## Confidence
Control flow: code-derived. All times: needs measurement.
