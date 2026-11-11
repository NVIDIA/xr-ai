<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Conversation context and compaction (agent-sdk/xr-ai-conversation/xr_ai_conversation/_context.py, _conversation.py:435-519, _context_update.py, _topics.py, eval/README.md)

## Role in a turn
Background work plus prompt sizing. Dialogue/topic context is rendered into every answer prompt (so it sets prefill cost); compaction is an extra LLM `chat()` call run as a background `asyncio.Task`, off the critical path of the CALLING turn but sharing the same LLM server with it.

## Critical-path calls
Per turn, synchronous and cheap (event-loop thread), before the answer LLM request (`_conversation.py:252-300`):
1. `reconcile_transcript` (`_context.py:164-200`): blake2b-128 digest of user+assistant text for each exchange in the front-end history window, window-containment scan `O(window^2)` worst case (`:120-126`).
2. `_schedule_dialogue_compaction` (`_conversation.py:297`, `_context.py:216-274`): sums `utf8_size` (re-encodes each pending exchange) `:239`; if > trigger and no job running, `asyncio.create_task(_compact_dialogue)` with forked Relay context.
3. `_answer_messages` (`_conversation.py:329-355`): snapshots, `_fit_section` loops (encode sizes; drops oldest raw entries until under budget, logs a warning per omission, `:540-564`), sort blocks, `_messages` builds 3 chat messages (`:567-590`).
Compaction job (background, `_context.py:276-302`): `await compact(previous, batch, summary_budget)` -> `llm.chat(messages)` (`_conversation.py:505`, one non-streaming call, no tools, no max_tokens) -> `_bounded_summary` (`:573-605`): if the summary exceeds the byte limit, ONE more `chat` (`_condense_summary`, `_conversation.py:507-519`) on `_clip(summary, payload_budget)`; if that still overshoots or is cut off, hard `_clip`. On success replaces the covered prefix, and the done-callback re-schedules immediately if still above trigger (`_context.py:465-490`) -> chained compactions back-to-back.
Incremental topic sources: `schedule_context_compaction` (`_context.py:355-412`) after every update (`_conversation.py:411-417`); `_compact_source` `:435-466`; same machinery.
Summary-mode topics: no LLM; state replaced in O(1), oversized text clipped (`_conversation.py:396-409`).

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| `dialogue_budget_bytes` | 8000 B (rendered completed dialogue per request) | `_conversation.py:50` |
| `topic_budget_bytes` | 4000 B per topic | `:51` |
| `compaction_input_budget_bytes` | 12000 B (previous summary + entries per compaction request; excludes instructions) | `:52` |
| compaction trigger | section budget // 2 (dialogue 4000 B, topic 2000 B) of uncompacted rendered text | `_conversation.py:422-426` |
| summary budget | section budget // 4 (dialogue 2000 B, topic 1000 B) | `:425` |
| `_SUMMARY_HEADER_BYTES` | 256 | `:54` |
| field clip for one exchange | dialogue budget // 4 = 2000 B per user/assistant field in prompt | `:362` |
| update clip | topic budget // 2 = 2000 B per update in prompt | `:351` |
| newest exchange kept verbatim | always (prefix = all but last) | `_context.py:241-245` |
| LLM params for compaction | `chat(messages)` only: no `max_tokens` (server default), no temperature, thinking per endpoint default (off for Nemotron presets), timeout 60 s | `_conversation.py:466, :505, :519` |
| Retries | none; a failure leaves entries for next attempt (logged error); `_TruncatedSummary` if `finish_reason == "length"` | `_conversation.py:530-537`, `_context.py:519-526` |
| Timers / polling | none (event-driven) | - |
| System prompt | 760 B core text (+ app instructions) | `prompts/core.txt` |

## Cost drivers
Dialogue length growth rate (bytes per exchange), the budgets above, number of topics (each adds up to 4000 B to every prompt and its own compaction jobs), update rate of incremental topics (each update may trigger a job), concurrency of compaction with live answers on the same LLM, eval-visible behaviour: summaries of at most budget//4 bytes (eval/README: no failed compactions, facts kept).

## Estimate formula
Prompt size per answer: `P_bytes <= 760 + len(app_instructions) + ~330 (context header) + min(dialogue ~ summary<=2000 + raw<=8000-ish) + sum_topics(<=4000) + current_request + ~60 (time line)`; tokens ~ bytes/4 (English; guess; ~bytes/3 for CJK/code). With 0 topics: <= ~2.4 K tokens + instructions; each topic +~1 K tokens. `t_prefill_ctx = P_tokens / prefill_tps` (affects every turn; prefix caching helps only for the stable system prompt + earlier unchanged blocks, and dialogue blocks shift each turn).
Compaction call: `t_compact = t_prefill(<=12 KB payload + ~1.1 KB instruction ~ 3.3 K tokens) + n_out(<=~500 tokens, 2000 B summary)/decode_tps (+ optional condense call of similar size)`; e.g. at 50 tps decode 500 tokens = 10 s of GPU decode in the background (guess). Frequency: about once per `4000 B / bytes_per_exchange` exchanges (~every 5-8 turns at ~500-800 B per exchange, guess).
Interference: if compaction overlaps an answer, answer latency rises by the share of server throughput taken (needs measurement).

## Hidden latency / improvement candidates
* Compaction is scheduled BEFORE the answer request is sent (`_conversation.py:297` precedes `:300`): the compaction request enters the server queue first/simultaneously, so the user-visible answer's TTFT can be delayed by up to a compaction prefill (3 K tokens: 100-400 ms on a 30B-class model, guess) when a compaction fires on that turn. Scheduling it after the answer finishes (the `_commit_exchange` path already schedules at `:327`) or at low priority would remove that contention; the call at `:297` appears redundant with `:327`, except for history reconciled from the front end.
* No `max_tokens` on compaction: a verbose model can run past the 2000 B summary budget (then costs a second `condense` LLM call, `_context.py:600-605`); setting `max_tokens ~= summary_budget/3` bounds decode and avoids the second call (saves up to one whole extra call, several seconds of background GPU).
* Same LLM endpoint as the live answer: using a separate (smaller) model for compaction removes interference.
* Per-turn digest of the whole FE history window (blake2b, `_context.py:106-107`): O(window) microseconds; fine.
* `_fit_section` logs a warning and silently omits old context from the prompt when compaction lags (keeps latency bounded; behaviour loss only).
* Chained compactions (done-callback re-schedule) can run several LLM calls back-to-back when many entries accumulated (e.g. after a long pause or large topic dump).

## What to measure live
Run a scripted session through `Conversation` (public API) with `llm` wrapped by a timing proxy: record per-call (answer vs compaction) start/end, prompt_tokens (`ChatResponse.raw["usage"]` for chat calls), n_out; count compactions per N turns; measure answer TTFT in turns where a compaction fires vs not. `eval/compaction_eval.py` (`--samples`, `--preset nemotron_omni`) already drives long sessions and can be used as the workload.

## Confidence
Triggers/budgets/ordering: code-derived. Bytes-per-token ratio, prefill/decode times, interference magnitude: needs measurement/guess.
