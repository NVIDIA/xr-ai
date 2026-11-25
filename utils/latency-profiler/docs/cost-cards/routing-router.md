<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# AgentRouter (agent-sdk/xr-ai-routing/xr_ai_routing/_router.py, _contracts.py)

## Role in a turn
Sits between the final transcript and the destination agent. Picks a destination (one `DecisionService.decide` call
when more than one candidate), asks the destination for admission, then runs `answer`. A side task may speak an
acknowledgment while a slow answer is pending. Everything before `destination.answer()` is pure overhead that delays
first output, so routing-path count matters more than any single call.

## Critical-path calls
Entry `_handle` (_router.py:684). Paths, in order of evaluation:
1. Focus check (`_router.py:687-703`): if a destination holds focus and `routing_aid().eligible`, candidates=[focused] and
   attempt 0 **skips `_select`** (line 707). No LLM call. If focus is ineligible: `release()` awaited, focus cleared,
   falls to multi-route.
2. `_select` (`:1154`): 0 candidates -> None (clarify); 1 candidate -> returns it without a model call (`:1157-1165`);
   >1 candidates -> **one `DecisionService.decide`** with state = {participant, time, query, completed_history (bounded by
   `_DEFAULT_CONTEXT_BUDGET_BYTES`=12000, `:41`), destinations[]} and criteria incl. a `clarify` label (`:1167-1215`).
   Awaited serially before admit.
3. `destination.admit()` (`:715`): destination-defined cost (may itself call an LLM). Awaited serially.
4. Admission outcomes (`:718-777`):
   - `accept`: continue to answer.
   - `release`: `release()` awaited, optional text emitted (TTS), **loop for a 2nd attempt** over remaining candidates
     -> 2nd `_select` + 2nd `admit`.
   - `clarify`: emit text, return. No answer.
   - `decline`: remove from candidates, `release()`, **2nd attempt**; after attempts exhausted falls back to the default
     destination via standalone `answer()` without `admit()` (`:741-777`).
5. `admission.preamble` emitted before answer (`:784`); preamble disables the acknowledgment (`:854`).
6. Acknowledgment race (`:833-905`, `_acknowledge_if_needed :985`), only if `acknowledgment_llm` and
   `acknowledgment_delay_secs` configured and no preamble: runs concurrently with `destination.answer()`:
   - `delay_task = sleep(delay_secs)` and `prediction_task = _predict_slow_answer` (one more `decide()`, `:1102`) race.
   - If prediction says `slow` it proceeds immediately, else waits for the delay timer. If the answer started meanwhile
     it returns silently.
   - Then `acknowledgment_llm.chat(max_tokens=64, enable_thinking=False)` (non-streaming) then `emit()` (-> TTS).
   - `response_started()` (`:798`): the first call cancels an uncommitted ack; if an ack is already committed, the answer
     **waits for `acknowledgment_delivered`** (`:821-823`), i.e. a mispredicted ack delays the real answer by the ack's
     delivery time.
7. `destination.answer(turn, started=...)` (`:866`).

## Fixed waits & timers
- `_DEFAULT_CONTEXT_BUDGET_BYTES = 12_000` (`:41`): bounds history in the routing prompt (~3k tokens worst case).
- `_ACKNOWLEDGMENT_MAX_TOKENS = 64` (`:44`).
- `acknowledgment_delay_secs`: constructor arg (None disables); floor on when an un-predicted-slow ack can fire.
- Routing attempts bounded to 2 (`for attempt in range(2)`, `:706`).
- No explicit timeouts on `decide()` in the router (service-level endpoint timeout applies).

## Cost drivers
- Candidate count (1 => no model call).
- History bytes in the select prompt (<=12KB) and destination description lengths.
- Destination `admit()` latency (unbounded by router).
- Number of attempts (misroute/decline multiplies select+admit).
- Ack: delay_secs, `decide()` latency, ack LLM latency (64 tok), TTS first chunk of ack, acknowledgment_delivered wait.
- Concurrency: select/predict/ack/answer all hit the same GPU services; ack+predict add 2 extra model calls per turn
  competing with the answer for `max-num-seqs`.

## Estimate formula
- Focused: `t_route = admit`.
- Single candidate: `t_route = admit`.
- Multi-route happy path: `t_route = decide(ctx_tokens) + admit`.
- Caught misroute (one decline/release): `t_route = 2*decide + 2*admit + release (+ tts(release_text) if text)`.
- Exhausted declines -> default: `~2*decide + 2*admit + 2*release + default_answer_start`.
- Clarify: `decide + tts_first_chunk(clarify_text)`; user gets no answer.
- Ack (when predicted slow, answer slower than ack): first audio = `min(answer_first_audio, max(decide_slow, 0) + ack_chat64 + tts_first)`.
  When the ack wins and answer arrives while ack still speaking: answer start = max(answer_ready, ack_delivered).
  Cost of a wrongly-committed ack: `+ack_tts_duration` on the real answer.
Terms needed: decide(ctx), chat64, tts_first_chunk, admit (per destination), release. Best/typical/worst:
needs measurement.

## Hidden latency / improvement candidates
- Select, admit serial: when a destination's `admit()` is itself an LLM call, speculative parallel admit for the top-2 candidates
  would save one decide+admit on a miss (cost: extra GPU load). Saving ~1 decide on misroute paths.
- Focused fast path skips select but still pays admit; destinations that need no admission check should return fast.
- Ack race adds two model calls per slow turn (predict + chat), both competing with the answer on shared vLLM seqs.
- `acknowledgment_delivered.wait()` couples answer start to ack TTS duration; a mispredicted ack can add ~1-3 s.
- History prompt grows to 12KB for the router even for trivial queries; trimming relevance would cut decide() prefill.

## What to measure live
- `decide()` latency vs state size (history bytes 0..12KB, #destinations 2..8), cold vs prefix-warm.
- Router path matrix with synthetic `RoutableAgent`s (controllable admit/answer latency, dispositions) driving every path above;
  timestamps from nemo_relay scopes already emitted (`conversation.destination-route`, `conversation.response-started`,
  `conversation.intent-acknowledgment*`) and from wrapper destinations -- no SDK modification needed.
- Ack: predict latency, ack chat64 latency, penalty when ack committed then answer ready.
- Router accuracy vs confidence on a labeled utterance set (for expected misroute cost).

## Confidence
Call structure: code-derived (read in full for `_handle`, `_select`, `_acknowledge_if_needed`). All numbers: needs-measurement.
