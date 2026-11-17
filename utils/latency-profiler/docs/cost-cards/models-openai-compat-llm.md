<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# OpenAICompatLLM chat / stream / stream_events (agent-sdk/xr-ai-models/xr_ai_models/_openai_compat.py, _thinking.py, _factory.py, _config.py, _protocols.py, presets/)

## Role in a turn
The text LLM call: the largest single term on the critical path of an agent turn (prefill + decode). Called once per answer (Conversation, streaming), once per tool-loop iteration (non-streaming `chat` in the samples), plus background compaction `chat` calls.

## Critical-path calls
1. `_build_payload` (sync, event-loop thread): `[_msg_to_openai(m) ...]`, tools to_openai, extras, `apply_thinking` - :313-343. Microseconds to ~1 ms for large prompts.
2. `_request_headers` (:85) - trivial.
3. HTTP POST to `{base_url}/v1/chat/completions` (:293). `chat`: `await client.post` (:373) serial; `stream`/`stream_events`: `async with client.stream("POST")` (:406/:464) then per-line loop. One request per call; no fan-out, no `gather`.
4. Response parse: `resp.json()` + `_parse_chat_response` (:377) for chat; per-line `json.loads` for streams.
5. `stream_events` only: after `[DONE]` validates tool calls and emits them (:544-580).

## Fixed waits & timers
| Constant | Default | Where | Effect |
|---|---|---|---|
| httpx timeout (connect/read/write/pool each) | 60.0 s LLM/VLM/embedding; 30.0 s STT/TTS/OCR-default | `OpenAICompatLLM.__init__(timeout=60.0)` :287; `EndpointSpec.timeout` _config.py:89; `_timeout()` _config.py:908-909 (60 for llm/vlm/embedding else 30) | upper bound only; in `chat` covers whole generation (no bytes until done), in streams resets per chunk |
| per-call `timeout=` | None -> client default | :371, :404, :456 | replaces all four phases |
| `max_tokens` | None -> field NOT sent (:332) -> server default (vLLM: remaining context length) | `_build_payload` | no client cap; SDK samples pass 2048 (`agent-samples/xr-render-demo/.../vision/agent.py:124`), router ack 64 (`xr_ai_routing/_router.py:44` `_ACKNOWLEDGMENT_MAX_TOKENS`, used :1042), Conversation passes none (`_conversation.py:300`) |
| `temperature` | None -> not sent (:334) -> server default (vLLM: 1.0 unless generation_config.json overrides) | :334 | samples use 0.0 for tool loops |
| thinking switch | Nemotron3-Nano, Nemotron-Omni, Cosmos VLM: `chat_template`, default False -> `chat_template_kwargs:{enable_thinking:false}` sent on EVERY request; Cosmos3-Nano-Reasoner: `control:none` -> nothing sent | presets/nemotron3_nano.py:21, nemotron_omni.py:19, cosmos_vlm.py:24, cosmos3_nano_reasoner.py:19; `apply_thinking` _thinking.py:136-172 | thinking off avoids hidden reasoning decode |
| `thinking_budget` | None -> not sent | _thinking.py:167-168 (`thinking_token_budget`, only if thinking not off and control in None/chat_template) | caps reasoning tokens |
| health probe timeout | 3.0 s | :106 | not on turn path |
| retries / backoff | NONE anywhere in the SDK client (`OCRRequestError` doc: "not retried automatically"); `resp.raise_for_status()` raises straight to caller | :376 | a transient 503 costs full retry by caller |
| connection pool | httpx defaults (library, unset in repo): 100 max conns, 20 keepalive, keepalive_expiry 5 s, HTTP/1.1 | :310 | a gap >5 s between calls to the same server closes the idle socket -> new TCP (+TLS if https) handshake on next call: ~0.1-1 ms localhost, 10-100+ ms remote/TLS |
| `trust_env=False` | proxies/env ignored | :310 | no proxy latency |
| streaming `stream_options` | not sent | - | no usage chunk |
| `logprobs`, `top_p`, `n`, `seed` | never sent by LLM client | - | - |

## Cost drivers
* Prompt tokens (prefill): Conversation prompt is bounded by byte budgets (see conversation-context-compaction.md; roughly 2.3k tokens max with no topics, +~1k tokens per topic section at defaults), core prompt 760 B (`prompts/core.txt`). Tool definitions are re-sent in every request of the tool loop (JSON schema from pydantic `model_json_schema()`, `tool_calling.py:127`), e.g. 20 tools ~ 3-6k tokens.
* Output tokens (decode): unbounded unless caller sets `max_tokens`.
* Thinking state (reasoning tokens decoded but not shown).
* Number of HTTP calls per turn (tool hops + compaction + routing ack).
* Prefix caching on the server: `_messages()` keeps system prompt identical prefix (`_conversation.py:567-590`); the `Current time` line is last in the context message so the prefix up to it is cacheable; the dialogue blocks change every turn so prefix hit covers only system + earlier unchanged blocks.

## Estimate formula
Non-streaming: `t_chat = t_build + t_net_rtt + t_queue + t_prefill(P_uncached) + n_out / decode_tps + t_json(n_out) + t_close`
Streaming first-text: `t_ttft_text = t_build + t_net_rtt + t_queue + t_prefill(P_uncached) + n_reason / decode_tps + 1/decode_tps`; completion = `t_ttft_text + (n_out-1)/decode_tps + n_chunks * c_chunk`.
Measured terms to plug in: `t_prefill(P)` per model/GPU (tokens/s prefill), `decode_tps`, `t_queue` under concurrency (vLLM batch contention with compaction, VLM and other agents sharing the GPU), `t_net_rtt`, `c_chunk` (~15-40 us guess).
From code constants alone: client-side overhead best ~0.5 ms / typical 1-3 ms / worst (50 K-token prompt or multi-MB multimodal payload) ~10-30 ms. Server-side term: needs measurement.

## Hidden latency / improvement candidates
* No `max_tokens` on the Conversation answer path (`_conversation.py:300`) and on compaction calls (`:466, :505, :519`): a runaway answer or summary decodes to the server context limit; the server-side default may be thousands of tokens. A cap of ~256 on spoken answers could save 100s of ms-seconds on tail cases (needs measurement).
* Compaction `chat()` uses the same LLM endpoint as the live answer and is scheduled BEFORE the answer starts (`_conversation.py:297` `_schedule_dialogue_compaction` -> `create_task`), so a compaction request (prompt up to ~12 KB ~ 3 K tokens + decode of up to 2000-byte summary ~ 500 tokens) is in the vLLM queue simultaneously with the user-facing request; expect it to slow answer TTFT/decode by the batch interference share (0-30 % guess).
* `stream_events` withholds tool calls until `[DONE]` (:544-580): harmless for latency (args can't be executed earlier anyway without risk) but means tool dispatch cannot overlap with the end of generation; stream+tool gives no gain over `chat`.
* Thinking flag is always sent as `false` for Nemotron presets: ensures no reasoning cost, but per-call `enable_thinking=True` + `thinking_budget` is the only latency/quality knob; VLM wrapper cannot toggle per call (`ask_images` has no `enable_thinking`, :699-709).
* Idle keep-alive 5 s: for voice agents with >5 s between LLM calls, the first call each time pays a new connection. Pass a shared `httpx.AsyncClient(limits=httpx.Limits(keepalive_expiry=...))` via `client=` (constructor param :290) to avoid. Saving: 0.1-1 ms localhost; 10-100 ms remote.
* No client retry means one failure = caller-level retry; none charged on happy path.
* `_msg_to_openai` per call rebuilds the whole message list; negligible.

## What to measure live
* `OpenAICompatLLM.chat` vs `stream_events` for the same messages: wall time, TTFT (first `text` event), first `reasoning` event, n_out (use `ChatResponse.raw["usage"]`), per prompt length sweep (0.5k/2k/8k/32k tokens) to fit `t_prefill(P)`; then with a concurrent second request to quantify contention.
* `max_tokens` sweep to fit `decode_tps`.
* Idle > 5 s then call vs back-to-back (keepalive expiry penalty).
* API entry points: `xr_ai_models.make_llm(ModelsConfig..., name)` then `await llm.chat(...)` / `llm.stream_events(...)`; `ChatResponse.raw["usage"]`, `["timings"]` if vLLM returns it. No SDK code modification required; wrap with `time.perf_counter()`.
* Compare `enable_thinking=None/True/False` and `thinking_budget`.

## Confidence
Wire behaviour and constants: code-derived. httpx default pool/keepalive numbers: library knowledge (not in repo; verify with the installed httpx version). All millisecond figures for prefill/decode/queueing: needs measurement; client-side microsecond figures: guess.
