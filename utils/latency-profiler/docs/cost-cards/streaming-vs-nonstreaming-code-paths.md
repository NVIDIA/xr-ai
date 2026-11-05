<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Streaming vs non-streaming code paths (xr_ai_models/_openai_compat.py and callers)

All line numbers: `agent-sdk/xr-ai-models/xr_ai_models/_openai_compat.py` unless stated.

## 1. What is on the wire

| API | HTTP | Request body differences | Response read |
|---|---|---|---|
| `OpenAICompatLLM.chat` (:345) | `client.post(...)` (:373) | no `stream` key (:328 only adds it when True) | whole body buffered by httpx, then `resp.json()` (:377), `_parse_chat_response` (:149) |
| `OpenAICompatLLM.stream` (:379) | `client.stream("POST")` (:406) | `"stream": true`; no `stream_options` (so no usage chunk) | `resp.aiter_lines()` (:411), SSE `data: ` lines, `json.loads` per line (:418) |
| `OpenAICompatLLM.stream_events` (:426) | same as `stream` (:464) | identical payload to `stream` | same line loop but stricter parse (:469-543) |
| `OpenAICompatVLM.ask_image(s)` (:676/:699) | `_llm.chat` (:715) | message content = N `image_url` parts (data URLs) + text part | as `chat` |
| `OpenAICompatVLM.stream` / `stream_images` (:765/:789) | `_llm.stream` (:805) | as above plus `stream:true` | as `stream`; reasoning and tool deltas dropped |
| `OpenAICompatVLM.ask_video` (:736) | `_llm.chat` | one `video_url` part with a full `data:video/...;base64` URL | as `chat` |
| `OpenAICompatTTS.synthesize` (:938) | `post` JSON `{input, response_format:"wav"}` | whole audio buffered (`resp.content`, :957) | audio returned only after full synthesis |
| `_PocketTTS.stream` (:984) | `client.stream("POST")` JSON `{input, response_format:"pcm", stream:true}` | server must honour `stream` | `resp.aiter_bytes()` (:1017); sample rate/channels from headers `x-audio-sample-rate`, `x-audio-channels` |
| Riva `RivaTTS.synthesize` / `_stream` (_riva_grpc.py:216/:245) | gRPC `synthesize_online` for BOTH paths | `synthesize` = concatenated `_stream` then WAV wrap (:239-243) | one `asyncio.to_thread(next, responses)` per chunk (:276) |

Same model parameters either way: `payload` is built by one function `_build_payload` (:313). Streaming adds only `stream: true`. Both honour `max_tokens`, `temperature`, thinking switch, `tools`. Neither sets `logprobs`, `n`, `stream_options`, or `top_p`.

Client config shared by all: `httpx.AsyncClient(timeout=<60s LLM/VLM/embed, 30s STT/TTS>, trust_env=False)` (:310, :855, :935, :1070). That is httpx's default Limits (max_connections=100, keepalive 20, keepalive_expiry=5 s; httpx library defaults, not set in repo code), default HTTP/1.1, no retries, no custom transport. `timeout` is a per-phase timeout (connect/read/write/pool), NOT total wall-clock. In streaming, the 60 s read timeout restarts on every received chunk, so a stream can run indefinitely as long as bytes keep arriving; in `chat()` the read timeout is "time to first byte after request sent", i.e. the 60 s must cover the ENTIRE generation (server sends nothing until done). A per-call `timeout=` overrides all four phases (:371, :404, :456).

## 2. What is buffered/held in client code

### chat()
* Everything: httpx buffers the whole response; `resp.json()` is one parse of the complete body (cost scales with n_out, ~tens of microseconds for a few hundred tokens).
* `ChatResponse.raw` keeps the whole provider dict including `usage` (token counts are available only on this path).
* Reasoning: `_parse_chat_response` reads `reasoning_field` (or `reasoning`/`reasoning_content`) (:154-157). Reasoning text is returned separately; it is NOT folded into `content` unless the server does it (the `<think>` stripping in tools/_vision.py `visible_text` :164 is only applied to VLM non-streaming answers).
* Tool calls: returned as-is from the provider message (:159-169), no JSON validation of `arguments`.

### stream()
* Yields only `delta.content` strings (:422-424). Reasoning deltas and tool_call deltas are silently discarded. Malformed JSON lines and chunks without `choices[0].delta` are skipped (`continue`, :420) - no error.
* A model that decides to call a tool produces zero output and no error through this API. Returns on `[DONE]` (:416). It does not check `finish_reason` or that `[DONE]` arrived (truncated stream ends silently on EOF).
* No usage info; `finish_reason` not exposed (cannot detect `length` truncation).

### stream_events()
* Yields `text` and `reasoning` events immediately as deltas arrive (:496-507); text event and reasoning event are separate objects (each delta chunk may yield up to 2 events).
* Tool calls: fragments are ACCUMULATED in `tool_fragments` (:461, :518-539) and are NOT emitted until after `[DONE]` + a `finish_reason` + all fragments validated (JSON-parse of `arguments`, must be an object, :559-577). Then all `tool_call` events are yielded back-to-back, followed by `finish` (:578-580). Consequence: tool-call latency = full generation time of the call (TTFT + args tokens/decode) + `[DONE]`; there is no way to start a tool before generation ends. Text produced before a tool call (preamble) is still streamed live.
* Strict: malformed JSON -> `ValueError` (:479) (stream() would skip), missing `[DONE]` -> `ValueError` (:545), missing finish_reason -> `ValueError` (:547), `tool_calls` finish_reason with no fragments -> `ValueError` (:551), `error` chunk -> `RuntimeError` (:483). Only choice index 0 is read (:491).
* Reasoning: yielded live; consumers that ignore `kind == "reasoning"` (Conversation does, `_conversation.py:303`) see nothing until the model leaves its thinking phase. Nemotron presets default `enable_thinking=False` (presets/nemotron3_nano.py:21, nemotron_omni.py:19) so reasoning is normally empty.
* The `finish` event is the only place `finish_reason` appears. No usage.

## 3. Where first-token / visible-content timing can differ

* **chat():** latency observed by caller = HTTP request send + server queue + prefill + n_out * decode + response transfer + parse. TTFT is not observable. Visible text and reasoning both only arrive at the end. With thinking ON, total includes all reasoning tokens (`thinking_token_budget` caps it, `_thinking.py:167-168`, only when thinking not off and control in (None, chat_template)).
* **stream()/stream_events():** first visible text = HTTP send + queue + prefill + (reasoning tokens, if thinking on; these are not yielded by stream()) + first content token. For `stream()` the first yielded item is the first non-empty `content` delta; empty-content deltas (role-only first chunk, reasoning chunks) are skipped (:423). For `stream_events()` the first event of ANY kind (incl. `reasoning`) is earlier than first `text`. Profiling must therefore record first-event, first-reasoning, first-text separately.
* **Tool turns:** stream_events text TTFT is meaningless if the model goes straight to a tool call; time-to-tool-dispatch = time-to-`finish` event (see above). `chat()` returns the same moment (end of generation) minus one `[DONE]` round and minus per-chunk client work, so for tool turns streaming has NO latency benefit and slightly higher client overhead.
* **VLM:** prefill dominates (image tokens); streaming helps only the decode portion. `tools/vision.py` logs "Image VLM stream first token latency_ms" (vision.py:192-196) measured from just before `nemo_relay.llm.stream_execute` (includes Relay setup + image base64 + HTTP + server prefill).
* **TTS:** `synthesize` returns after whole-utterance synthesis (RTF * audio_duration); `_PocketTTS.stream` first chunk after server first PCM block (header read at :1008 occurs after response headers, which the server may flush before or after first audio - needs measurement). If the Pocket server does not send `x-audio-sample-rate`, `_PocketTTS.stream` silently falls back to a SECOND request `synthesize()` (:1026-1030) after the first streaming request has been closed: first-chunk latency = (cost of the failed/streamed first request) + whole-utterance synth. Riva: both `synthesize` and `stream` use `synthesize_online`; only the aggregation differs.

## 4. Per-chunk client overhead (code-derived)

Per SSE line in `stream_events` (:469-543): `aiter_lines` split + `startswith` + `json.loads` of a ~150-300 byte JSON + ~6-10 isinstance checks + constructing a frozen `ChatStreamEvent` dataclass per event + one async-generator `yield` hop. Estimated 15-40 microseconds CPU per chunk (guess; measure). `stream()` slightly cheaper (no validation, one yield). Add the consumer chain; in Conversation each text event costs: `stream_events` yield -> `_stream_response` yield (`_conversation.py:302-305`) -> `answer()` loop -> `await runtime.publish(ResponseOutput)` (:180) per chunk; publish cost is outside this card but is the per-token term that scales with n_out. httpx `aiter_lines` also decodes in TCP-read-sized blocks, not per token, so with fast servers several SSE lines can arrive in one read (client-side batching is invisible, delivery to consumer is still line by line).

VLM tool path adds per chunk (tools/vision.py:214-225): `_stream_vlm` wraps each token into `{"choices":[{"delta":{"content":token}}]}` dict -> Relay `stream_execute` Python/Rust boundary + the `lambda chunk: fragments.append(stream_text(chunk))` callback (:182) -> consumer calls `stream_text(chunk)` again (:189) -> `ImageQueryChunk` pydantic construct (:198) -> `AsyncTool.stream` (async_tools.py:41-88): per chunk `queue.put(model_validate(chunk))` with `Queue(maxsize=1)` (:53), a fresh `asyncio.create_task(queue.get())` + `asyncio.wait` (:61-66) per chunk. This is lockstep with maxsize=1: the producer cannot read the next token until the consumer has taken the previous one. Estimated 80-200 microseconds per token (guess), negligible versus decode time (~10-30 ms/token) unless consumer is slow (each `await` consumer step stalls the HTTP read; TCP buffer absorbs it).

Non-streaming per-request overhead: one JSON dump of request (`httpx json=` uses stdlib `json.dumps` on the event-loop thread; for a VLM request with 1-4 base64 images 0.3-5 MB this is ~1-10 ms of blocked event loop - guess) and one `resp.json()`.

## 5. Experiment guidance (streaming vs non-streaming)

Hold fixed: prompt, `max_tokens`, `temperature`, thinking state, server. Differences to expect:
1. Total time to complete text: streaming ~= non-streaming + n_chunks * per-chunk overhead (tiny) - nothing is gained at completion.
2. TTFT: only streaming can measure it; use `stream_events` for tool-capable turns, `stream` for text-only; record first `reasoning` and first `text` separately.
3. Tool turns: expect parity (held until [DONE]).
4. Token counts: only `chat()` gives `usage` (raw). For streaming, count `text` events as a proxy (each SSE chunk usually = 1 token on vLLM with default stream interval; not guaranteed) or add `stream_options` via `default_extras` (`{"stream_options":{"include_usage":true}}` goes through :337; `stream_events` explicitly tolerates the resulting empty-choices usage chunk :487-489, `stream()` would hit IndexError and skip it (:420)).
5. Timeout semantics differ: 60 s default = total generation budget for chat, inter-chunk gap for streaming.
6. Error visibility: chat raises `HTTPStatusError` after `logger.error` (:374-376); streaming raises after reading body (:407-410), mid-stream failures surface as `httpx.ReadError`/`RemoteProtocolError`/`ValueError` after partial output already consumed (partial output may already have been published to TTS).

Confidence: code-derived for all wire/buffering behaviour; per-chunk microsecond figures are guesses.
