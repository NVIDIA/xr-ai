<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# DecisionService / SystemOne (agent-sdk/xr-ai-models/xr_ai_models/_systemone.py, _factory.py:39, _protocols.py:229-313; server: services/clef-server/clef_server/_service.py)

## Role in a turn
A single-shot, non-generative classifier ("choice-only" decisions with probabilities), used as a fast gate or router instead of an LLM call. One HTTP request, one GPU forward pass on the server, no decode loop.

## Critical-path calls
1. Client-side validation of state and questions (pure Python loops over questions/criteria) - `_systemone.py:258-285`. Microseconds.
2. If `images`: `_normalize_image` each image (read file if Path, base64 encode, sync on event loop) `:293`, then prefix check JPEG/PNG only `:294-297`.
3. `await client.post(url=/v1/systemone, json=payload, timeout=effective_timeout)` `:298-303`. Single serial request; no streaming, no retries.
4. `response.json()` and response validation (every answer, probabilities sum, confidence match) `:305-342`. Microseconds.
5. Server side (for the estimate): body read -> `json.loads` -> `inference_lock` (global, single worker executor) -> image decode + processor + forward (`_service.py:165-203`, `:206-222`). Requests are fully serialized: concurrent decisions queue behind each other.

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| client timeout | 30.0 s (decision is not in the 60 s category list; `DecisionSpec.endpoint` default 30.0, `_config.py:570`; YAML `timeout:` overrides) | `_config.py:908-909`, `_factory.py:53` (passes `spec.timeout`) |
| per-call `timeout` | capped: `min(requested, self.timeout)` | `_systemone.py:350-361` (a per-call value can only shorten) |
| retries | none | - |
| `logprobs`/`max_tokens`/`temperature` | none sent; the wire has no generation params. Probabilities come from the server head, not from OpenAI `logprobs` | payload `:289` is only `{model, state, questions, images?}` |
| server `max_length` | 4096 tokens (checked-in YAML); body limit 1 MiB | docs/source/reference/clef-flash.md:59-63; `_service.py:165-176` |
| image admission | count <= `max_length // 64`; aggregate source pixels <= `max_length * 32*32` (~4.2 Mpx total at 4096) | docs clef-flash.md:124-132, `_images.py:56-59` |
| server concurrency | `inference_lock` + 1-worker executor | `_service.py:195-199` |
| connection pool | httpx defaults (keepalive_expiry 5 s) | `_systemone.py:245` |

## Cost drivers
* Prompt tokens: `state` (str or mapping serialized by the server) + each question's `instructions` + all criteria strings; limited to 4096 tokens by server.
* Number of questions in one request (all evaluated in one request, server cost grows with question count/tokens).
* Number and size of images: each image is decoded, processed, and tokenized on the single serialized worker; base64 inflation +33 % must fit the 1 MiB body limit (a full 1080p JPEG q90 is typically 300-600 KB raw -> 0.4-0.8 MB base64: near the limit; two such images exceed it -> HTTP 413). Docs advise resizing/cropping on the client.
* Queueing behind other decision requests (global lock).

## Estimate formula
`t_decide = t_validate(~0.05 ms) + t_b64(images) + t_json_dump + t_rtt + t_queue_lock + t_decode_img(n_img, px) + t_forward(tokens, n_img_tokens) + t_response_parse(~0.1 ms)`.
Measured terms: `t_forward` per token count (prefill-only, no decode: expected 10s of ms for ~1k tokens on a modern GPU - guess), `t_decode_img` per image resolution, `t_queue_lock` under concurrent calls. Best/typical/worst from code constants alone: client overhead 0.2 / 1-3 / 10 ms (multi-MB images); total needs measurement.

## Hidden latency / improvement candidates
* Images re-encoded to base64 on every call (no cache) `_systemone.py:293` and referenced images are re-processed per request on the server ("There is no cross-request reference cache", clef-flash.md:142-143). If a fixed reference image is used each call, cost is paid each time: dropping or pre-resizing references can save the server decode+tokenize time (needs measurement; 5-30 ms per image guess).
* Pre-resize client-side to ~448-768 px: reduces base64, body bytes, server decode and image tokens. Saving: 1-20 ms transfer/decode, more in forward time (fewer image tokens).
* Global serialization: a router decision and a vision decision in parallel will serialize; `asyncio.gather` does not help.
* A request that fails validation after server tokenization (HTTP 422) is wasted time; no client-side length precheck.
* No `max_tokens` semantics: nothing to cap; forward is fixed-cost.

## What to measure live
* `make_decision(config, name).decide(state, questions, images=[...])` wall time: text-only vs +1, +2 images at 224/448/768/1080p; vs question count (1, 3, 8); vs state size (100/500/2000 tokens).
* Two concurrent `decide()` calls (verify serialization: t_total ~ 2x).
* Response `usage` field (`DecisionResponse.usage`) for token counts to fit the formula.
* Idle >5 s then call (new connection).

## Confidence
Client path and constants: code-derived. Server forward time and image decode cost: needs measurement. "No logprobs on the wire" is code-derived; how the server derives probabilities was not read beyond `_service.py:165-222` (single forward, no generation per docs).
