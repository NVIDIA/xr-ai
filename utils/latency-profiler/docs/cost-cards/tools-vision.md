<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Vision query tools (agent-sdk/xr-ai-tools/xr_ai_tools/vision.py, _vision.py, image.py)

## Role in a turn
Model-callable tools (`query_image`, `query_images`, `query_video`, `stream_image_query`) that turn an image reference into a VLM answer. They sit on the critical path of any turn that needs vision: the tool loop waits for the whole VLM answer (non-streaming) before the LLM sees it.

## Critical-path calls
1. `Tool.invoke` -> `json.loads`, `model_validate` (`tools.py:84-100`), pydantic; then `typed.tool_execute` Relay wrapper (`tools.py:75`).
2. `_ImageInference.answer` (`vision.py:114-154`): for each input `images.resolve(ref)` (LookupError -> unavailable result) `:120-127` (dict ops, trivial).
3. `image_sanitizer()` context: `nemo_relay.scope.get_handle()` + `register_llm_sanitize_request` (+ deregister on exit) `_vision.py:30-50`, two native calls per request.
4. `nemo_relay.llm.execute(VLM_CALL_NAME, relay_request(...), self._ask_vlm, codec=OpenAIChatCodec(), response_codec=OpenAIChatCodec())` `vision.py:130-141`. Relay builds/encodes an OpenAI chat request containing only `xr-image://` URIs (no bytes), runs sanitizer (`copy.deepcopy` of request content `_vision.py:191`), then invokes `_ask_vlm`.
5. `_ask_vlm` (`vision.py:204-212`): `vision_inputs` re-decodes the Relay request (:205), resolves images again to bytes (:207), `timestamped_question` (:208), then `vlm.ask_images(...)` (`models-vlm-image-paths.md`) - the actual wait; passes Relay-provided headers (`headers_from_relay`, `_relay.py`).
6. `openai_response(text)` -> Relay response codec -> `response_text` -> `visible_text` regex `<think>.*?</think>` DOTALL (`_vision.py:164-167`) -> `ImageQueryResult`.
Streaming variant `_ImageInference.stream` (`vision.py:156-202`): `nemo_relay.llm.stream_execute` + `vlm.stream_images`; wrapped by `AsyncTool.stream` (see tools-tool-calling-loop.md).
All steps serial; no `gather`/`create_task` (except AsyncTool's producer task).

## Fixed waits & timers
| Constant | Value | Where |
|---|---|---|
| `_MAX_IMAGES` per call | 4 | `vision.py:36` |
| VLM timeout | inherited from VLM client: 60 s | `_openai_compat.py:623` |
| VLM `max_tokens` / `temperature` | None -> not sent (tool passes none: `vision.py:206-211`) | server default |
| `ImageRegistry` capacity | 128 | `image.py:221` |
| Failure handling | any Exception -> text "VLM server unavailable - please retry." `available=False`; no retry/backoff | `vision.py:149-154` |
| Timeouts specific to tools | none | - |
| VLM answer system prompt | caller-supplied `system_prompt` (default "") | `vision.py:231` |

## Cost drivers
#images (<=4) x resolution -> image tokens (server prefill); image bytes (base64+JSON+upload); decode length of the answer (unbounded); Relay wrapper overhead (constant, small); prior steps that produced the image (frame capture, JPEG encode).

## Estimate formula
`t_query = t_validate(~0.05 ms) + t_relay_exec_overhead(measure; est. 0.2-1 ms) + t_vlm(n_img, px, n_out) [see models-vlm-image-paths.md] + t_visible_text(<0.1 ms) + t_result_render(<0.1 ms)`.
Dominated by `t_vlm`. Streaming variant: `t_first_token = overhead + t_b64/json + rtt + prefill`, matching the log line "Image VLM stream first token latency_ms" (`vision.py:192-196`).

## Hidden latency / improvement candidates
* Image bytes never go through Relay (URIs only), so Relay deepcopy/sanitize overhead is small; but the VLM sees full-resolution images (no resize) - biggest saving is downscaling (see VLM card).
* `images.resolve` is called twice per image (check at `:121` then inside `_ask_vlm` `:207`); trivial.
* Non-streaming `query_image` waits for whole answer before the agent LLM continues; `StreamingImageQueryTool` exists for text pass-through, but a native-tool-loop LLM consumes full text anyway (benefit only for direct speech streaming).
* The model-facing answer text round-trips through the agent LLM again (tool result -> next `chat`): total cost = VLM latency + one LLM prefill including the answer. If the VLM answer could be spoken directly (`return_direct`, `Tool(..., return_direct=True)` option exists `tools.py:43`) that final LLM hop (hundreds of ms) would be skipped; the vision tools are constructed without it.
* Failed VLM returns a polite string with no retry (cheap failure, but the user sees degraded answer).

## What to measure live
* Public API without code changes: `ImageQueryTool(images=registry, vlm=make_vlm(...)).invoke(json.dumps({"image":{"uri":ref.uri},"query":"..."}))`; compare to calling `vlm.ask_image` directly with same bytes to isolate Relay + tool overhead (needs `nemo_relay` runtime; without an active scope it may still work).
* Streaming: time to first `ImageQueryChunk` from `StreamingImageQueryTool.stream({...})`.
* n_img 1/2/4; resolutions.

## Confidence
Call graph: code-derived. Relay overhead: needs measurement (native code not read). VLM time: needs measurement.
