<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# OpenAICompatVLM image/video paths (agent-sdk/xr-ai-models/xr_ai_models/_openai_compat.py:139-253, :607-826; agent-sdk/xr-ai-tools/xr_ai_tools/_pixels.py)

## Role in a turn
The visual question-answering call (ask_image(s), stream/stream_images, ask_video). Prefill of image tokens usually dominates; encoding of the image on the client is secondary but runs on the event-loop thread.

## Critical-path calls
1. Image acquisition (outside this class): hub frame -> `frame_to_pil` -> JPEG q90 bytes (`_pixels.py:26-56`, `current_frame.py:113-115`, in `asyncio.to_thread`).
2. `_build_image_messages` (:654-674): `_normalize_image` per image, SERIAL, SYNC on the event loop: `Path.read_bytes()` (:216) or `bytes -> base64.b64encode -> str` (:207-209, :219) or pass-through for str URLs/data URLs (:214). No resize, no re-encode, no downscale, no cache. Order: all images, then the text part (:669-671).
3. `OpenAICompatLLM.chat` (:715) or `.stream` (:805): `json=payload` serializes the whole multi-MB string on the event loop, POST, wait for response (see models-openai-compat-llm.md).
4. `ask_video` (:736-763): `_normalize_video` reads the entire file (`Path.read_bytes()` :249) and base64 encodes it (+33 %) into a single `video_url` part; requires `capabilities.video` else raises (:753).
5. Tools layer wraps this in Relay (`xr_ai_tools/vision.py:128-141`, see tools-vision.md).

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| request timeout | 60.0 s (VLM category) | `OpenAICompatVLM.__init__` :623; `_config.py:909` |
| `max_tokens` | None -> not sent -> server default | :682, :705 (no default set by VLM class) |
| `temperature` | None -> not sent | :683 |
| thinking | Cosmos-Reason1 `cosmos_vlm`: `chat_template` default False (presets/cosmos_vlm.py:24); Cosmos3-Nano-Reasoner: `control: none` (cosmos3_nano_reasoner.py:19); VLM wrapper exposes NO per-call `enable_thinking` (calls `_llm.chat` without it :715) | fixed per endpoint |
| JPEG quality | 90 | `_pixels.py:55` |
| resize | none anywhere in the VLM class or `_pixels.py`; frames go at native camera resolution | - |
| max images per tool call | 4 | `xr_ai_tools/vision.py:36` (`_MAX_IMAGES`), enforced in request models :54, :68 |
| `ImageRegistry` capacity | 128 refs (LRU by touch) | `xr_ai_tools/image.py:221` |
| video server-side | video disabled by default on vlm-server (`max_videos_per_prompt` 0) | presets/cosmos_vlm.py docstring, cosmos3_nano_reasoner.py docstring |

## Cost drivers
* #images (<=4 per tool call) x image tokens per image (model/processor dependent; scales with resolution, e.g. Qwen2.5-VL-style dynamic resolution: ~ (W/28)*(H/28) tokens) -> prefill time. Native 1080p = ~2.6 K tokens per image for 28x28 patches (guess from processor family; verify per model).
* Image bytes: JPEG q90 1080p ~0.3-0.6 MB -> base64 0.4-0.8 MB; 4K ~1.5-3 MB; PNG frames from video memory (`get_historical_frame` "recorded PNG frame") can be 2-6 MB each.
* Video: whole file as one base64 string; server must decode H.264.
* Decode tokens of the answer (no cap by default).
* Image data URLs are in the HTTP JSON body: multi-MB request upload cost (loopback ~free; LAN 1 Gbps: 8 ms/MB; WAN: much more).

## Estimate formula
`t_vlm = t_img_acquire + n_img * (t_b64(bytes) ~ bytes/1GB/s) + t_json_dump(total_bytes ~ 1-2 ms/MB) + t_upload(total_bytes/bandwidth) + t_server_decode_imgs(n_img, px) + t_prefill(P_text + n_img * tok_per_img) + n_out/decode_tps`.
Terms to measure: `tok_per_img` and server image decode time for the real resolutions; `decode_tps`. Code-only client overhead: 0.5 ms (1 small image) / 3-8 ms (2 x 1080p) / 20-40 ms (4 x PNG 4 MB, sync `read_bytes` + b64 + json) - all on the event-loop thread, i.e. it also delays unrelated coroutines (TTS pumping, other turns) while running.

## Hidden latency / improvement candidates
* No client-side downscale: sending 1080p/4K where 448-768 px would answer the question multiplies image tokens 4-10x; likely the largest saving of any image-path change (hundreds of ms of prefill per image, needs measurement). Downscale in `to_thread` before `registry.put`.
* base64 + JSON dump + file read are synchronous on the event loop (:216, :219, `httpx json=`): wrap in `asyncio.to_thread` or pre-encode once when stored in the registry. Saving: ~1-10 ms of loop blocking per call.
* The same image is base64-encoded again for every query (registry stores raw bytes, `vision.py:206-211` resolves bytes each time): cache data URL alongside bytes. Saving ~1 ms per repeat query.
* Images are encoded serially in a Python list comprehension (:669); negligible vs prefill.
* Multi-image requests: one request is cheaper than N requests (one queue slot) but prefill is linear in total image tokens.
* `ask_video` base64 of the whole file: send fewer frames (as images with timestamps via `VideoQueryTool`) or a shorter clip; server decode of H.264 is extra.
* Streaming (stream_images) cuts perceived latency only after prefill; `visible_text` `<think>` stripping applies only to non-streaming (`_vision.py:164`, used `vision.py:142`) so a streamed answer from a thinking model can leak `<think>` text.

## What to measure live
* `VLMService.ask_image` with the same image at 224/448/768/1080p/4K, JPEG vs PNG: total time, `ChatResponse.raw["usage"]["prompt_tokens"]` (gives tokens-per-image), TTFT via `stream`.
* n_img = 1,2,4 at fixed resolution to confirm linearity.
* Event-loop stall: run a 1 ms heartbeat task during the call and record max gap (shows sync encode/json cost).
* `ask_video` with 2 s / 10 s clips vs equivalent N frames via `ask_images`.

## Confidence
Client path, no-resize, serial sync encoding: code-derived. Tokens-per-image and server decode cost: needs measurement. Client millisecond costs: guess from throughput of base64/json (verify).
