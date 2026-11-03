<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Thinking switch, presets, config and factory (agent-sdk/xr-ai-models/xr_ai_models/_thinking.py, presets/*.py, _config.py, _factory.py)

## Role in a turn
Not a runtime call. These decide, per endpoint, which request fields are sent (thinking on/off, timeouts, which client class). They set the biggest hidden latency knob: whether the model spends tokens on reasoning.

## Critical-path calls
1. `make_llm/make_vlm/...` (once at startup) -> builds `OpenAICompatLLM(...)` with `ThinkingSpec`, `default_extras` (`_factory.py:91-141`). Construction creates a new `httpx.AsyncClient` per service object (`_openai_compat.py:310`); each service has its own connection pool.
2. Per request: `apply_thinking(spec, payload, enable_thinking, thinking_budget)` (`_thinking.py:136-172`) mutates the payload; for `system_prompt` control it inserts/prefixes a `detailed thinking on|off` line at the start of the first system message (`:175-204`; regex scan of system text per call).

## Fixed waits & timers
| Constant | Value | Where |
|---|---|---|
| Nemotron3-Nano default thinking | off (`chat_template`, default False) -> sends `chat_template_kwargs.enable_thinking=false` every call | presets/nemotron3_nano.py:21 |
| Nemotron-Omni default thinking | off | presets/nemotron_omni.py:19 |
| Cosmos-Reason1 VLM | off | presets/cosmos_vlm.py:24 |
| Cosmos3-Nano-Reasoner | `control: none` (nothing sent; preset docstring: the official chat template has no reasoning switch, so reasoning behaviour is whatever the model does by default - needs measurement) | presets/cosmos3_nano_reasoner.py:19 |
| Magpie TTS timeout | 60 s | presets/magpie_tts.py:9 |
| Parakeet STT / Pocket TTS timeout | 30 s | presets/parakeet_stt.py:9, pocket_tts.py:10 |
| OCR timeout | 120 s | presets/nemotron_ocr.py:10 |
| Default timeouts by category | 60 s llm/vlm/embedding, 30 s others | `_config.py:908-909`, `EndpointSpec.timeout` `:89` |
| `thinking_budget` | not sent unless caller passes it | `_thinking.py:167-168` |
| No default `max_tokens`/`temperature` anywhere in config or presets | - | - |

## Cost drivers
Reasoning tokens (when on): decode time = n_reason / decode_tps, unbounded unless `thinking_budget` (server-enforced via `thinking_token_budget`; omitted for `control=none` and `system_prompt` models).

## Estimate formula
`delta_t_thinking = n_reason / decode_tps` added to non-streaming latency and to streaming visible-TTFT (reasoning arrives as `reasoning` events first). Preset default adds 0. Measure `n_reason` distribution for the real prompts.

## Hidden latency / improvement candidates
* Reasoning is off by default in all Nemotron/Cosmos-Reason1 presets: good. Per-call opt-in without budget is a latency hazard (preset docstring: thinking-on "burns the whole max_tokens budget", nemotron3_nano.py:9-12).
* The `cosmos3_nano_reasoner` preset cannot turn reasoning off from the client (control none) - a reasoner may emit `<think>`; `visible_text` strips it only in the non-streaming VLM tool path (`xr_ai_tools/_vision.py:164`). Expect long decode on that model (needs measurement).
* One httpx client per service object: LLM + VLM + decision to the same host do not share connections (minor).

## What to measure live
Same prompt with `enable_thinking` None/False/True and `thinking_budget` 0/128/512: latency, `usage.completion_tokens`, `reasoning` length. API: `llm.chat(msgs, enable_thinking=..., thinking_budget=...)`.

## Confidence
Code-derived for what is sent; token costs: needs measurement.
