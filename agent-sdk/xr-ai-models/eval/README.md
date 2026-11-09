<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Structured-output eval

A live-model eval of JSON Schema output and forced tool calls. Run it for any
model or server before declaring `capabilities.json_schema` or
`capabilities.tool_choice` for it. See the
[models reference](https://nvidia.github.io/xr-ai/latest/reference/agent-sdk-models.html#structured-output-and-forced-tool-calls)
for the API.

Start the model through its repository entry point (for example the
`model-servers` Nemotron Omni LLM on port 8108), then run from
`agent-sdk/xr-ai-models/`:

```bash
uv run --project . --group eval python eval/structured_output_eval.py
```

Select the model the same way as the conversation evals: `--base-url` with
`--preset` (default `http://localhost:8108` with `nemotron_omni`), or
`--models path/to/models.json` with `--model`. The eval turns both capabilities
on for the client it builds, so it measures the endpoint, not what its preset
declares. Other options: `--samples` (default 5), `--temperature` (default 0.6),
`--thinking-budget` (default 512), `--thinking off|on|both`, `--modes`, and
`--report path.json` for every sample.

It exits non-zero when, with reasoning off or on, a JSON Schema mode returns an
answer that does not match its schema, or a forced tool mode returns no call.
Forced-call arguments are checked against the schema and reported, but do not
gate: a server may constrain only the call. Correct answers and latency are
reported but do not gate.

## Modes

| Mode | Request | Role |
|---|---|---|
| `json_schema` | `chat(response_schema=...)` | Gated: every answer matches |
| `json_schema/stream` | `stream_events(response_schema=...)` | Gated: every answer matches |
| `tool_named` | `chat(tools=[t], tool_choice="t")` | Gated: every response calls |
| `tool_named/stream` | `stream_events(tools=[t], tool_choice="t")` | Gated: every response calls |
| `tool_required` | `chat(tools=[t], tool_choice="required")` | Gated: every response calls |
| `prompt_only` | Schema stated in the system message, no constraint | Reference: no feature |
| `tools_auto` | `chat(tools=[t])`, the model may skip the call | Reference: today's commit-tool pattern |
| `json_schema_flipped` | `response_format` with the opposite schema statement: stated without reasoning, left out with it | Ablation of the client's rule, not reachable through the API |

With reasoning on, each request sends `thinking_budget` and a `max_tokens` of
the budget plus 1024. In the `tools_*` modes the answer is the first call's
arguments, or the response text when the model calls nothing, as a tool loop
would treat it.

## Cases

`cases/structured_output.yaml` holds the prompts and JSON Schemas: the tea
sample's `ChangeDecision` (important and unimportant changes) and
`TranscriptSummary`, a nested session state with an enum and bounded integers,
an instrument reading with a unit enum and a nullable note, and a bounded list
of steps. Each case has checks that grade the answer (an expected value, a
substring, a numeric range or a minimum length), plus accepted and rejected
example answers that the repository tests use to check the graders offline.

## Results: Nemotron Omni

`nemotron_omni` on vLLM 0.31.0 (`nemotron_v3` reasoning parser, `qwen3_coder`
tool parser), 16 samples per case, mode and reasoning setting (96 per row),
temperature 0.6 and a 512-token thinking budget:

| Mode | Reasoning | Called | Valid | Correct | p50 s |
|---|---|---|---|---|---|
| `json_schema` | off | - | 96/96 | 96/96 | 0.48 |
| `json_schema` | on | - | 96/96 | 94/96 | 2.35 |
| `json_schema/stream` | off | - | 95/96 | 95/96 | 0.53 |
| `json_schema/stream` | on | - | 96/96 | 93/96 | 2.22 |
| `tool_named` | off | 96/96 | 95/96 | 95/96 | 0.78 |
| `tool_named` | on | 96/96 | 96/96 | 94/96 | 3.67 |
| `tool_required` | off | 96/96 | 95/96 | 93/96 | 0.85 |
| `tool_required` | on | 96/96 | 96/96 | 94/96 | 3.43 |
| `prompt_only` | off | - | 79/96 | 66/96 | 0.53 |
| `tools_auto` | on | 83/96 | 87/96 | 87/96 | 3.08 |
| `json_schema_flipped` | off | - | 96/96 | 83/96 | 0.53 |
| `json_schema_flipped` | on | - | 84/96 | 83/96 | 2.53 |

That run failed the gate once: one streamed `json_schema` answer filled its
remaining tokens with whitespace. Across three full runs, 1 of 1,152
`response_schema` answers failed that way and 6 of 1,728 forced calls had
arguments outside their schema. The flipped ablation supports the client's
rule both ways: stating the schema without reasoning cut correct answers
(an enum biased toward its first value), and leaving it out after reasoning
lost 12 answers to whitespace.
