<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# llama-nemotron-llm-server

OpenAI-compatible LLM HTTP server using HuggingFace transformers.

Default model: [`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) (~16 GB at BF16).

An 8B dense Llama 3.1 reasoning model from NVIDIA, post-trained for
tool-calling, RAG, and chat. Deploys via plain HuggingFace transformers —
**no custom modeling code, no `trust_remote_code`, no native Mamba/Triton
kernels**. Licensed for commercial use under the NVIDIA Open Model License +
Llama 3.1 Community License.

## Quickstart

```bash
cd ai-services/llm/llama_nemotron
uv sync
uv run llama_nemotron_llm_server --config llama_nemotron_llm_server.yaml
```

First run downloads weights (~16 GB) to the shared `models/` cache at the repo root.

## Endpoints

| Endpoint                  | Method | Description                                       |
|---------------------------|--------|---------------------------------------------------|
| `/health`                 | GET    | Health check (`{"status": "ok"}`)                 |
| `/v1/models`              | GET    | List available models                             |
| `/v1/chat/completions`    | POST   | Chat completion (OpenAI-compatible, `tools` OK)   |

## Config keys (`llama_nemotron_llm_server.yaml`)

| Key              | Type       | Default            | Description                                                                                    |
|------------------|------------|--------------------|------------------------------------------------------------------------------------------------|
| `model`          | str        | (required)         | HuggingFace model ID (any Llama-3.1-compatible chat template)                                  |
| `port`           | int        | `8106`             | HTTP port                                                                                      |
| `host`           | str        | `0.0.0.0`          | Bind address                                                                                   |
| `hf_token`       | str        | `""`               | HuggingFace token (for gated models)                                                           |
| `system_prompt`  | str        | `""`               | Optional system prompt prepended to all requests (e.g. `"detailed thinking off"`)              |
| `max_new_tokens` | int        | `1024`             | Max tokens to generate                                                                         |
| `model_cache`    | str        | `../../../models`  | Weight cache path (relative to YAML)                                                           |
| `dtype`          | str        | `bfloat16`         | Torch dtype (`bfloat16`, `float16`, `float32`)                                                 |
| `stop`           | list[str]  | `[]`               | Default stop sequences applied if request omits `stop` (empty: rely on tokenizer EOS handling) |

## Reasoning toggle — per-turn via system prompt

Llama-3.1-Nemotron-Nano-8B-v1 flips between reasoning-on and reasoning-off
mode based on whether a system or user message contains the literal tokens
`"detailed thinking on"` or `"detailed thinking off"`:

```bash
# Reasoning ON — the model emits a <think>...</think> preamble before the final answer.
curl -X POST http://localhost:8106/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "detailed thinking on"},
      {"role": "user", "content": "Why does water expand when it freezes?"}
    ],
    "max_tokens": 2048
  }'

# Reasoning OFF — fast, direct answer (recommended for voice / low-latency use).
curl -X POST http://localhost:8106/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "detailed thinking off"},
      {"role": "user", "content": "Say OK"}
    ],
    "max_tokens": 16
  }'
```

The model's default behavior (no toggle specified) is reasoning-on. Pipecat-nat
injects the toggle per-turn based on a keyword heuristic in
`NatBackend.infer()` — see the pipecat-nat README.

See the [model card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1)
for recommended sampling parameters: `temperature=0.6`, `top_p=0.95` for
reasoning-on; greedy decoding for reasoning-off.

## Tool calling (native Llama 3.1 format)

The server accepts OpenAI-shape `tools=[...]` (and `tool_choice`) in the
chat-completions request. When tools are provided, two things happen:

1. `tools` is forwarded into `tokenizer.apply_chat_template(..., tools=tools)`
   so the model sees its native tool schema format.
2. The generated assistant text is parsed for tool-call JSON. The parser
   handles the three shapes Llama-3.1-Nemotron is known to emit (bare
   `{"name": ..., "parameters": ...}` JSON, `<|python_tag|>{...}`, and
   `<TOOLCALL>[{...}]</TOOLCALL>`) and returns them in OpenAI wire format:
   `choices[0].message.tool_calls = [{id, type: "function",
   function: {name, arguments}}]` with `finish_reason: "tool_calls"`.

Conversation history containing prior tool rounds is passed through verbatim:
assistant messages with `tool_calls` and `tool`-role messages with
`tool_call_id` flow into `apply_chat_template` so multi-turn tool-calling
loops (e.g. LangChain `ChatOpenAI.bind_tools()`, NAT's `tool_calling_agent`)
work out of the box.

```bash
curl -X POST http://localhost:8106/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "detailed thinking off"},
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "max_tokens": 256
  }'
```

Requests without a `tools` array are handled exactly as before — the
response shape is byte-identical (`finish_reason: "stop"`, no `tool_calls`
field).

## Swap models

Edit `llama_nemotron_llm_server.yaml`:

```yaml
model: nvidia/Llama-3.3-Nemotron-Super-49B-v1  # example
```

Any HuggingFace `AutoModelForCausalLM`-compatible model with a Llama 3.1-style
chat template works. Adjust `max_new_tokens`, `dtype`, and `stop` as needed
for the new model.

## Notes

- **No continuous batching.** FastAPI + raw transformers is simpler but slower
  under load than vLLM. For single-user voice agents, this is fine.
- **First-run download** goes to the shared `models/` cache at the repo root.
- **No `trust_remote_code`** — Llama 3.1 is natively supported by the
  transformers package, so there are no custom modeling files to download
  or native kernels (like `mamba-ssm`) to compile.
- **Reasoning preamble is NOT stripped server-side.** Unlike the sibling
  `nemotron/` server which strips `<think>...</think>`, this server returns
  the raw assistant output. Use `"detailed thinking off"` to avoid the
  preamble entirely in latency-sensitive paths.
