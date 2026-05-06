<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# llama-nemotron-llm-server

OpenAI-compatible LLM server for
[`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1),
implemented as a thin launcher around vLLM.

An 8B dense Llama 3.1 reasoning model from NVIDIA, post-trained for tool-calling,
RAG, and chat. Licensed for commercial use under the NVIDIA Open Model License +
Llama 3.1 Community License (~16 GB VRAM at BF16).

## Quickstart

```bash
cd ai-services/llm/llama_nemotron
uv sync
uv run llama_nemotron_llm_server --config llama_nemotron_llm_server.yaml
```

First run downloads weights (~16 GB) to the shared `models/` cache at the repo
root. Subsequent runs start offline.

## Endpoints

All endpoints are provided by vLLM's OpenAI-compatible server at
`http://localhost:8106`:

| Endpoint               | Method | Description                                   |
|------------------------|--------|-----------------------------------------------|
| `/health`              | GET    | Server health                                 |
| `/v1/models`           | GET    | List models (returns `{"id": "llm", ...}`)    |
| `/v1/chat/completions` | POST   | Chat completion with `tools=[...]` support    |

## Config keys (`llama_nemotron_llm_server.yaml`)

| Key                      | Type | Default                              | Description                                   |
|--------------------------|------|--------------------------------------|-----------------------------------------------|
| `model`                  | str  | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | HuggingFace model ID                        |
| `host`                   | str  | `0.0.0.0`                            | Bind address                                  |
| `port`                   | int  | `8106`                               | HTTP port                                     |
| `hf_token`               | str  | `""`                                 | HuggingFace token for gated models            |
| `model_cache`            | str  | `../../../models`                    | Weight cache (relative to YAML)               |
| `max_num_seqs`           | int  | `8`                                  | vLLM concurrent request limit                 |
| `tensor_parallel_size`   | int  | `1`                                  | vLLM TP — raise for multi-GPU serving         |
| `max_model_len`          | int  | `32768`                              | Max context tokens                            |
| `gpu_memory_utilization` | float | `0.85`                              | vLLM `--gpu-memory-utilization`               |
| `enforce_eager`          | bool | `false`                              | Skip CUDA graph capture (faster cold start, slightly slower steady state) |
| `enable_tool_choice`     | bool | `true`                               | Pass `--enable-auto-tool-choice` to vLLM      |
| `tool_call_parser`       | str  | `llama3_json`                        | vLLM `--tool-call-parser`                     |
| `vllm_backend`           | str  | `pip`                                | `pip` (wrapper venv) or `docker` (NGC container) |
| `vllm_image`             | str  | `nvcr.io/nvidia/vllm:26.04-py3`      | Used when `vllm_backend: docker`              |

## Tool calling (native Llama-3.1 format)

Tool calling is handled entirely server-side by vLLM's `llama3_json` parser.
Clients send OpenAI-shape `tools=[...]` in the chat-completions request:

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

The server returns `choices[0].message.tool_calls = [{id, type: "function",
function: {name, arguments}}]` with `finish_reason: "tool_calls"`.

Conversation history containing prior tool rounds is passed through verbatim:
assistant messages with `tool_calls` and `tool`-role messages with
`tool_call_id` flow into vLLM's chat template so multi-turn tool-calling loops
(LangChain `ChatOpenAI.bind_tools()`, NAT's `tool_calling_agent`) work out of
the box. Disable tool calling entirely with `enable_tool_choice: false` in the
YAML.

## Reasoning toggle — per-turn via system prompt

Llama-3.1-Nemotron-Nano-8B-v1 flips between reasoning-on and reasoning-off
mode based on whether a system or user message contains the literal tokens
`"detailed thinking on"` or `"detailed thinking off"`:

```bash
# Reasoning ON — model emits a <think>…</think> preamble before the final answer.
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

The reasoning preamble is **not** stripped server-side; clients that don't want
to see `<think>…</think>` should request reasoning-off mode or strip the block
themselves. Default behavior (no toggle specified) is reasoning-on.

See the [model card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1)
for recommended sampling parameters: `temperature=0.6, top_p=0.95` for
reasoning-on; greedy decoding for reasoning-off.

## Choosing the vLLM runtime (pip vs Docker)

`vllm_backend: pip` (default) runs vLLM from the wrapper's venv;
`vllm_backend: docker` runs `nvcr.io/nvidia/vllm:<tag> vllm serve …` in a
container. Both honor identical config keys. See
[`docs/ai-services.md`](../../../docs/ai-services.md#choosing-the-vllm-runtime-pip-vs-docker)
for prerequisites (Docker Engine, NVIDIA Container Toolkit, NGC auth) and
runtime details.

## Swap models

Edit `llama_nemotron_llm_server.yaml`:

```yaml
model: nvidia/Llama-3.3-Nemotron-Super-49B-v1   # example
```

Any HuggingFace model with a Llama-3.1-style chat template works under vLLM
with the `llama3_json` tool-call parser. Adjust `max_model_len`,
`gpu_memory_utilization`, and `tensor_parallel_size` to fit the new model.

## License

NVIDIA Open Model License + Llama 3.1 Community License. Commercial use
permitted. See the [model card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1)
for the full text.
