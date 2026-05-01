<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# nemotron3-nano-llm-server

OpenAI-compatible LLM server for
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4),
implemented as a thin launcher around vLLM.

A 30B-total / 3.5B-active hybrid Mamba-Transformer MoE reasoning + tool-calling
model from NVIDIA, quantized to NVFP4 (weights) with FP8 KV cache. Runs on a
single Blackwell-class GPU via FlashInfer's FP4 MoE kernels. Commercial use
is permitted under the NVIDIA Nemotron Open Model License.

## Quickstart

```bash
cd ai-services/llm/nemotron3_nano
uv sync
uv run nemotron3_nano_llm_server --config nemotron3_nano_llm_server.yaml
```

First run downloads ~17 GB of NVFP4 weights and the `nano_v3_reasoning_parser.py`
plugin into the shared `models/` cache at the repo root. Subsequent runs start
offline.

## Endpoints

All endpoints are provided by vLLM's OpenAI-compatible server at
`http://localhost:8107`:

| Endpoint               | Method | Description                                      |
|------------------------|--------|--------------------------------------------------|
| `/health`              | GET    | Server health                                    |
| `/v1/models`           | GET    | List models (returns `{"id": "llm", ...}`)       |
| `/v1/chat/completions` | POST   | Chat completion with `tools=[...]` support       |

## Config keys (`nemotron3_nano_llm_server.yaml`)

| Key                    | Type | Default                                                | Description                                                                    |
|------------------------|------|--------------------------------------------------------|--------------------------------------------------------------------------------|
| `model`                | str  | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`          | HuggingFace model ID                                                           |
| `host`                 | str  | `0.0.0.0`                                              | Bind address                                                                   |
| `port`                 | int  | `8107`                                                 | HTTP port                                                                      |
| `hf_token`             | str  | `""`                                                   | HuggingFace token for gated models                                             |
| `model_cache`          | str  | `../../models`                                         | Weight + parser cache (relative to YAML)                                       |
| `max_num_seqs`         | int  | `8`                                                    | vLLM concurrent request limit                                                  |
| `tensor_parallel_size` | int  | `1`                                                    | vLLM TP — raise for multi-GPU serving                                          |
| `max_model_len`        | int  | `32768`                                                | Max context tokens                                                             |
| `gpu_memory_utilization` | float | `0.6`                                                | vLLM `--gpu-memory-utilization`. Lowered from vLLM's default (0.92) so this LLM can share a GPU with vlm-server + stt-server; raise toward 0.85–0.92 on a dedicated GPU. |
| `enforce_eager`        | bool | `true`                                                 | Passes vLLM `--enforce-eager`. Skips CUDA graph capture + FlashInfer MoE autotune, which for this model are silent and take 3–8 min on first run. Eager mode starts in ~5 s after weight load and is 10–20% slower per token — negligible for voice-agent workloads. Set `false` for maximum steady-state throughput if you can tolerate the longer startup. |

## Tool calling (native Qwen3-Coder format)

Tool calling is handled entirely server-side by vLLM's `qwen3_coder` parser.
Clients send OpenAI-shape `tools=[...]` in the chat-completions request:

- Request: `{messages: [...], tools: [{type: "function", function: {...}}]}`
- Model emits: `<tool_call><function=name><parameter=foo>bar</parameter></function></tool_call>`
- Server parses and returns: `choices[0].message.tool_calls = [{id, type, function: {name, arguments}}]`, `finish_reason = "tool_calls"`

No client-side parsing needed.

## Reasoning mode (`thinking`)

The model's chat template defaults to `enable_thinking=True`, so every response
begins with a `<think>…</think>` preamble followed by the answer / tool call.
vLLM's `--reasoning-parser nano_v3` (auto-downloaded from the model card) splits
this cleanly:

- `message.reasoning_content` — the think text (LangChain / OpenAI clients ignore this field)
- `message.content` — the actual spoken answer
- `message.tool_calls` — structured tool call when the model chose to call

To disable thinking per-request, clients can send
`extra_body={"chat_template_kwargs": {"enable_thinking": false}}`.

## Sampling recommendations (from the model card)

- **Tool calling**: `temperature=0.6, top_p=0.95`
- **Reasoning (thinking-on, pure chat)**: `temperature=1.0, top_p=1.0`

Our primary use case is tool calling, so agent configs should typically set
`temperature=0.6, top_p=0.95`. Thinking still runs — it's just less diverse
token sampling inside the `<think>` block (and the think content is never
shown to the end user anyway).

## Hardware notes

- **Required for native FP4 compute**: Blackwell (B200, RTX PRO 6000, Jetson
  Thor, DGX Spark — all listed on the model card's test hardware line).
  FlashInfer's FP4 MoE kernels target these architectures.
- **On Hopper (H100) / Ampere (A100)**: FlashInfer will emulate FP4 or fail to
  load the kernel. For those targets swap the model ID to the BF16 variant
  `nvidia/Nemotron-Nano-3-30B-A3B` — same architecture, same tool-call parser,
  same reasoning plugin, ~60 GB VRAM at BF16 instead of ~20 GB at NVFP4.
- **VRAM at NVFP4**: ~20 GB (30B params × 4 bits weights + FP8 KV cache +
  activations). Fits comfortably on a single B200 or RTX PRO 6000.

## Swap models

Edit `nemotron3_nano_llm_server.yaml`:

```yaml
model: nvidia/Nemotron-Nano-3-30B-A3B          # BF16 unquantized
# model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8    # FP8 (Hopper-compatible)
```

Any NVIDIA Nemotron-3 family variant works because they share the same chat
template, `qwen3_coder` tool-call format, and `nano_v3` reasoning parser.

## Notes

- **No custom modeling code in this repo** — all inference logic lives in vLLM.
  Compare to the ~750-line `llama_nemotron_llm_server` which hand-rolls
  template rendering, LMFE grammar, and tool-call parsing. vLLM does all three.
- **`trust_remote_code=True`** — required by the Nemotron-3-Nano model (custom
  modeling classes). Ships directly from NVIDIA.
- **`execvp` design** — this wrapper is intentionally thin. It reads config,
  fetches the reasoning-parser plugin, sets FlashInfer env vars, and hands
  off to `vllm serve` via `os.execvp`. The launcher's signal forwarding goes
  straight to vLLM with no extra wiring.

## License

NVIDIA Nemotron Open Model License. Commercial use permitted.
See the [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
for the full text.
