<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# embedding-server

OpenAI-compatible embeddings server for
[`nvidia/llama-nemotron-embed-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2),
implemented as a thin launcher around vLLM.

A 1B-parameter Matryoshka embedding model from NVIDIA. Supports output
dimensions of 384 / 512 / 768 / 1024 / 2048 via prefix truncation.
Used by `rag-mcp-server` for dense semantic retrieval. (~2 GB VRAM at BF16,
low `gpu_memory_utilization` default leaves headroom for co-resident VLM/LLM
servers.)

## Quickstart

```bash
cd ai-services/embedding-server
uv sync
uv run embedding_server --config embedding_server.yaml
```

First run downloads weights (~2 GB) to the shared `models/` cache at the repo
root. Subsequent runs start offline.

## Endpoints

All endpoints are provided by vLLM's OpenAI-compatible server at
`http://localhost:8109`:

| Endpoint          | Method | Description                                      |
|-------------------|--------|--------------------------------------------------|
| `/health`         | GET    | Server health                                    |
| `/v1/models`      | GET    | List models (returns `{"id": "embed", ...}`)     |
| `/v1/embeddings`  | POST   | Embed text (OpenAI `/v1/embeddings` format)      |

## Config keys (`embedding_server.yaml`)

| Key                      | Type  | Default                               | Description                                                   |
|--------------------------|-------|---------------------------------------|---------------------------------------------------------------|
| `model`                  | str   | `nvidia/llama-nemotron-embed-1b-v2`   | HuggingFace model ID                                          |
| `host`                   | str   | `0.0.0.0`                             | Bind address                                                  |
| `port`                   | int   | `8109`                                | HTTP port                                                     |
| `hf_token`               | str   | `""`                                  | HuggingFace token for gated models                            |
| `model_cache`            | str   | `../../../models`                     | Weight cache (relative to YAML)                               |
| `max_num_seqs`           | int   | `32`                                  | vLLM concurrent request limit                                 |
| `tensor_parallel_size`   | int   | `1`                                   | vLLM TP — raise for multi-GPU serving                         |
| `max_model_len`          | int   | `8192`                                | Max input tokens (model native max)                           |
| `gpu_memory_utilization` | float | `0.20`                                | vLLM `--gpu-memory-utilization` (low: model is only 1B)       |
| `enforce_eager`          | bool  | `false`                               | Skip CUDA graph capture                                       |
| `embedding_dim`          | int   | `768`                                 | Matryoshka dim for consumers — **not passed to vLLM**         |
| `vllm_backend`           | str   | `pip`                                 | `pip` (wrapper venv) or `docker` (NGC container)              |
| `vllm_image`             | str   | `nvcr.io/nvidia/vllm:26.04-py3`       | Used when `vllm_backend: docker`                              |

## Matryoshka dimensions

vLLM always returns 2048-dimensional vectors. Consumers must truncate to
`embedding_dim` before comparison. The first N dimensions of a Matryoshka
vector are a valid embedding at dimension N, so `vec[:768]` is a proper
768-dim embedding.

## Example request

```bash
curl -s http://localhost:8109/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "embed", "input": "query: how do I reset the device?"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['data'][0]['embedding']))"
# → 2048  (truncate to embedding_dim in the consumer)
```

## Choosing the vLLM runtime (pip vs Docker)

`vllm_backend: pip` (default) runs vLLM from the wrapper's venv;
`vllm_backend: docker` runs `nvcr.io/nvidia/vllm:<tag> vllm serve …` in a
container. See
[`docs/ai-services.md`](../../../docs/ai-services.md#choosing-the-vllm-runtime-pip-vs-docker)
for prerequisites.
