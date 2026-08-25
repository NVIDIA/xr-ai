<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# llama-nemotron-llm-server

OpenAI-compatible vLLM wrapper for
`nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on port 8106.

Run this command from the repository root:

```bash
uv run --project services/llama-nemotron-llm llama_nemotron_llm_server \
  --config services/llama-nemotron-llm/llama_nemotron_llm_server.yaml
```

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#per-server-notes)
for tool calling, reasoning, model swaps, persistence, and runtime guidance.
The adjacent YAML and its comments define the standalone configuration.
