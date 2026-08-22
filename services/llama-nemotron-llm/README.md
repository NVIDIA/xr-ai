<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# llama-nemotron-llm-server

OpenAI-compatible vLLM wrapper for
`nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on port 8106.

```bash
uv run --project services/llama-nemotron-llm llama_nemotron_llm_server \
  --config services/llama-nemotron-llm/llama_nemotron_llm_server.yaml
```

Refer to [AI inference servers](../../docs/source/components/ai-services.md#per-server-notes)
for tool calling, reasoning, model swaps, persistence, and runtime guidance.
Exact configuration fields are generated from the adjacent YAML.
