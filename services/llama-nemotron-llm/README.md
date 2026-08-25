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

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="quickstart"></a><a id="endpoints"></a>
<a id="config-keys-llama_nemotron_llm_serveryaml"></a>
<a id="tool-calling-native-llama-31-format"></a>
<a id="reasoning-toggle-per-turn-via-system-prompt"></a>
<a id="choosing-the-vllm-runtime-pip-vs-docker"></a>
<a id="swap-models"></a><a id="license"></a>

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#per-server-notes)
for tool calling, reasoning, model swaps, persistence, and runtime guidance.
The adjacent YAML and its comments define the standalone configuration.
