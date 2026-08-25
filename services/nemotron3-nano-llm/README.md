<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# nemotron3-nano-llm-server

OpenAI-compatible vLLM wrapper for Nemotron 3 Nano 30B-A3B. It selects NVFP4
on Blackwell and FP8 on earlier supported GPU architectures and serves port
8107.

Run this command from the repository root:

```bash
uv run --project services/nemotron3-nano-llm nemotron3_nano_llm_server \
  --config services/nemotron3-nano-llm/nemotron3_nano_llm_server.yaml
```

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="quickstart"></a><a id="endpoints"></a>
<a id="config-keys-nemotron3_nano_llm_serveryaml"></a>
<a id="tool-calling-native-qwen3-coder-format"></a>
<a id="reasoning-mode-thinking"></a>
<a id="sampling-recommendations-from-the-model-card"></a>
<a id="hardware-notes"></a><a id="swap-models"></a><a id="notes"></a>
<a id="license"></a>

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#per-server-notes)
for model selection, tool calling, reasoning, hardware, persistence, and runtime
guidance. The adjacent YAML and its comments define the standalone
configuration.
