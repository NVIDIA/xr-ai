<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# nemotron3-nano-llm-server

OpenAI-compatible vLLM wrapper for Nemotron 3 Nano 30B-A3B. It selects NVFP4
on Blackwell and FP8 on earlier supported GPU architectures and serves port
8107.

```bash
uv run --project services/nemotron3-nano-llm nemotron3_nano_llm_server \
  --config services/nemotron3-nano-llm/nemotron3_nano_llm_server.yaml
```

See [AI inference servers](../../docs/source/components/ai-services.md#per-server-notes)
for model selection, tool calling, reasoning, hardware, persistence, and runtime
guidance. Exact configuration fields are generated from the adjacent YAML.
