<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# nemotron35-lightning-llm-server

OpenAI-compatible vLLM wrapper for the text-only Nemotron 3.5 Lightning
30B-A3B model. It serves port 8108 with tool calling and parsed reasoning.

Run this command from the repository root:

```bash
uv run --project services/nemotron35-lightning-llm \
  nemotron35_lightning_llm_server \
  --config services/nemotron35-lightning-llm/nemotron35_lightning_llm_server.yaml
```

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#per-server-notes)
for model behavior, persistence, and runtime guidance. The adjacent YAML and
its comments define the standalone configuration.
