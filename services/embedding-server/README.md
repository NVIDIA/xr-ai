<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# embedding-server

OpenAI-compatible vLLM wrapper for
`nvidia/llama-nemotron-embed-1b-v2` on port 8109.

Run this command from the repository root:

```bash
uv run --project services/embedding-server embedding_server \
  --config services/embedding-server/embedding_server.yaml
```

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#per-server-notes)
for endpoints, dimensions, deployment, cache, and runtime guidance. The
adjacent `embedding_server.yaml` and its comments define the standalone
configuration.
