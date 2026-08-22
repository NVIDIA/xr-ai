<!--
 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

# embedding-server

OpenAI-compatible vLLM wrapper for
`nvidia/llama-nemotron-embed-1b-v2` on port 8109.

```bash
uv run --project services/embedding-server embedding_server \
  --config services/embedding-server/embedding_server.yaml
```

Refer to [AI inference servers](../../docs/source/components/ai-services.md#per-server-notes)
for endpoints, dimensions, deployment, cache, and runtime guidance. Exact
configuration fields are generated from `embedding_server.yaml`.
