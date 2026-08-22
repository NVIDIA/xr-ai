<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# RAG service

Private typed msgpack/ZMQ service that indexes Markdown and text documents with
an `xr-ai-models` embedding endpoint and serves dense retrieval to
`xr_ai_tools.rag.RAGTools`.

```bash
uv run --project services/rag-service rag_service \
  --config agent-samples/my-agent/yaml/rag_service.yaml
```

See [AI inference servers](../../docs/source/components/ai-services.md#per-server-notes)
for startup order, profile requirements, caching, and integration guidance.
