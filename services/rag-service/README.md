<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# RAG service

Private typed msgpack over ZMQ service that indexes Markdown and text documents with
an `xr-ai-models` embedding endpoint and serves dense retrieval to
`xr_ai_tools.rag.RAGTools`.

Run this command from the repository root after starting the embedding service
selected by the sample's model profile:

```bash
uv run --project services/rag-service rag_service \
  --config agent-samples/tea-making-sample/yaml/rag_service.yaml
```

That sample configuration points `documents_dir` at its application-owned
documents and `models_config` at a profile with an embedding role. Set both
fields when adapting the service to another application.

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#per-server-notes)
for startup order, profile requirements, caching, and integration guidance.
