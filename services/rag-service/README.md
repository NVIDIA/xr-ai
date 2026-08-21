<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# RAG service

Indexes Markdown and text documents and exposes dense retrieval through private
RPC using msgpack over ZMQ. Applications construct `xr_ai_tools.rag.RAGTools`
with the private endpoint; the transport itself is not an agent-facing API.

The checked-in `rag_service.yaml` is a reference configuration. Copy it into
the consuming application's `yaml/` directory, set `documents_dir` and
`models_config` relative to that copy, and launch the service with the explicit
configuration path:

```bash
uv run --project services/rag-service rag_service \
  --config agent-samples/my-agent/yaml/rag_service.yaml
```

The referenced model profile needs an `embedding` role:

```json
{
  "models": {
    "embedding": {
      "adapter": {"preset": "nemotron_embedding"},
      "endpoint": {"base_url": "http://localhost:8109"},
      "deployment": {"ownership": "reused", "service": "embedding"}
    }
  }
}
```

At startup, the service recursively loads `.md` and `.txt` files, chunks and
embeds changed content, then touches its ready file. Embeddings are cached by
document content, indexing settings, and model profile. Set `cache_key` when a
remote endpoint changes its backing model without changing the profile.
`min_score` filters unrelated passages before results reach an agent.

Launch the embedding service first, this service second, and the consuming
worker last. Construct `RAGTools` in the worker and expose its finite tools to
the agent; application code should not call the private transport directly.
