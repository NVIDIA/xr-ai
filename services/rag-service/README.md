<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# RAG service

Indexes Markdown and text documents and exposes dense retrieval over private
msgpack/ZMQ. Applications access it through the native `xr_rag` NAT function
group; the service transport is not an agent-facing API.

Configure `documents_dir`, `models_config`, and `endpoint` in a copied
`rag_service.yaml`. The referenced model profile needs an `embedding` role:

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
