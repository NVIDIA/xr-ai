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

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="quickstart"></a><a id="endpoints"></a>
<a id="config-keys-embedding_serveryaml"></a>
<a id="matryoshka-dimensions"></a><a id="example-request"></a>
<a id="choosing-the-vllm-runtime-pip-vs-docker"></a>

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html#per-server-notes)
for endpoints, dimensions, deployment, cache, and runtime guidance. The
adjacent `embedding_server.yaml` and its comments define the standalone
configuration.
