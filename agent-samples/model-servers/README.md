<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Shared model servers

Starts the reusable STT, LLM/VLM, and embedding services selected by a model
deployment profile. The services persist after the orchestrator exits so sample
restarts reuse warm weights.

```bash
uv run --project agent-samples/model-servers model_servers
```

To select the self-hosted NIM profile or stop every persisted model service:

```bash
uv run --project agent-samples/model-servers model_servers --models vlm_llm_nim
uv run --project agent-samples/model-servers model_servers --stop
```

See [Customizing model servers](../../docs/source/guides/customizing-model-servers.md)
for deployment and hardware profiles, GPU placement, credentials, persistence,
and custom configurations.
