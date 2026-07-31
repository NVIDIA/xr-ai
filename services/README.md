<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Services

Reusable model-serving and typed capability processes live here as direct
child projects. A common depth keeps process declarations, editable dependency
paths, operational ownership, and standalone commands predictable.

| Service | Responsibility | Default port or endpoint |
|---|---|---|
| `vlm-server/` | Cosmos vision-language inference | 8100 |
| `stt-server/` | Parakeet speech recognition | 8103 |
| `magpie-tts/` | Magpie speech synthesis | 8104 |
| `piper-tts/` | Piper speech synthesis | 8105 |
| `llama-nemotron-llm/` | Llama Nemotron text generation | 8106 |
| `nemotron3-nano-llm/` | Nemotron 3 Nano text generation | 8107 |
| `nemotron-omni-llm/` | Nemotron Omni multimodal generation | 8108 |
| `video-memory-service/` | Recorded-video queries and historical frame decoding | `tcp://localhost:8310` |
| `openxr-service/` | Headless OpenXR session and current head pose | `tcp://localhost:8330` |

Model servers expose OpenAI-compatible HTTP and are consumed through
`xr-ai-models`. XR capability services expose typed RPC and are consumed by
native functions in `xr-ai-nat`. MCP compatibility adapters remain under
`agent-mcp-servers/`.

Sample-specific orchestrators and workers remain together under
`agent-samples/`; they are application processes rather than reusable
services.

The move from `ai-services/` changes where six standalone reference YAMLs
resolve their ignored model caches. Before deleting a legacy checkout, follow
the [cache migration procedure](../docs/ai-services.md#migrating-standalone-caches-from-ai-services)
to merge existing weights into the repository-root `models/` directory.
