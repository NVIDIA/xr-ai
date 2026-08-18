<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Services

Reusable runtime, model-serving, and typed capability processes live here as
direct child projects. A common depth keeps process declarations, editable
dependency paths, operational ownership, and standalone commands predictable.

| Service | Responsibility | Default port or endpoint |
|---|---|---|
| `xr-media-hub/` | XR-Media-Hub and its internal LiveKit transport | 8080 (HTTPS and WSS proxy) |
| `cloudxr-runtime/` | CloudXR OpenXR runtime and WebRTC signaling proxy | 49100 (native), 48322 (WSS) |
| `vlm-server/` | Cosmos vision-language inference | 8100 |
| `stt-server/` | Parakeet speech recognition | 8103 |
| `magpie-tts/` | Magpie speech synthesis | 8104 |
| `magpie-tts-nim/` | Streaming Magpie synthesis through NVIDIA NIM | 9000 |
| `piper-tts/` | Piper speech synthesis | 8105 |
| `llama-nemotron-llm/` | Llama Nemotron text generation | 8106 |
| `nemotron3-nano-llm/` | Nemotron 3 Nano text generation | 8107 |
| `nemotron-omni-llm/` | Nemotron Omni multimodal generation | 8108 |
| `embedding-server/` | Nemotron text embeddings | 8109 |
| `video-memory-service/` | Recorded-video queries and historical frame decoding | `tcp://localhost:8310` |
| `openxr-service/` | Headless OpenXR session and current head pose | `tcp://localhost:8330` |
| `rag-service/` | Dense document retrieval | `tcp://localhost:8340` |

Model servers are consumed through `xr-ai-models`; most expose OpenAI-compatible
HTTP, while Magpie NIM uses its typed offline and online synthesis endpoints.
XR capability services expose typed RPC and are consumed by native tools in
`xr-ai-tools`.

Sample-specific orchestrators and workers remain together under
`agent-samples/`; they are application processes rather than reusable
services.
