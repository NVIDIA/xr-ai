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
| `device-io-hub/` | DeviceIOHub and its internal LiveKit transport | 8080 (HTTPS and WSS proxy) |
| `cloudxr-runtime/` | CloudXR OpenXR runtime and WebRTC signaling proxy | 49100 (native), 48322 (WSS) |
| `vlm-server/` | Cosmos vision-language inference | 8100 |
| `stt-server/` | Parakeet speech recognition | 8103 |
| `magpie-tts/` | Magpie speech synthesis | 8104 |
| `piper-tts/` | Piper speech synthesis | 8105 |
| `llama-nemotron-llm/` | Llama Nemotron text generation | 8106 |
| `nemotron3-nano-llm/` | Nemotron 3 Nano text generation | 8107 |
| `nemotron-omni-llm/` | Nemotron Omni multimodal generation | 8108 |
| `embedding-server/` | Nemotron text embeddings | 8109 |
| `nim-server/` | Generic self-hosted NVIDIA NIM launcher | configured HTTP and optional gRPC ports |
| `video-memory-service/` | Recorded-video queries and historical frame decoding | `tcp://localhost:8310` |
| `openxr-service/` | Headless OpenXR session and current head pose | `tcp://localhost:8330` |
| `rag-service/` | Dense document retrieval | `tcp://localhost:8340` |

Local model wrappers expose OpenAI-compatible HTTP. A generic NIM exposes the
protocol of its selected image: LLM and VLM NIMs use OpenAI-compatible HTTP,
while speech NIMs use Riva gRPC. Workers consume both through `xr-ai-models`.
XR capability services expose typed RPC and are consumed by native tools in
`xr-ai-tools`.

Sample-specific orchestrators and workers remain together under
`agent-samples/`; they are application processes rather than reusable
services.

Refer to [AI inference servers](https://nvidia.github.io/xr-ai/latest/components/ai-services.html) for model
deployment, persistence, NIM, caches, and per-service operational guidance.
