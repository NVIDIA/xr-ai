<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Release migration

This release removes deprecated SDK aliases and the standalone Pipecat
compatibility package. Update out-of-tree code as follows:

| Removed surface | Replacement |
|---|---|
| `xr_ai_agent` | Import `ProcessorEndpoint` and IPC types from `xr_ai_hub`. |
| `xr_ai_pipecat` | Compose `xr_ai_voice.VoiceSession` and `VoiceAgent` with `xr_ai_runtime.AgentRuntime`. Pipecat remains private to `xr-ai-voice`. |
| `BrainProcessor` and `make_voice_pipeline` | Put application behavior in an `Agent` subscriber and let `VoiceAgent` own the voice pipeline. |
| `xr_ai_models.config`, `factory`, `openai_compat`, and `protocols` | Import public names directly from `xr_ai_models`. This includes `KIND_OPENAI_COMPAT`, `ModelKind`, `Category`, and `Spec`. |

The source directories now match their Python imports:
`agent-sdk/xr-ai-hub-client/` became `agent-sdk/xr-ai-hub/` and
`agent-sdk/xr-ai-agent-runtime/` became `agent-sdk/xr-ai-runtime/`. Distribution
names remain `xr-ai-hub-client` and `xr-ai-agent-runtime`, so package dependency
names do not change.

If upgrading a checkout that already downloaded model weights, follow
{ref}`the model-cache migration <migrating-model-caches-from-ai-services>` to
reuse the ignored caches rather than downloading them again.
