<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-conversation

Routable conversation agent with streamed voice output with participant-scoped background context
and rolling-summary compaction. Live-model evals are in [`eval/`](eval/).

Refer to the [conversation SDK reference](https://nvidia.github.io/xr-ai/latest/reference/agent-sdk-conversation.html)
for the conversation, context, and lifecycle contracts. Routing is provided by
the separate `xr-ai-routing` package.
