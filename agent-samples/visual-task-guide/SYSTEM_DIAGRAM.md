<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Visual task guide system

```text
voice/text -> VoiceSession -> TaskGuideWorkflowConfig
                              | controls: start / next / reset / status
                              | state queries: deterministic current/next step
                              | validation: neutral visual count -> step check
                              | other questions: neutral visual count + RAG
                              v
camera -> XR Media Hub -> StreamingVisionConfig -> current-frame VLM
                                                      :8100 / reused
                              v
                       read-only NAT guide agent -> xr_rag NAT group
                              |                     -> RAG service :8340
                              |                     -> embedding :8109 / reused
                              |                     guide LLM :8106 / reused
                              v
                      voice + agent.response reply
```

Vision runs only on a user request. It receives no task target or expected
answer. Validation parses the structured count and compares it with the trusted
current step afterward. Direct count questions return that parsed observation;
other questions pass it to the read-only guide agent alongside bounded RAG.

Task state is `not_started`, `running`, or `completed`. Only the native task
control functions mutate it. Visual results are question evidence only, so
they never advance the task. Progress is session-local and resets on reconnect.
