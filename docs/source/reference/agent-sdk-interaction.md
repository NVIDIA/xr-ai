<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-interaction

`xr-ai-interaction` provides the transport-neutral messages exchanged while a
participant interacts with agents: `UserQuery`, `ResponseOutput`,
`ParticipantJoined`, `ParticipantLeft`, and `Interrupted`. It depends only on
Pydantic.

Input and output boundaries adapt these messages to a medium. `VoiceAgent`
publishes recognized speech as `UserQuery` and speaks `ResponseOutput` from
`VOICE_OUTPUT_TOPIC`; a text front end can publish and render the same
messages. Routing and conversation use them without depending on voice.
Applications choose topic names; the models define only the payloads.

## Python API reference

The generated {doc}`Python reference </reference/python/xr_ai_interaction/index>`
documents the exported names and their fields and validation.
