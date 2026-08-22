<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

This sample is the smallest complete voice-and-vision application in the
repository. A participant can ask a spoken or typed question about the latest
camera frame. The response streams to Piper TTS and to the `vlm.response` data
topic.

The sample launches DeviceIOHub and its worker. It reuses Parakeet STT,
Cosmos3 Nano Reasoner, and Piper TTS from the shared model stack. Before the
agent reports ready, the worker sends a representative image request through
the VLM so the first user query does not pay the multimodal warmup cost.

## Run

Run all commands from `agent-samples/simple-vlm-example/`. Start the shared
models first:

```bash
uv run --project ../model-servers model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample from the same terminal:

```bash
uv sync
uv run simple_vlm_example
```

Open the authenticated web-client URL printed by DeviceIOHub, allow microphone
and camera access, and connect. Speak or type a question after the agent reports
ready.

The shared models remain running after the sample stops. From this directory,
stop them with `uv run --project ../model-servers model_servers --stop`.

Refer to the [sample reference](../../docs/source/reference/simple-vlm-example.md)
for architecture, configuration, warmup behavior, voice gating, and Relay
output.
