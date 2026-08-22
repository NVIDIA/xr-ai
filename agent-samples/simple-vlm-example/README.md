<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

This sample answers voice and text questions against each participant's latest
camera frame. Responses stream to both Piper TTS and the `vlm.response` data
topic.
See the [sample guide](../../docs/source/reference/simple-vlm-example.md) for
architecture, configuration, warmup behavior, voice gating, and Relay output.

## Run

The sample reuses model services and never starts or stops them. Its fixed
`yaml/models.json` expects Parakeet STT on port 8103, Cosmos3-Nano on port
8100, and Piper TTS on port 8105. Start compatible services before the sample.
For the repository defaults, run these from the repository root (the Piper
command stays in the foreground):

```bash
uv run --project agent-samples/model-servers model_servers
uv run --project services/piper-tts piper_tts_server \
  --config services/piper-tts/piper_tts_server.yaml
```

Then, in another terminal:

```bash
cd agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

Open the web client shown in the hub banner, connect, and then speak or type a
question.
