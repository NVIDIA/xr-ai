<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance

This sample combines a foreground tea guide with independent background
observers (transcripts, visual-change watching, periodic video observations).
See the [sample guide](../../docs/source/reference/tea-making-sample.md) for
architecture, configuration, output contracts, safety, and adaptation guidance.

## Run it

Start the reusable model services, then start Piper TTS in a terminal:

```bash
uv run --project agent-samples/model-servers model_servers
uv run --project services/piper-tts piper_tts_server \
  --config services/piper-tts/piper_tts_server.yaml
```

Then start the tea stack from the repository root:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample

# Allow direct event-viewer access from a trusted private network.
uv run --project agent-samples/tea-making-sample tea_making_sample \
  --expose-web-events
```

Open `https://localhost:8080`, accept the self-signed certificate on first use,
allow microphone and camera access, and connect.
The optional event viewer is `http://127.0.0.1:8092`. Begin voice commands with
“Agent” or “Hey Agent.”
