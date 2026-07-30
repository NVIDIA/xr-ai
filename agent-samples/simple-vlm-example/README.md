<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

This sample answers voice and text questions against each participant's latest
camera frame. Responses stream to both Piper TTS and the `vlm.response` data
topic. Sending the literal text `ping` uses the configured default question,
`Describe what you see.`

The worker is a package under `worker/simple_vlm_example_worker/`:

- `__main__.py` parses launcher arguments.
- `config.py` resolves worker, model-overlay, voice-gate, and prompt settings.
- `app.py` composes the native runtime.
- `prompts/system.txt` owns the VLM system prompt.

`VoiceSession` owns STT/TTS/VLM readiness, the hub voice transport, voice-gate
processing, streaming TTS, signals, and cleanup. The application registers
`StreamingVisionConfig` in-process and adapts that native function with
`xr_ai_nat.adapters.as_voice_handler`. Typed text uses the same participant-aware
turn path as speech. Participant leave events release cached live-frame state,
and a newer turn cancels and interrupts a superseded response.

No MCP client or MCP tool invocation is part of this sample.

## Run

```bash
cd agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

Open the web client shown in the hub banner, connect, and then speak, type a
question, or send `ping`.

The default local endpoints are declared in `yaml/models.yaml`. Keep the
existing backend selection in `yaml/simple_vlm_example_worker.yaml`:

- `model_backend: local` loads the file named by `models_yaml`.
- `model_backend: nim` loads `yaml/models.nim.yaml` and skips the local VLM
  process while STT and TTS remain local.
- Point `models_yaml` at `models.omni.yaml` to use the shipped Omni overlay.

Voice-gate behavior remains in `yaml/voice_gate.yaml`. Worker timing, frame
freshness, the default `ping` question, and the package prompt path are in
`yaml/simple_vlm_example_worker.yaml`.
