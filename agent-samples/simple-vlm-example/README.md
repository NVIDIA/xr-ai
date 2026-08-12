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
- `config.py` resolves worker, model-profile, voice-gate, and prompt settings.
- `app.py` composes the native runtime.
- `prompts/system.txt` owns the VLM system prompt.

`VoiceSession` owns STT/TTS/VLM readiness, the hub voice transport, voice-gate
processing, streaming TTS, signals, and cleanup. The application invokes the
native `LiveVisionResponder` directly; it acquires the participant's current
frame and streams an injected VLM through Relay's managed LLM path. The camera
frame is redacted from Relay telemetry. Typed text uses the same
participant-aware turn path as speech. Participant leave events release cached
live-frame state, and a newer turn cancels and interrupts a superseded response.

No MCP client or MCP tool invocation is part of this sample.

## Run

```bash
cd agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

Open the web client shown in the hub banner, connect, and then speak, type a
question, or send `ping`.

The worker and orchestrator consume the deployment profile selected by
`models_config` in `yaml/simple_vlm_example_worker.yaml`:

- `models.local.json` manages local STT, VLM, and TTS services.
- `models.hosted.json` uses hosted NVIDIA NIM for VLM and omits the local VLM
  process.
- `models.omni.json` reuses the Nemotron-Omni VLM service on port 8108.

The same profile owns model behavior, endpoints, credentials, readiness, and
launcher process ownership.

Voice-gate behavior remains in `yaml/voice_gate.yaml`. Worker timing, frame
freshness, the default `ping` question, and optional prompt overrides are in
`yaml/simple_vlm_example_worker.yaml`; the default prompt ships inside the
worker package.
