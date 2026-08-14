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
- `agent.py` owns participant-scoped vision turns and cancellation.
- `config.py` resolves worker, model-profile, voice-gate, and prompt settings.
- `app.py` composes the native runtime.
- `prompts/system.txt` owns the VLM system prompt.

`VoiceAgent` owns `VoiceSession`, which provides STT/TTS/VLM readiness, the hub
voice transport, voice-gate processing, streaming TTS, signals, and cleanup.
It publishes accepted speech and typed text as `UserQuery` on this sample's
topic. `SimpleVlmAgent` subscribes to that topic, owns participant-scoped
streaming and cancellation around the transport-independent
`StreamingVisionTool`, and publishes chunks to `voice.output`. The tool has no
voice dependency and sends its provider stream through Relay's managed LLM
path. The camera frame is redacted from Relay telemetry. `VoiceAgent` publishes
participant departure and interruption on sample-named topics;
`SimpleVlmAgent` subscribes and releases its own cached frames and tasks. A
newer turn cancels and interrupts a superseded response. `app.py` only composes
the two agents and their dependencies.

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

## Relay visibility

The worker writes a compact Relay lifecycle stream to `relay-events.jsonl`
beside `worker.log` in the per-run log directory printed at startup. The JSONL
records include runtime publications, receiving-agent callbacks, the complete
`simple-vlm.turn` lifetime, and nested vision tool and VLM calls. Per-token
`llm.chunk` marks, incremental `voice.output` fragments, and empty stream
terminators are omitted. `VoiceAgent` emits one `voice.response` scope containing
the complete text and timing for both non-streamed and aggregated incremental
output. Each real STT request is a `voice.stt` scope with a transcript result
mark, and each sentence synthesis is a `voice.tts` scope. Raw audio is summarized
by byte count, duration, and sample rate; TTS records synthesis rather than
client playback. The completed LLM and turn records remain available alongside
them. No telemetry server or network exporter is required. Live camera bytes are
replaced with `<redacted:live-camera-frame>`; prompts, questions, responses,
participant IDs, and correlation metadata remain visible and may contain
sensitive data.

```bash
tail -F /tmp/log_simple-vlm-example_*/relay-events.jsonl
```

Voice-gate behavior remains in `yaml/voice_gate.yaml`. Wake phrases match at
the start of the transcript or after sentence-final `.`, `?`, or `!` followed
by whitespace or a closing quote; preceding text is discarded before dispatch.
Worker timing, frame freshness, the default `ping` question, and optional prompt
overrides are in `yaml/simple_vlm_example_worker.yaml`; the default prompt ships
inside the worker package.
