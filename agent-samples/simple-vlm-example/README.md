<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

For a walkthrough of running this sample, see
[`docs/source/getting_started/quickstart.md`](../../docs/source/getting_started/quickstart.md).

This sample answers voice and text questions against each participant's latest
camera frame. Responses stream to both Piper TTS and the `vlm.response` data
topic.

The worker is a package under `worker/simple_vlm_example_worker/`:

- `__main__.py` parses launcher arguments.
- `agent.py` owns participant-scoped vision turns and cancellation.
- `config.py` resolves worker, model, voice-gate, and prompt settings.
- `app.py` composes the native runtime (`VoiceAgent` + `SimpleVlmAgent`).
- `prompts/system.txt` owns the VLM system prompt.

No MCP client or MCP tool invocation is part of this sample.

## Run

The sample reuses model services and never starts or stops them. Its fixed
`yaml/models.json` expects Parakeet STT on port 8103, Cosmos3-Nano on port
8100, and Piper TTS on port 8105. Start compatible services before the sample.
For the repository defaults, run these from the repository root (the Piper
command stays in the foreground):

```bash
uv run --project agent-samples/model-servers model_servers
uv run --project services/piper-tts piper_tts_server
```

Then, in another terminal:

```bash
cd agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

Open the web client shown in the hub banner, connect, and then speak or type a
question.

`yaml/models.json` owns model behavior, endpoints, and readiness. All model
deployments are `reused`; the orchestrator launches only the DeviceIOHub and
worker. Before announcing readiness, the worker completes a small streaming
request with a 1280x720 JPEG so the first user query does not pay the
multimodal initialization cost.

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
them. No telemetry server or network exporter is required. Image locations are
replaced with `<redacted:image>`; prompts, questions, responses, participant
IDs, and correlation metadata remain visible and may contain sensitive data.
Opaque live-frame handles remain small even when the source frame is large.

```bash
tail -F /tmp/log_simple-vlm-example_*/relay-events.jsonl
```

Voice-gate behavior remains in `yaml/voice_gate.yaml`. Wake phrases match at
the start of the transcript or after sentence-final `.`, `?`, or `!` followed
by whitespace or a closing quote; preceding text is discarded before dispatch.
Worker timing, frame freshness, and optional prompt overrides are in
`yaml/simple_vlm_example_worker.yaml`; the default system prompt ships inside
the worker package.
