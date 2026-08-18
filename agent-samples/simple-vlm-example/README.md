<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

This sample answers voice and text questions against each participant's latest
camera frame. Responses stream to the selected TTS backend and the
`vlm.response` data topic.

The worker is a package under `worker/simple_vlm_example_worker/`:

- `__main__.py` parses launcher arguments.
- `agent.py` owns participant-scoped vision turns and cancellation.
- `config.py` resolves worker, model-profile, voice-gate, and prompt settings.
- `app.py` composes the native runtime.
- `prompts/system.txt` owns the VLM system prompt.

`VoiceAgent` privately owns STT/TTS/VLM readiness, the hub voice transport,
voice-gate processing, streaming TTS, signals, and cleanup. It publishes every
final pre-gate STT result on `voice.transcript`, including speech without the
wake phrase, and publishes accepted speech and typed text as `UserQuery` on
this sample's topic. `SimpleVlmAgent` subscribes to that topic, selects the participant's
current image with `CurrentFrameTool`, passes its opaque reference to the
transport-independent `StreamingImageQueryTool`, and publishes chunks to
`voice.output`. The query tool has no voice dependency and sends its provider
stream through Relay's managed LLM path. Camera bytes stay out of tool results
and image locations are redacted from VLM telemetry. `VoiceAgent` publishes
participant departure and interruption on sample-named topics;
`SimpleVlmAgent` subscribes and releases its own cached frames and tasks. A
newer turn cancels and interrupts a superseded response. `app.py` only composes
the two agents and their dependencies.

No MCP client or MCP tool invocation is part of this sample.

## Run

```bash
cd agent-samples/simple-vlm-example
uv sync
uv run main.py --piper
```

Open the web client shown in the hub banner, connect, and then speak or type a
question.

`--piper` uses the existing CPU Piper service and remains the default when no
TTS flag is supplied. `--magpie` starts Magpie Multilingual NIM 1.9.0 on the
GPU and consumes its online PCM response as it is generated:

```bash
uv run main.py --magpie
```

Magpie requires Docker, NVIDIA Container Toolkit, an `NGC_API_KEY`, and a GPU
with enough free memory for the VLM, STT, and TTS NIM. The supplied profile is
tuned for one 96 GB GPU. Its first launch downloads the NIM image and exports
an optimized model store under `models/nim-magpie-tts/`, which can take about
20 minutes; warm starts normally take under a minute. Piper does not require
the Magpie image, NGC credentials, or GPU memory.

The worker and orchestrator consume the deployment profile selected by the
active worker YAML:

- `simple_vlm_example_worker.yaml` and `models.local.json` keep the Piper path.
- `simple_vlm_example_worker.magpie.yaml` and `models.local.magpie.json` select
  streaming Magpie NIM. The Magpie worker config inserts 240 ms of silence
  between separately synthesized sentences without delaying the first one.
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
mark, and each sentence TTS request is a `voice.tts` scope. Raw audio is summarized
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
Worker timing, frame freshness, and optional prompt overrides are in the
selected worker YAML; the default system prompt ships inside the worker
package.
