<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

This sample answers voice and text questions against each participant's latest
camera frame. Responses stream to both Piper TTS and the `vlm.response` data
topic.

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
uv run simple_vlm_example
```

For the local deployment, startup first prints a strict per-GPU VRAM preflight.
The existing `vlm_server.yaml` and `stt_server.yaml` files own their GPU placement,
ports, and absolute GiB reservations. The shared hardware `gpu_profile.yaml` owns
the device safety reserve, and vLLM utilization is derived from the detected
physical total. See the requirements guide for measurement and certification.

`HF_TOKEN` is required by default; pass `--allow-anonymous` to run without one
(see [`credentials.md`](../../docs/source/getting_started/credentials.md)).

The VLM and STT keep running after you exit so the next run skips the model
reload; free the VRAM with `cd ../model-servers && uv run model_servers --stop`.

Open the web client shown in the hub banner, connect, and then speak or type a
question.

The worker and orchestrator consume the deployment profile selected by
`models_config` in `yaml/simple_vlm_example_worker.yaml`:

- `models.local.json` manages local STT, VLM, and TTS services.
- `models.hosted.json` uses hosted NVIDIA NIM for VLM and omits the local VLM
  process.
- `models.omni.json` reuses the Nemotron-Omni VLM service on port 8108.

The same profile owns model behavior, endpoints, credentials, readiness, and
launcher process ownership. Before announcing readiness, the worker completes a
small streaming request with a 1280x720 JPEG so the first user query does not
pay the multimodal initialization cost.

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
