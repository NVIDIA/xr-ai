<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Magpie NIM HTTP speech service

`services/magpie-nim-tts` exposes a running Magpie Riva NIM through an
OpenAI-style `/v1/audio/speech` endpoint compatible with the existing Pocket
TTS client. The service calls the typed `xr_ai_models` Riva client. Consumers
use HTTP and do not need the Riva dependency.

## Run

Start the Magpie NIM backend separately. This service provides HTTP adaptation;
it does not download models, launch the NIM container, or compile engines.
It is separate from `services/magpie-tts`, which loads NeMo in-process.

From `services/magpie-nim-tts/`:

```bash
uv --config-file ../../uv.toml sync
uv run magpie_nim_tts --config magpie_nim_tts.yaml
```

Configure the backend address, readiness URL, voice, language, sample rate,
and post-synthesis pause in `magpie_nim_tts.yaml`. The default HTTP port is the
configured Pocket client port, 8105; stop any other listener before using it.
The backend must already be healthy before the adapter announces readiness.
A repeated launch reuses a healthy, ownership-marked adapter with the same
configuration. A running pre-split TTS adapter reporting the
`nim-model-adapter` identity is also reusable when its complete configuration
matches; ownership and health checks still apply. Other legacy adapter kinds
are not accepted. Configuration changes require stopping and restarting it.

## HTTP contract

`POST /v1/audio/speech` accepts `input`, `response_format` (`wav` or `pcm`),
and `stream`. WAV is the default and is buffered. For incremental raw audio,
send `response_format: "pcm"` and `stream: true`; PCM chunks are forwarded as
they arrive over gRPC. Audio is mono signed 16-bit PCM at the configured
sample rate. Streaming responses include `X-Audio-Sample-Rate` and
`X-Audio-Channels` headers. Voice and language are server settings.

This is the existing Pocket-compatible subset, not the complete OpenAI TTS
API: other formats and per-request voice, speed, or instruction controls are
not implemented. Additional request fields are ignored. The service exposes
`GET /v1/models` and readiness at `/health` and `/v1/health/ready`.

The service appends silence after each successful, nonempty synthesis request,
for both buffered and streamed responses. The default pause is 300 ms;
`post_synthesis_pause_ms: 0` disables it. The pause is encoded as zero-valued
samples, so it remains audible with buffered playback. No silence is appended
for empty or failed synthesis. The service does not split text at punctuation;
callers that submit one sentence per request receive one pause per sentence.

Synthesis is serialized. Both buffered and streaming requests watch for client
disconnects while queued and during synthesis. Abandoned queued requests never
start an RPC. Disconnects and timeouts cancel an active upstream RPC;
the synthesis lock stays held until its blocking read finishes, including
repeated cancellation during cleanup. Errors before the first audio return
an HTTP error; errors after streaming starts terminate the response without
inserting error text into PCM.

## Test

From the service directory:

```bash
uv --config-file ../../uv.toml run --project ../../tests python -m pytest ../../tests/test_magpie_nim_lifecycle.py ../../tests/test_magpie_nim_speech.py -q
```

Tests use controlled local gRPC and HTTP servers and require no model or GPU.
