<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM example

The simple VLM example is the smallest complete voice-and-vision application in
the repository. It answers spoken or typed questions about each participant's
latest camera frame and streams the answer to both Piper TTS and the
`vlm.response` data topic. Start with the {doc}`quickstart
</getting_started/quickstart>`; this page owns the sample's design and
operational details.

## Composition

The orchestrator starts only DeviceIOHub and the worker. The fixed
`yaml/models.json` profile marks Parakeet STT, Cosmos3 Nano, and Piper TTS as
reused services, so operators start those endpoints independently.

`VoiceAgent` owns service readiness, hub transport, voice gating, TTS, signals,
and cleanup. It publishes accepted speech and typed text as a participant-scoped
`UserQuery`. `SimpleVlmAgent` selects the participant's current image with
`CurrentFrameTool`, passes its opaque reference to
`StreamingImageQueryTool`, and publishes response chunks to voice output.
Camera bytes remain on the hub path, image locations are redacted from VLM
telemetry, and the sample has no MCP path.

A newer participant turn cancels the superseded vision request and interrupts
its voice response. Participant departure releases the sample agent's cached
frames and tasks. This is the reference composition for a single foreground
streaming image query.

## Readiness and warmup

Before announcing readiness, the worker performs a streaming VLM request with
a 1280×720 JPEG and consumes the response. This exercises the production
multimodal path so the first user query does not pay its initialization cost.
Endpoint health alone is not sufficient for that warmup.

## Configuration

- `yaml/models.json` owns model adapters, endpoints, readiness, and reuse
  declarations.
- `yaml/simple_vlm_example_worker.yaml` owns timing, frame freshness, and
  optional prompt overrides.
- `yaml/voice_gate.yaml` owns wake phrases and the follow-up window.
- `worker/simple_vlm_example_worker/prompts/system.txt` is the default VLM
  instruction.

Exact fields and defaults are rendered in the generated
{doc}`configuration <configuration>` reference.

Wake phrases match at the start of a final transcript or after sentence-final
`.`, `?`, or `!` punctuation followed by whitespace or a closing quote. Text
before that boundary and the phrase itself are discarded before dispatch.

## Relay output

The worker writes `relay-events.jsonl` beside `worker.log` in the per-run log
directory printed at startup. It records runtime publications, receiving-agent
callbacks, the complete `simple-vlm.turn` lifetime, and nested vision and model
calls. Per-token model marks, incremental voice fragments, and empty stream
terminators are omitted. Voice, STT, and TTS each emit one semantic scope for
the completed operation; raw audio is represented only by size, duration, and
sample rate.

Image locations appear as `<redacted:image>`. Prompts, questions, responses,
participant identities, and correlation metadata remain visible and may be
sensitive. Inspect a running sample with:

```bash
tail -F /tmp/log_simple-vlm-example_*/relay-events.jsonl
```
