<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-voice

The voice runtime for XR agents. Pipecat implements the media pipeline, but
applications work with XR concepts rather than Pipecat modules:

- `VoiceAgent` publishes accepted speech, text, participant departure, and
  interruption as voice-owned schemas on application-named topics; it
  subscribes to `voice.output`.
- `VoiceSession` owns readiness, hub transport, private pipeline assembly,
  signals, execution, and cleanup behind `VoiceAgent`.
- `HubVoiceTransport` is available when an application needs to share one transport explicitly.

## Usage

Applications register one `VoiceAgent` with the shared runtime:

```python
from xr_ai_runtime import Topic
from xr_ai_voice import (
    UserQuery,
    VadConfig,
    VoiceAgent,
    VoiceInterrupted,
    VoiceSession,
)
from xr_ai_voicegate import VoiceGateConfig

session = VoiceSession(
    stt=stt,
    tts=tts,
    vad=VadConfig(),
    voice_gate=VoiceGateConfig(),
)
queries = Topic("my-sample.user-query", UserQuery)
interruptions = Topic("my-sample.interrupted", VoiceInterrupted)
voice = VoiceAgent(
    session,
    query_topic=queries,
    interrupted_topic=interruptions,
)
runtime.register("voice", voice)

async with runtime:
    await voice.run(runtime)
```

`VoiceAgent` publishes accepted speech, typed text, participant departure, and
interruption on application-named topics. Application agents subscribe to the
events they own, perform cleanup in their own subscriber methods, and may
publish finite or incremental `VoiceOutput` messages:

```python
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceOutput

await runtime.publish(
    VOICE_OUTPUT_TOPIC,
    VoiceOutput(text="Move your hand away.", interrupt=True),
    participant_id="alice",
    source="safety-monitor",
)
```

Incremental producers reuse a `response_id`, set `final=False` while more
chunks remain, and end with `final=True`. Aggregation is private to voice/TTS,
and producer identity is part of the response key so independent agents cannot
merge output accidentally. Output is serialized per participant; urgent output
sets `interrupt=True` to replace active and queued speech. Producers may copy
the originating query's `timestamp_us` into `VoiceOutput` so the TTS response
preserves the input timestamp.

Relay telemetry treats `voice.output` as a high-cardinality transport topic and
does not emit runtime scopes per fragment. `VoiceAgent` instead emits one
`voice.response` agent scope for each finite response or completed incremental
stream. Its input contains the combined text, streaming flag, fragment count,
and interrupt flag; metadata identifies the participant, producer, response,
timestamp, and completion status.

The media pipeline also emits one `voice.stt` function scope per final or
bounded partial-probe transcription and one `voice.tts` function scope per
sentence synthesis. STT inputs contain only byte count, duration, and sample
rate; a nested `voice.stt.result` mark carries the transcript. TTS inputs carry
the sentence being synthesized. Raw audio is never written to Relay events.
These scopes measure provider work and downstream handoff, not client playback.

`VoiceSession` is the media engine owned by `VoiceAgent`. It manages
readiness, hub transport, VAD/STT, voice gating, TTS, signals, and cleanup. It
The lower-level `VoiceSession.run()`, `enqueue_query()`, and
`enqueue_response()` methods are public for runtime integrations.
`VoiceSession.endpoint` is available only after entering the session, so model
health probes complete before the default hub transport opens its sockets.

does not execute application handlers. Typed-text ingress is also internal to
`VoiceAgent`.

When wake phrases and the listening chime are enabled, the VAD/STT stage probes
the opening audio while the user is still speaking. A recognized phrase emits
the chime immediately, while only the final transcript enters the voice gate as
a query. STOP commands use the same early-probe path for immediate interruption.
