<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-voice

The voice runtime for XR agents. Pipecat implements the media pipeline, but
applications work with XR concepts rather than Pipecat modules:

- `VoiceAgent` publishes every raw microphone chunk on `voice.audio`, every
  final pre-gate STT result on `voice.transcript`, and accepted speech, text,
  participant departure, and interruption on typed topics; it subscribes to
  `voice.output`.
- `VoiceAgent` privately owns readiness, hub transport, pipeline assembly,
  signals, execution, and cleanup.
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
)
from xr_ai_voicegate import VoiceGateConfig

queries = Topic("my-sample.user-query", UserQuery)
interruptions = Topic("my-sample.interrupted", VoiceInterrupted)
voice = VoiceAgent(
    query_topic=queries,
    stt=stt,
    tts=tts,
    vad=VadConfig(),
    voice_gate=VoiceGateConfig(),
    interrupted_topic=interruptions,
)
runtime.register("voice", voice)

async with runtime:
    await voice.run(runtime)
```

## Voice tuning and data echo

`VadConfig` controls utterance boundaries and bounded early transcription:

| Field | Default | Meaning |
|---|---:|---|
| `silence_duration` | `0.8` | Seconds of silence that finalize an utterance. |
| `min_speech` | `0.15` | Minimum speech duration accepted as an utterance. |
| `silero_threshold` | `0.5` | Silero VAD speech-probability threshold. |
| `stop_probe_after_s` | `0.25` | Cadence for up to three early wake/STOP transcription probes; set to `0` or less to disable probes. |

`VoiceAgent.text_topic` controls the completed-response echo sent through the
hub data channel. Its default is `"agent.response"`; set it to `""` when the
application publishes its own response data so each turn is delivered only
once. This setting does not disable TTS or Relay telemetry.

`VoiceAgent` publishes every incoming microphone chunk on `voice.audio` before
VAD, STT, or wake-phrase filtering. This includes silence and speech that does
not contain the configured wake phrase, so agents that need the complete input
stream can subscribe to `VOICE_AUDIO_TOPIC`. `VoiceAudio.data` is the original
interleaved little-endian float32 PCM from the hub; the payload also carries its
sample rate, channel and sample counts, capture timestamp, and track ID. The
participant ID and `voice` source are runtime metadata. The topic disables
Relay telemetry so raw audio is never recorded in runtime scopes.

Each non-empty final STT result is published on `VOICE_TRANSCRIPT_TOPIC` as a
`VoiceTranscript` before wake-phrase filtering. It therefore includes speech
that the gate later rejects. The participant ID and `voice` source are runtime
metadata. Early wake/STOP probes are internal and are not published as final
transcripts. Accepted speech continues separately as `UserQuery`, with any
matched wake phrase and preceding background speech removed by the gate.

Delivery uses one FIFO worker and bounded queue per participant and track.
`audio_capacity` defaults to 32 queued chunks for each stream. If a subscriber
cannot keep up and that queue fills, voice drops the oldest queued chunk and
keeps the newest live audio; an in-flight chunk is never reordered. Subscriber
failures are logged without stopping later chunks, and `VoiceAgent` cancels and
awaits every audio worker during participant cleanup and shutdown.

Accepted speech, typed text, participant departure, and interruption remain on
application-named topics. Application agents subscribe to the events they own,
perform cleanup in their own subscriber methods, and may publish finite or
incremental `VoiceOutput` messages:

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
preserves the input timestamp. `VoiceStreamClosedError` identifies an empty final
chunk for a stream that voice already closed, including when wrapped in the
runtime publication exception group.

Lifecycle publication runs on `VoiceAgent`-owned tasks, so participant cleanup
cannot block the shared media processor. The agent cancels and awaits those
tasks during shutdown.

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

The media session is private to `VoiceAgent`. The agent manages readiness, hub
transport, VAD/STT, voice gating, TTS, typed-text ingress, signals, and cleanup;
it does not execute application handlers. `VoiceAgent.endpoint` and
`VoiceAgent.transport` expose the owned hub boundary when another application
component needs frames, status publication, or the shared transport. Callers do
not construct, run, or close a separate session.

When wake phrases and the listening chime are enabled, the VAD/STT stage probes
the opening audio on a fixed cadence while the user is still speaking. Probe
audio includes a short silent tail so offline STT can finalize the wake word. A
recognized phrase emits the chime immediately, while only the final transcript
enters the voice gate as a query. An in-flight probe gets a short grace period
before final STT and is then cancelled, so a slow probe cannot stall the audio
pipeline. A missed probe never inserts a late chime in front of response speech.
STOP commands use the same early-probe path for immediate interruption.

The final transcript accepts a wake phrase at its beginning or after
sentence-final `.`, `?`, or `!` punctuation followed by whitespace or a closing
quote. Text before that boundary and the matched phrase are discarded. A phrase
after a comma, semicolon, or within ordinary prose does not activate the gate.
