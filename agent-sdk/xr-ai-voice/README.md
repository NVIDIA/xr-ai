<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-voice

The voice runtime for XR agents. Pipecat implements the media pipeline, but
applications work with XR concepts rather than Pipecat modules:

- `VoiceAgent` publishes every final pre-gate STT result on `voice.transcript`
  and accepted speech, text, participant departure, and interruption on typed
  topics; it subscribes to `voice.output`.
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

When the configured TTS object implements `StreamingTTSService`, the media
pipeline forwards each signed-16-bit PCM chunk as it arrives. Other TTS
services keep the complete-WAV path. `VoiceAgent.inter_sentence_pause_ms`
inserts silence only before the second and later synthesized sentences in one
response; it never delays the first sentence and defaults to `0`.

Each non-empty final STT result is published on `VOICE_TRANSCRIPT_TOPIC` as a
`VoiceTranscript` before wake-phrase filtering. It therefore includes speech
that the gate later rejects. The participant ID and `voice` source are runtime
metadata. Early wake/STOP probes are internal and are not published as final
transcripts. Accepted speech continues separately as `UserQuery`, with any
matched wake phrase and preceding background speech removed by the gate.

Transcript publication uses one private bounded FIFO owned by `VoiceAgent`, so
a slow runtime subscriber cannot delay STT, voice gating, or accepted queries.
The queue preserves order and drops its oldest pending transcript when full;
shutdown cancels the active delivery and discards pending transcripts. Runtime
subscribers should enqueue long-running work internally and return promptly.

Accepted speech, typed text, participant departure, and interruption remain on
application-named topics. Application agents subscribe to the events they own,
perform cleanup in their own subscriber methods, and may publish finite or
incremental `VoiceOutput` messages:

`VoiceAgent` consumes only untopiced client data when `text_input=True`; named
application and control messages are never interpreted as user queries. The
hub preserves the original data-channel topic for every processor.

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
sentence TTS request. STT inputs contain only byte count, duration, and sample
rate; a nested `voice.stt.result` mark carries the transcript. TTS inputs carry
the sentence being synthesized. Raw audio is never written to Relay events.
These scopes measure provider work and downstream handoff, not client playback.

The media session is private to `VoiceAgent`. The agent manages readiness,
VAD/STT, voice gating, TTS, typed-text ingress, signals, and cleanup; it does not
execute application handlers. Applications that already need the public
`HubVoiceTransport` for frames or status publication construct it explicitly
and pass it to `VoiceAgent`; the agent does not expose its private session or
transport.

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
