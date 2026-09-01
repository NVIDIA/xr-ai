<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-voice

`xr-ai-voice` owns the voice-facing boundary for an XR agent. Pipecat implements
the private media pipeline; applications use typed runtime events and model
service protocols. Refer to {doc}`python/index` for exact constructors, fields,
and defaults.

<a id="usage"></a>
(agent-sdk-voice-usage)=
## Voice agent

Applications register one `VoiceAgent` with their runtime:

```python
from xr_ai_runtime import Topic
from xr_ai_voice import UserQuery, VadConfig, VoiceAgent, VoiceInterrupted
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

`VoiceAgent` owns model readiness, hub transport, VAD and STT, voice gating, typed
text ingress, TTS, signals, pipeline cancellation, and cleanup. Its media
session remains private. Applications that need a shared public
`HubVoiceTransport` construct and inject one explicitly.

Each non-empty final STT result is queued for publication on
`VOICE_TRANSCRIPT_TOPIC` before wake-phrase filtering. Accepted speech and
untopiced typed text become `UserQuery`; named application and control messages
are never interpreted as user text. Optional participant join, leave, and
interruption topics let application agents own their state cleanup.

Transcript delivery uses one private 32-entry FIFO so a slow subscriber cannot
delay STT or command gating. The queue preserves retained-item order and drops
its oldest pending transcript when full. Shutdown cancels active delivery and
discards pending transcripts. Runtime subscribers must enqueue long-running
work internally and return promptly.

Join publication occurs after the voice gate handles its greeting.
`ProcessorEndpoint` suppresses duplicate roster joins while a participant
remains connected and emits a new join after a leave and reconnect. Join and
leave publication is serialized per participant and does not block the media
processor. `VoiceAgent` cancels and awaits all owned delivery tasks on shutdown.

(voice-tuning-and-data-echo)=
## Voice output and interruption

Applications publish finite or incremental `VoiceOutput` values. Chunks in one
stream share `response_id` and end with `final=True`. Output is serialized per
participant; `interrupt=True` flushes queued hub audio and replaces active
speech. Without aggregation, producer identity is part of the stream key so
independent agents cannot merge accidentally.

`text_topic` controls the completed-response data echo and defaults to
`agent.response`. Set it to an empty string when the application owns its own
caption channel. The echo describes intended completed text, not client playback
acknowledgement.

When a participant joins, the voice transport sends 320 ms of paced silence.
The first chunk causes the hub to publish the return track, and the remaining
interval gives the participant time to subscribe before an immediate greeting
or response. If no output follows immediately, the pre-roll drains and does not
delay a later response. During speech, the sender maintains up to 120 ms of
downstream reserve to absorb ordinary event-loop and IPC jitter. It sends
initial chunks immediately rather than waiting to fill the reserve, and
interruption flushes participant-scoped queued audio. The hub setting
`return_audio_max_buffer_s` must be at least `0.12` for built-in voice output.

(multiple-voice-producers)=
## Multiple speech producers

Applications with foreground replies, background monitors, and alerts may
register one `VoiceAggregationAgent`. Producers publish to
`VOICE_CONTRIBUTION_TOPIC`; only the aggregator publishes to
`VOICE_OUTPUT_TOPIC`.

```python
from xr_ai_voice import VOICE_CONTRIBUTION_TOPIC, VoiceAggregationAgent, VoiceOutput

aggregation = runtime.register(
    "voice-aggregation",
    VoiceAggregationAgent(llm=llm),
)

await ctx.publish(
    VOICE_CONTRIBUTION_TOPIC,
    VoiceOutput(text="The timer is done."),
)
```

Aggregation is participant-scoped. One finite contribution passes through
after a short coalescing window; simultaneous finite updates are rewritten into
one utterance. Completed text is published immediately, while a bounded
open-loop spoken-duration estimate schedules later speech. Tune its word rate
and playback bounds for the selected voice; the estimate is not an audio
acknowledgement.

Rewrite timeout or failure falls back to ordered source text. Urgent output
bypasses coalescing, cancels a rewrite, and interrupts active speech. Bounded
queues prefer recent alerts over routine updates and log every drop. Dropping
or interrupting a streaming contribution quarantines its response ID through
its terminator or idle expiry so stale fragments cannot reopen speech.

Applications call `release(participant_id)` on departure and `stop()` before
runtime shutdown. The aggregator logs accepted contributions discarded during
release or shutdown.

## Voice gating and early probes

VAD and STT probe the opening audio while the user is still speaking. Probe
audio includes a silent tail so offline STT can finalize a phrase. A partial
global-STOP match interrupts active output immediately, but it does not commit
the user's intent: final STT remains authoritative for global-stop versus query
routing. A slow probe receives a short grace period and is then cancelled.

A wake phrase is accepted at the beginning of a transcript or after
sentence-final `.`, `?`, or `!` punctuation followed by whitespace or a closing
quote. Text before the boundary and the phrase are removed. A phrase after a
comma, semicolon, or inside ordinary prose does not activate the gate. Partial
STOP classification checks both raw text and the tail after a configured wake
phrase, so `stop` and `hey agent stop` interrupt equally early.

Global STOP uses a closed imperative grammar for direct requests such as
`stop`, `stop it`, `stop talking`, `be quiet`, and `shut up`, with a bounded set
of conversational prefixes and punctuation. Up to two prefixes may be drawn
from `please`, `hey`, `okay`, `ok`, `uh`, `um`, `wait`, `no`, `just`,
`alright`, `sorry`, `whoa`, `hang on`, `I said`, or `can`, `could`, `would`,
or `will` followed by `you`. Negations (`don't stop`), questions (`should I
stop?`), reported speech (`you said stop`), unconfigured arbitrary prefixes,
and scoped or multi-action commands (`stop monitoring xyz`) are not global
stops. They follow the ordinary gate rules.

The early path emits only an interruption, not a chime or stop acknowledgement.
If the final transcript remains a global stop, normal stop handling emits the
acknowledgement. If STT revises partial `hey agent stop` to final `hey agent stop
monitoring xyz`, the final transcript instead dispatches `stop monitoring xyz`
to the agent; the latency-saving interruption is not undone.

## Relay telemetry

Voice output fragments use a low-cardinality runtime topic. `VoiceAgent` emits
one semantic `voice.response` scope per finite response or completed stream.
The media pipeline emits one `voice.stt` scope per transcription and one
`voice.tts` scope per sentence synthesis. STT records audio size, duration, and
sample rate plus a transcript result mark; TTS records its sentence. Raw audio
is never written to Relay events, and these timings end at provider handoff,
not client playback.
