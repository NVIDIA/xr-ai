<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-voice

The voice runtime for XR agents. Pipecat implements the media pipeline, but
applications work with XR concepts rather than Pipecat modules:

- `VoiceAgent` publishes every final pre-gate STT result on `voice.transcript`
  and accepted speech, text, participant lifecycle, and interruption on typed
  topics; it subscribes to `voice.output`.
- `VoiceAggregationAgent` optionally serializes candidate speech per
  participant and rewrites simultaneous finite updates into one concise
  response before publishing to `voice.output`.
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

## Multiple voice producers

Applications with foreground responses, background monitors, timers, and
alerts can register one `VoiceAggregationAgent`. Producers publish the same
`VoiceOutput` payload to `VOICE_CONTRIBUTION_TOPIC`; only the aggregator
publishes the final response to `VOICE_OUTPUT_TOPIC`:

```python
from xr_ai_voice import (
    VOICE_CONTRIBUTION_TOPIC,
    VoiceAggregationAgent,
    VoiceOutput,
)

aggregation = runtime.register(
    "voice-aggregation",
    VoiceAggregationAgent(llm=llm),
)
```

A producer uses its runtime context while handling an application event:

```python
await ctx.publish(
    VOICE_CONTRIBUTION_TOPIC,
    VoiceOutput(text="The timer is done."),
)
```

The application stops participant-owned aggregation tasks before the runtime:

```python
async with runtime:
    try:
        await voice.run(runtime)
    finally:
        await aggregation.stop()
```

Applications should call `await aggregation.release(participant_id)` from
their participant-left subscriber so buffered speech and stream state are
discarded as soon as that participant departs. The application owns its
lifecycle topic; the shared aggregator therefore exposes cleanup directly
rather than subscribing to one hard-coded topic name.

Aggregation is participant-scoped. A lone finite contribution passes through
without an LLM call after the short coalescing window. Two or more finite
contributions in one batch are rewritten into a single concise utterance. As
soon as raw or rewritten text is complete, the aggregator marks that response
final so `VoiceAgent.text_topic` can update the client independently of audio
playback. The aggregator separately reserves an estimated spoken duration,
based on `speech_rate_wpm` and clamped between `minimum_playback_s` and
`maximum_playback_s`, so updates produced while that output is playing form the
next batch instead of becoming separate queued utterances. This is open-loop
pacing rather than a client playback acknowledgement. An estimate that is too
short moves queuing downstream into TTS; one that is too long creates silence
between utterances, but neither delays the completed-response data echo. Tune
all three playback settings for the selected TTS voice. The echo represents the
complete intended utterance, not a playback acknowledgement: a later urgent
barge-in can interrupt its audio after the full text has already reached the
client.

Rewrites have an aggregator-enforced `rewrite_timeout_s` deadline, which is also
passed to the model service. If a rewrite fails or times out, the original text
is joined so updates are not lost. The earliest input timestamp is preserved.
An `interrupt=True` contribution bypasses the coalescing delay and LLM rewrite,
cancels a rewrite already in progress, and replaces active speech without model
latency. Routine updates already collected in its batch remain ordered for the
next batch. `queue_capacity` bounds all pending contributions for one
participant, including chunks waiting behind another stream. At capacity,
routine input replaces the oldest routine contribution but never an urgent one.
Urgent input replaces the oldest routine contribution when possible, otherwise
the oldest urgent contribution, so the freshest alert wins. Every drop is
logged, and dropping a stream fragment quarantines that stream ID. Batches also
have a bounded size; retained contributions remain ordered for following
batches.

A lone incremental contribution starts streaming immediately with a new
aggregator-owned response ID. Its final content chunk closes the downstream
response immediately; the playback reservation continues only inside the
aggregator's scheduler. Other contributions wait behind that reservation, and
finite updates that accumulated while it ran are coalesced afterwards. An
urgent contribution interrupts the active stream. A stream that stops producing
chunks is closed after `stream_idle_timeout_s`, allowing pending updates to
proceed. An interrupted or dropped stream ID remains quarantined until its
terminator arrives. Each ignored fragment refreshes the quarantine; after a
full `stream_idle_timeout_s` without another fragment, the ID may begin a new
stream. This expiry bounds stale stream state independently from pending queue
capacity.

Call `stop()` before shutting down the runtime so the agent can cancel its
participant-owned tasks. Contributions published after shutdown starts are
logged and dropped. Shutdown and participant release log accepted contributions
that are still pending or in pre-publication processing. Output already handed
to the voice pipeline during the playback hold is not counted as discarded.

## Voice tuning and data echo

`VadConfig` controls utterance boundaries and bounded early transcription:
the generated Python reference lists every field, type, default, and description.
Set `stop_probe_after_s` to `0` or less to disable early probes.

`VoiceAgent.text_topic` controls the completed-response echo sent through the
hub data channel. Its default is `"agent.response"`; set it to `""` when the
application publishes its own response data so each turn is delivered only
once. This setting does not disable TTS or Relay telemetry.

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

Accepted speech, typed text, participant lifecycle, and interruption remain on
application-named topics. Pass `participant_joined_topic` and/or
`participant_left_topic` to publish `VoiceParticipantJoined` and
`VoiceParticipantLeft`; participant identity is runtime metadata rather than
event payload. Join publication occurs after the voice gate handles its greeting.
The hub suppresses duplicate roster joins while a participant remains connected
and emits a new join after a leave/reconnect. Application agents subscribe to the
events they own, perform cleanup in their own subscriber methods, and may publish
finite or incremental `VoiceOutput` messages:

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
chunks remain, and end with `final=True`. Without the optional aggregation
agent, producer identity is part of the response key so independent agents
cannot merge output accidentally. `VoiceAgent` serializes output per
participant; urgent output sets `interrupt=True` to replace active and queued
speech. Producers may copy the originating query's `timestamp_us` into
`VoiceOutput` so the TTS response preserves the input timestamp.
`VoiceStreamClosedError` identifies an empty final chunk for a stream that
voice already closed, including when wrapped in the runtime publication
exception group.

Lifecycle publication runs on `VoiceAgent`-owned tasks, so participant cleanup
cannot block the shared media processor. Join and leave publications are
serialized per participant, preserving lifecycle order even when a subscriber
is slow; different participants remain independent. The agent cancels and
awaits those tasks during shutdown.

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
