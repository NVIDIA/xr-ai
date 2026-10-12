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

`VoiceAgent` owns application readiness, hub transport, VAD and STT, voice gating,
typed text ingress, TTS, signals, pipeline cancellation, and cleanup. Its media
session remains private. Applications that need a shared public
`HubVoiceTransport` construct and inject one explicitly.

For the startup probe contract, refer to {py:class}`~xr_ai_voice.VoiceAgent`.
For deployment order and migration of existing applications, refer to
{ref}`consumer-model-readiness`.

Each non-empty final STT result during an active conversation is queued for
publication on `VOICE_TRANSCRIPT_TOPIC` before optional wake-phrase filtering.
Conversation control phrases and speech while the conversation is closed are
withheld. With conversation controls disabled, all non-empty final results are
published before wake-phrase filtering. Accepted speech and
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

Each running `VoiceAgent` remembers its 1,024 most recently closed stream keys.
Output that reuses one of those keys is ignored, and the runtime logs one
warning for that participant, producer, and `response_id`. Streams close by
finalization, cancellation, or eviction; waiting does not reopen a retained
key. Use one identifier per inbound query, such as
`ctx.metadata.message_id`, or omit `response_id` when publishing a finite
response in one `VoiceOutput`.

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

(voice-conversation-controls)=
## Conversation controls

The shipped voice samples start with their microphone conversation closed.
After connecting, say `Hey agent, let's start talking`, then speak naturally
without a wake phrase. Say `Hey agent, let's stop talking` to close the
conversation and interrupt the current response. A short `stop` interrupts the
response while keeping the conversation open.

Configure the phrases in each sample's voice-gate YAML:

```yaml
conversation:
  enabled: true
  start_phrase: "Hey agent, let's start talking"
  stop_phrase: "Hey agent, let's stop talking"
  require_wake_phrase: false
  phrase_window_s: 6.0
```

Matching ignores case, punctuation, and apostrophes; `let us` also matches
`let's`. Exact control prefixes can span final STT results: `Hey agent`, followed
by `let's start talking`, starts the conversation if their speech onsets fall
within `phrase_window_s`. An unrelated or unrecognized utterance, an expired
window, or a disconnect discards pending fragments. Quoting the phrase inside a
longer utterance does not activate it. Control phrases never become agent
queries or transcript-topic events.

These controls use the sample's existing STT service, including the independent
NIM model stack. They do not identify speakers or reject other voices on the
same microphone. Conversation state is scoped to each connected participant.
Untopiced typed messages bypass speech gating and are delivered as queries even
while the microphone conversation is closed. They cannot open or close a
microphone conversation.

Set `conversation.require_wake_phrase: true` to require `magic_phrases` during
an active conversation, preserving the optional chime and follow-up grace
window. Set `conversation.enabled: false`, or omit the mapping, for the original
wake-only behavior. An empty `magic_phrases` list in that mode restores
immediate always-on speech dispatch.

## Voice gating and early probes

VAD and STT probe the opening audio while the user is still speaking. Probe
audio includes a silent tail so offline STT can finalize a phrase. During an
active conversation, a partial global-STOP match interrupts output immediately,
but it does not commit
the user's intent: final STT remains authoritative for global-stop versus query
routing. A slow probe receives a short grace period and is then cancelled.
With wake phrases or conversation controls configured, one utterance can make
up to three bounded partial STT requests plus the authoritative final request. Set `stop_probe_after_s` to
zero to disable the additional requests and early interruption path.

When wake gating is active, a wake phrase is accepted at the beginning of a
transcript or after sentence-final `.`, `?`, or `!` punctuation followed by
whitespace or a closing quote. Text before the boundary and the phrase are
removed. A phrase after a comma, semicolon, or inside ordinary prose does not
activate the gate. In that mode, partial STOP classification checks both raw
text and the tail after a configured wake phrase, so `stop` and
`hey agent stop` interrupt equally early. With natural speech enabled by
`conversation.require_wake_phrase: false`, final transcripts remain intact and
only a standalone short stop uses the global-stop path; `hey agent stop`
remains a query.

Global STOP uses a closed imperative grammar for direct requests such as
`stop`, `stop it`, `stop talking`, `be quiet`, and `shut up`, with a bounded set
of conversational prefixes and punctuation. Up to two prefixes may be drawn
from `please`, `hey`, `okay`, `ok`, `uh`, `um`, `wait`, `no`, `just`,
`alright`, `sorry`, `whoa`, `hang on`, `I said`, or `can`, `could`, `would`,
or `will` followed by `you`. Negations (`don't stop`), questions (`should I
stop?`), reported speech (`you said stop`), unconfigured arbitrary prefixes,
and scoped or multi-action commands (`stop monitoring xyz`) are not global
stops. They follow the ordinary gate rules.

A partial STOP match emits only an interruption, not a chime or stop
acknowledgement. Wake recognition may independently emit the optional chime,
but chime configuration, initialization, or playback does not control probing.
If the final transcript remains a global stop, normal stop handling emits the
acknowledgement. If STT revises partial `hey agent stop` to final `hey agent stop
monitoring xyz`, the final transcript instead dispatches `stop monitoring xyz`
to the agent; the latency-saving interruption is not undone.

(speaker-enrollment)=

## Speaker filtering and backend selection

The same conversation controls work with ordinary STT and optional speaker
filtering. Add `speaker` alongside `conversation` in the worker's voice-gate YAML:

```yaml
speaker:
  enabled: true
  backend: auto
  base_url: http://127.0.0.1:8102
```

Before announcing readiness, the worker probes the service's HTTP `/health`
endpoint. A ready compatible service selects diarization. In automatic mode,
connection absence selects the configured ordinary `STTService`, including the
unchanged NIM stack's HTTP STT adapter. Once a listener is observed, an unresponsive
or interrupted response means not-ready: the worker waits using the existing
service-readiness polling instead of silently selecting unfiltered STT. A stalled
listener can therefore hold startup until it is ready or startup is cancelled.
Incompatible identities and HTTP error statuses fail startup. Ambient proxy
settings do not redirect either the HTTP probe or the WebSocket stream.
The worker logs its selection. Ordinary STT cannot identify or reject other voices
on the same microphone. Set `speaker.backend: required` to wait for diarization
even when the service is absent, or `speaker.enabled: false` to select ordinary
STT while retaining conversation controls. Restart the worker to change backends.
Loss of a selected diarization service resets enrollment instead of switching to
unfiltered STT.

Speaker filtering uses Nemotron 3 Diarization with Multitalker Parakeet. A private
inference process shares model weights across workers. The private model adapter
opens one WebSocket at `/v1/audio/transcriptions/stream` per participant. The
connection owns that participant's enrollment and closes when the participant
leaves or its audio worker resets. After one configuration and audio-origin
message, the worker sends ordered 16 kHz mono signed-16 PCM frames and waits for
each event response before sending the next frame. The service derives timestamps
from accepted sample counts; reconnecting starts a fresh timeline and requires
enrollment again. Other voices contribute to
diarization and interference conditioning, but are not transcribed after
enrollment. Before enrollment, the service independently decodes each detected
speaker and enrolls the one whose conditioned transcript completes the start
phrase, including during overlapping speech. If multiple speakers complete the
phrase in the same accepted audio step, neither is selected. Once enrolled,
another voice's later start phrase cannot take over the connection.

The hub's existing microphone track identity is retained privately. Switching
to a different track closes the old stream, discards pending audio and controls,
revokes enrollment, and opens a new timeline at the next track's first audio.
Microphone stop and republish therefore cannot join old phrase fragments across
tracks. Gaps on the same track retain accepted-sample timing: receipt timestamps
do not establish capture continuity, and delivery jitter does not trigger guessed
resets. Explicit same-track capture epochs are outside this integration.

Start the inference process separately from the repository root:

```bash
uv --config-file uv.toml run --project services/speaker-stt \
  python -m speaker_stt --config services/speaker-stt/speaker_stt.yaml
```

Say the exact start phrase to nominate its speaker. Other speech can continue:
the service conditions each candidate transcript on that speaker's target and
interference masks. Candidate phrases finalize on each speaker's configured
inactivity boundary, without requiring global silence. A tie means multiple
exact matches completed in the same accepted audio step, not a claim about
absolute acoustic simultaneity. Split controls retain the same speaker label;
truncated candidate episodes cannot nominate from their continuing tail. No
candidate transcript dispatches a query before enrollment. After enrollment,
only the selected speaker can send queries or release the conversation. Wake
phrases remain optional through `conversation.require_wake_phrase`; filtering
cannot determine whom the wearer is addressing.

Conversation state belongs to a participant connection. Disconnect, inference
failure or processing overload requires
enrollment again. Per-participant queues retain at most two seconds or 200 audio
frames; overload drops pending audio and revokes enrollment. Workers wait two
seconds before retrying after failure or overload. Closing the WebSocket releases
its inference session, and the service bounds their count. At `max_utterance_s`,
an enrolled speaker's final transcript is retained, but the incomplete utterance
cannot activate a conversation control or enroll a new speaker.

Departure closes admission before asynchronous cleanup, and only an explicit
participant join permits new audio. Cleanup retains task and queue ownership
through stream close. Participant lifecycle and enrollment transitions retain
their order and survive pipeline interruptions. An active-to-reset transition
interrupts the current answer and emits one listening-reset notice; subsequent
outage retries do not repeatedly interrupt or announce the same reset.

Worker phrase settings do not require restarting the model service. When both
`conversation` and `speaker` are present, `conversation` owns the control settings.
Speaker-only YAML remains supported. Disabling conversation controls also disables
speaker enrollment; typed input retains its existing behavior.

Speech accuracy, identity stability, interference suppression, latency and GPU
memory require live qualification on the intended microphones. Enrollment selects
a session speaker; it is not authentication or a persistent biometric identity.
No speaker profiles are saved to disk.

## Relay telemetry

Voice output fragments use a low-cardinality runtime topic. `VoiceAgent` emits
one semantic `voice.response` scope per finite response or completed stream.
The batch-STT path emits one `voice.stt` scope per transcription; speaker
filtering does not currently emit STT scopes. The media pipeline emits one
`voice.tts` scope per sentence synthesis. Batch STT records audio size, duration, and
sample rate plus a transcript result mark; TTS records its sentence. Raw audio
is never written to Relay events, and these timings end at provider handoff,
not client playback.
