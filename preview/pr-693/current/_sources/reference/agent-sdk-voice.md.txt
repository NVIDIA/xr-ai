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

Applications construct speech from their loaded model profile and voice-gate
configuration, then register one `VoiceAgent` with their runtime:

```python
from pathlib import Path

from xr_ai_interaction import Interrupted, UserQuery
from xr_ai_models import load_models_config
from xr_ai_runtime import Topic
from xr_ai_voice import VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

models = load_models_config("yaml/models.json")
voice_gate = load_voice_gate_config(Path("yaml/voice_gate.yaml"))
queries = Topic("my-sample.user-query", UserQuery)
interruptions = Topic("my-sample.interrupted", Interrupted)
voice = VoiceAgent.from_models(
    models=models,
    query_topic=queries,
    tts=tts,
    vad=VadConfig(),
    voice_gate=voice_gate,
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

Set utterance-ending silence once through `VadConfig.silence_duration` passed
to `VoiceAgent`. It controls the active backend: Silero VAD for batch STT, or
per-speaker inactivity for speaker ASR. For speaker ASR, words decoded within
the same interval before a speaker becomes active also open their utterance,
since the decoder can lead diarization. This value takes precedence over the
voice-gate YAML's `speaker.silence_duration`; direct speaker sessions retain
their own configuration. Other `VadConfig` fields apply only to batch STT.
Speaker ASR checks inactivity at model-chunk boundaries, so the threshold is
not an exact end-to-end latency target. Changes apply to newly opened sessions.

For the startup probe contract, refer to {py:class}`~xr_ai_voice.VoiceAgent`.
For deployment order and model availability requirements, refer to
{ref}`consumer-model-readiness`.

Each non-empty final STT result during an active conversation is queued for
publication on `VOICE_TRANSCRIPT_TOPIC` before optional wake-phrase filtering.
In conversation mode, control phrases and speech while closed are withheld.
Hybrid mode also publishes accepted closed-state one-off questions, never
ignored candidate speech. With batch ASR in non-conversation modes, non-empty
final results are published before wake-phrase filtering; speaker ASR publishes
only explicitly authorized turns. Accepted speech and
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

Applications publish `ResponseOutput` values with `kind="reply"` or
`kind="update"`; {doc}`agent-sdk-interaction` defines the messages and the
`ResponseOutputs` helper. Chunks in one streamed reply share `response_id` and
end with `final=True`. Producer identity is part of the stream key so
independent agents cannot merge accidentally.

Each running `VoiceAgent` remembers its 1,024 most recently closed stream keys.
Output that reuses one of those keys is ignored, and the runtime logs one
warning for that participant, producer, and `response_id`. An empty terminator
for a closed stream, such as a reply's terminator after a barge-in, is ignored
without a warning. Streams close by finalization, cancellation, or eviction;
waiting does not reopen a retained key. Use one identifier per inbound query,
such as `ctx.metadata.message_id`, or omit `response_id` when publishing a
finite response in one `ResponseOutput`.

(multiple-voice-producers)=
### Multiple speech producers

Several agents can publish to the same participant. Each participant has one
voice floor that decides what is spoken and when:

- One output holds the floor at a time, from when its text starts until its
  last audio has been sent to the hub, including while TTS is still
  synthesizing it and across pauses in playback. The next output starts after
  it, so cutting the current output never cuts output queued behind it.
- Output is never replayed. Once any of it reaches TTS it is spent, whether it
  plays to the end or is cut; only output that has made no sound yet stays
  pending. A producer that must not lose a notice marks it urgent.
- A reply for request R plays in order with other replies for R. A reply for a
  different request supersedes the previous request, even when nothing is
  playing, cancels that request's current output and queued replies, and
  flushes queued hub audio. Later output for a superseded request is dropped.
  A reply never cuts an update: it waits for a playing update to finish, and
  an update still being prepared waits for the reply.
- An urgent update cancels current output unless that is itself urgent,
  supersedes the cancelled reply's request, flushes queued hub audio, and plays
  next. Urgent updates play in order without cutting each other.
- Other updates wait until neither the bot nor the participant is speaking, then
  all pending updates are spoken as one utterance. If the participant starts
  speaking or one of its updates is withdrawn before the utterance reaches TTS,
  including while the combiner runs, it is dropped at once and rebuilt from
  the updates still pending when it may speak again. `VoiceAgent` learns playback state from the media
  output, not from a duration estimate. By default the texts are joined with
  spaces; pass `combiner`, an async callable from the pending texts to one
  spoken text, to rewrite them. `llm_update_combiner(llm)` returns a combiner
  that rewrites them with an `LLMService` and falls back to the join on failure
  or when the rewrite stops at its token limit.
  A failing combiner, or one that returns no text, also falls back to the join. At most 64 updates wait per
  participant; the oldest is dropped with a warning.
- A new user query or a barge-in cancels replies and the response in flight and
  supersedes the active request. Pending and queued urgent updates are kept,
  including an urgent update cancelled before any of it reached TTS; output
  that was playing is not replayed.
- A `withdraw` from a producer drops that producer's pending updates and urgent
  updates that have not started. Output already playing is not cut; a batch being combined
  that contains a withdrawn update is dropped at once and rebuilt from the
  remaining updates. Other producers' updates are kept.
- A streamed reply that receives no fragment for `stream_idle_timeout_s`
  (default 15 s) is closed so a stalled producer cannot hold the floor. The
  producer is not told: its later fragments are dropped, and its terminator is
  still accepted, so `Reply.delivered` can be true for text that was never
  spoken.

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

## Voice-gate configuration reference

`xr-ai-voicegate` ships the annotated `voice_gate.yaml` below as its actual
packaged default. Each sample selects its own `voice_gate_yaml`; that file's
fields override the packaged baseline. Missing or empty files use the baseline.
There is no new discovery rule, public loader parameter, or inheritance between
sample files. `mode` chooses interaction independently of the speech backend;
`speaker.enabled` selects speaker ASR. Lists replace rather than append to
default values. Direct `VoiceGateConfig()` construction retains its typed
defaults; direct processor callers infer wake-only from non-empty phrases and
otherwise always-on. The YAML loader normalizes and validates the explicit mode
once against the packaged baseline.

```{literalinclude} ../../../utils/xr-ai-voicegate/xr_ai_voicegate/voice_gate.yaml
:language: yaml
```

(voice-conversation-controls)=
## Conversation controls

The Simple VLM, Tea making, lab monitoring and XR render samples start with
their microphone conversation closed.
After connecting, say `Hey agent, let's start talking`, then speak naturally
without a wake phrase. Say `Hey agent, let's stop talking` to close the
conversation and interrupt the current response. A short `stop` interrupts the
response while keeping the conversation open.

Choose a mode in each sample's voice-gate YAML:

| Mode | Before start | After start | Speaker ownership |
|---|---|---|---|
| `wake_only` | Wake-addressed turns only | No persistent conversation | No persistent owner |
| `conversation` | Exact start phrase only | Natural speech until stop | Start phrase enrolls its speaker |
| `hybrid` | Wake-addressed one-off or exact start | Natural speech until stop | One-off never enrolls; start enrolls |
| `always_on` | All recognized speech | No persistent conversation | No persistent owner |

`wake_only` and `hybrid` require non-empty `magic_phrases`. The packaged default
is `always_on`, preserving the existing ungated fallback; the four conversation
samples explicitly select `conversation`. Procedure capture instead uses
ungated narration with STOP disabled. Change the sample's `mode` field to
choose another interaction policy without maintaining a separate preset copy.

```yaml
mode: hybrid
magic_phrases: ["hey agent"]
```

While closed, `Hey agent, what is this?` answers one question without opening
the conversation or registering a persistent speaker. The next unaddressed
utterance is ignored. `Hey agent` alone permits one next utterance in the
existing `followup_grace_s` window; its onset must be in the window, but it may
finish later. Speaker ASR binds that permission to the same recognized speaker.
This temporary permission is not speaker registration or an idle conversation
timeout. A control prefix such as `Hey agent, let's` is withheld, never sent
as a question, and consumes an earlier bare-wake permission; it does not preserve
that permission for a third utterance. The complete start phrase opens the conversation.

Tune conversation controls separately:

With explicit `mode`, put all session controls in `conversation`, not `speaker`;
misplaced controls are rejected instead of silently replaced by defaults.

```yaml
conversation:
  start_phrase: "Hey agent, let's start talking"
  stop_phrase: "Hey agent, let's stop talking"
  require_wake_phrase: false
  phrase_window_s: 6.0
```

Matching ignores case, punctuation, apostrophes, and spacing; `let us` also
matches `let's`. Exact control prefixes can span final STT results, and a
boundary may fall inside a word: `Hey agent` (or `Hey ag`), followed by
`let's start talking` (or `ent, let's start talking`), starts the conversation
if their speech onsets fall within `phrase_window_s`.

An utterance that could begin a control phrase is withheld. The same speaker's
next utterance within `phrase_window_s` decides what the withheld words were:

- If it completes a control phrase, alone or joined, the control runs.
- If the joined words could still begin one, they stay withheld. If only the
  new utterance could, it begins a new window with the earlier words ahead of it.
- If it is a STOP on its own, the STOP runs.
- Otherwise the withheld words were speech. In an active conversation they lead
  that utterance's transcript, which is handled as if spoken unsplit: the agent
  query keeps every word, and with `require_wake_phrase` the wake rule applies
  to the joined text. The exception is a standalone wake phrase under
  `require_wake_phrase`: it already opened its follow-up window when withheld,
  so it is not released again.

In every other case the withheld words are discarded: a control or STOP ran,
no conversation is active, the next utterance comes from another speaker, is
empty or failed, or does not start within the window, or the participant
disconnects. Quoting the phrase inside a longer utterance does not activate it.
In conversation and hybrid modes, control phrases never become agent queries or
transcript-topic events. In wake-only and always-on modes, their words are
ordinary text subject to that mode's admission policy, not session controls.

These controls work with the sample's existing batch STT service, including NIM,
or with speaker ASR. Batch STT controls are participant-scoped and cannot identify
people sharing one microphone. Speaker ASR attributes each candidate before
authorization, enrolls only a unique complete start-phrase speaker, and accepts
only that owner while the conversation is open. While closed, ambiguous
simultaneous addressed candidates are discarded, never resolved by label/order.
An exact unique start has priority over concurrent one-off questions. Diagnostic
transcripts are not authorization. Stop releases the owner; reset, departure
and transport failure clear temporary permissions and ownership.
Untopiced typed messages bypass speech gating and are delivered as queries even
while the microphone conversation is closed. They cannot open or close a
microphone conversation.

Set `conversation.require_wake_phrase: true` to require `magic_phrases` during
an active conversation, preserving the optional chime and follow-up grace
window. This is an advanced **active-session** policy, not the switch for closed
one-off questions; use `mode: hybrid` for those. To require addressing for every
turn without a persistent session, choose `wake_only`; choose `always_on` for
ungated/passive input. `speaker.enabled` and `speaker.backend` choose recognition
independently. `auto` may select batch STT at startup if the speaker endpoint is
absent; protocol mismatch is an error, not a fallback. Required speaker ASR never
silently switches to batch after reset.

The speaker service and client use protocol **5** for diagnostics configuration
and explicit attributed one-turn authorization. Update/restart both together;
earlier peers are rejected rather than silently changing transcript display or
authorization.

## Voice gating and early probes

Global STOP recognition is enabled by default. For passive narration, set
`stop_commands_enabled: false` in the voice-gate YAML. This disables STOP
interruptions and acknowledgements for both partial and final transcripts,
including conversations without a wake phrase. Final transcripts still follow
the configured transcript and query-gating rules. Conversation controls and
scoped application commands such as `stop recording` retain their own routing.
Probes can remain enabled for wake acknowledgements; their timing is independent
of the STOP policy.

VAD and STT probe the opening audio while the user is still speaking. Probe
audio includes a silent tail so offline STT can finalize a phrase. During an
active conversation with STOP enabled, a partial global-STOP match interrupts
output immediately, but it does not commit the user's intent: final STT remains
authoritative for global-stop versus query
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

When enabled, global STOP uses a closed imperative grammar for direct requests
such as `stop`, `stop it`, `stop talking`, `be quiet`, and `shut up`, with a bounded set
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

The same conversation controls work with ordinary STT and speaker filtering.
Shipped samples select required speaker ASR alongside `conversation`:

```yaml
speaker:
  enabled: true
  backend: required
  base_url: http://127.0.0.1:8102
```

Before announcing readiness, the worker probes the service's HTTP `/health`
endpoint. A ready compatible service selects diarization. In automatic mode,
connection absence selects an explicitly supplied ordinary `STTService`, including
native Riva speech. Automatic mode requires that real batch client; it is not
the shipped default. `VoiceAgent.from_models(models=..., voice_gate=..., ...)`
uses the already-loaded model profile and the same voice-gate settings as its
sole speech selector. Required speaker mode constructs no batch client;
disabled or automatic mode constructs the existing `stt` role, including native
Riva when configured. The low-level constructor remains available for callers
supplying their own clients. Required speaker mode accepts `VoiceAgent(stt=None)` and
waits even when the service is absent; disabled or automatic speaker mode needs
a batch client. Once a listener is observed, an unresponsive
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
from accepted sample counts; reconnecting starts a fresh timeline and clears
temporary turn permissions and any enrollment. Conversation mode requires
enrollment again. Other voices contribute to
diarization and interference conditioning, but by default are not transcribed after
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

The annotated voice-gate configuration reference above documents
`speaker.diagnostics`, which defaults to `false`. Setting it to `true` displays
candidate transcripts before
enrollment and additionally transcribes non-owner voices after enrollment. It
returns activity and ignored text on `voice.speaker-diagnostic` through the
existing participant-routed data channel. These messages cannot invoke or
interrupt an agent or change enrollment. See
{doc}`../components/ai-services` for decoder ownership and diagnostic resource cost. The
simple VLM sample enables diagnostics for live inspection. They have no effect
when ordinary STT is selected.

The shared voice I/O layer also returns accepted, gate-normalized query text
on the stable `voice.user-query` client data topic. Speaker ASR uses `Speaker N`;
ordinary STT and typed queries use `User`. The application receives the same
typed query as before, without speaker-display metadata or a sample-specific
echo subscriber. Disabling speaker diagnostics does not disable accepted-query
display. Application `query_topic` names remain internal runtime routing, not
client display topics. This is accepted text after wake-prefix/control handling,
not every raw STT word. The separate runtime `voice.transcript` event retains its
existing mode/backend transcript-publication policy. Typed input is echoed too.
Applications may handle or ignore the new
topic in their data callback; the sample clients' catch-all message lists display it.
Participant targeting, not topic selection, provides routing isolation. The
existing hub return path is best effort, not an end-to-end receipt acknowledgment.
Display errors do not reject accepted queries. Accepted display and application
delivery start concurrently within one owned input task and both are awaited;
ignored diagnostic sends are awaited inline. A saturated local hub can therefore
delay input completion or later frames. No display-dropping policy is introduced.

Start the inference process separately from the repository root:

```bash
uv --config-file uv.toml run --project services/speaker-stt \
  python -m speaker_stt --config services/speaker-stt/speaker_stt.yaml
```

Alternatively, each sample-local model stack starts speaker ASR. Stop a
standalone instance before switching to the stack's hardware configuration;
their configuration fingerprints can differ. From the repository root:

```bash
uv --config-file uv.toml run --project agent-samples/simple-vlm-example/model-servers \
  simple_vlm_example_model_servers
```

It composes the `services/speaker-stt/speaker_stt.yaml`
defaults with that sample's `model-servers/yaml/<gpu-profile>.yaml` placement
and capacity overrides, waits for decoder warmup, and reuses matching warm
instances. The sample's model command with `--stop` includes the speaker service. Stop the stack
before changing its speaker-service YAML. Native NIM deployment configures the
worker's ordinary speech adapters for independently deployed Riva endpoints;
disable `speaker.enabled` when choosing that speech path. Refer to
{doc}`/guides/deploying-with-nim`. Automatic selection can use the configured
ordinary STT adapter when speaker ASR is absent at backend selection; once a
listener is observed, it waits for compatible service readiness instead.

The default profile does not start batch STT. To select ordinary STT, add an
`stt` role to the application's `yaml/models.json` and run an STT service for
that role. Consumers can manage batch STT in a custom profile, optionally
alongside speaker ASR. Refer to {doc}`/guides/customizing-model-servers` for
these deployment choices.

Say the exact start phrase to nominate its speaker. Other speech can continue:
the service conditions each candidate transcript on that speaker's target and
interference masks. Candidate phrases finalize on each speaker's configured
inactivity boundary, without requiring global silence. A tie means multiple
exact matches completed in the same accepted audio step, not a claim about
absolute acoustic simultaneity. Split controls retain the same speaker label;
truncated candidate episodes cannot nominate from their continuing tail. No
unqualified candidate transcript dispatches a query before enrollment. Wake-only
and hybrid modes explicitly authorize attributed closed-state one-off turns
without enrollment. After enrollment,
only the selected speaker can send queries or release the conversation. Wake
phrases remain optional through `conversation.require_wake_phrase`; filtering
cannot determine whom the wearer is addressing.

Interaction state belongs to a participant connection. Disconnect, inference
failure or processing overload clears temporary permissions and enrollment;
conversation and hybrid sessions must enroll again to reopen. Per-participant
queues retain at most two seconds or 200 audio
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
Existing mode-less speaker-only YAML retains conversation behavior. Explicit
mode chooses interaction independently of `speaker.enabled` and backend;
wake-only and always-on modes need no persistent enrollment. Typed input
retains its existing behavior.

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
