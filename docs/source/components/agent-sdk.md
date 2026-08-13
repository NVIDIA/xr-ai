<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Agent SDK

The `agent-sdk/` workspace holds the libraries an xr-ai agent is built
from:

- **`xr-ai-agent-runtime`** — agents exposing existing native tools and typed
  publish/subscribe.
- **`xr-ai-models`** — unified service protocols (`LLMService`, `VLMService`,
  `STTService`, `TTSService`, `EmbeddingService`) plus OpenAI-compatible HTTP clients, driven by a
  structured model deployment profile. Swapping a backend is a configuration
  edit, not a code edit.
- **`xr-ai-voice`** — the native voice runtime. `VoiceAgent` publishes
  `UserQuery` and lifecycle events to application-named topics and consumes
  `voice.output`. `VoiceSession` owns readiness, hub transport, voice gating,
  streaming responses, signals, and cleanup.
- **`xr-ai-pipecat`** — the direct Pipecat compatibility surface. Current
  samples use `VoiceSession`; `run_voice_pipeline` remains available to
  external consumers that still assemble Pipecat processors directly.
- **`xr-ai-hub-client`** — the minimal pyzmq + msgpack IPC library every agent uses
  to talk to the XR-Media-Hub (refer to {doc}`server-runtime`). No LiveKit or
  FastAPI dependency.
- **`xr-ai-tools`** — Relay-managed native tools and model tool-call
  workflow helpers.
- **`xr-ai-nat`** — legacy NeMo Agent Toolkit function groups retained while
  their concrete capabilities migrate.

---

## xr-ai-agent-runtime

`AgentRuntime` provides participant-scoped publish/subscribe. An agent exposes
ordinary `Tool` and `AsyncTool` instances from `xr-ai-tools`. Direct callers
use `execute()` or `stream()`, and model loops use the same `ToolSet` and
`handle_tool_call()` path as standalone tools. There is no runtime call
adapter.

Agent lifetime is not itself a runtime concern. Domain controls such as starting
or stopping monitoring remain ordinary tools. Agents are registered before
runtime startup and own their resources, tasks, and concurrency policy. Shared
state that is touched by tools and subscriptions must be protected by the
agent's lock or private queue. Model loops, planning, and memory remain agent
implementations. Raw audio and video remain on the hub path.

`ToolSet.namespaced({"vision": vision.tools, "planner": planner.tools})`
assigns unique model-visible names when tools from multiple agents are combined;
the agents and underlying tools remain unchanged. Participant identity needed
by direct execution belongs in the tool's request schema; participant and
correlation metadata on `RuntimeContext` applies to pub/sub. Agents create,
cancel, and await their own background tasks. `publish()` settles all fan-out
deliveries before propagating subscriber failures.

Relay records each publication and receiving-agent callback as nested runtime
and agent scopes. These scopes include the topic plus message, correlation,
participant, source, and subscriber metadata. Agent-owned work that outlives a
subscription callback starts a fresh Relay scope stack and adds its own agent
scope, preserving logical correlation in metadata without retaining an ended
callback as its parent.

`Topic.telemetry` controls runtime cardinality without changing delivery. Keep
`"full"` for commands, state changes, and lifecycle events. High-volume
transport streams use `"none"`; their receiving agent records one semantic
scope after aggregation. The voice output topic follows this pattern, producing
one `voice.response` scope for either a finite response or a completed stream.
Voice records provider work separately: `voice.stt` covers final and bounded
partial-probe transcription, while `voice.tts` covers each sentence synthesis.
Audio is summarized rather than stored, and playback on the remote client is
outside these spans.

---

## xr-ai-models

Worker code depends on the service protocols and constructs concrete
clients from a model deployment profile — no hand-rolled `httpx` calls in
callers, no model quirks leaking out of this package.

Each profile names the logical models the worker needs and separates adapter,
endpoint, and deployment metadata;
`make_llm(config, "llm")` / `make_vlm` / `make_stt` / `make_tts` return an
object satisfying the matching service protocol regardless of backend or
model-specific quirks (such as reasoning-field naming). Swapping a model is a
config edit, not a code change.

### Quickstart

```python
from xr_ai_models import load_models_config, make_llm, ChatMessage

config = load_models_config("yaml/models.local.json")
async with make_llm(config, "agent_llm") as llm:
    resp = await llm.chat(
        [ChatMessage(role="user", content="hello")],
        max_tokens=128,
        enable_thinking=True,
    )
    print(resp.content, resp.reasoning)
```

`models.local.json`:

```json
{
  "models": {
    "agent_llm": {
      "category": "llm",
      "adapter": {"preset": "nemotron3_nano"},
      "endpoint": {"base_url": "http://localhost:8107", "readiness": "health"},
      "deployment": {"ownership": "reused", "service": "agent-llm"}
    }
  }
}
```

JSON and YAML are both accepted; flat legacy entries remain compatible.

### Built-in presets

Refer to `xr_ai_models/presets/`:

| Preset | Service it targets | Notes |
|---|---|---|
| `cosmos_vlm`     | vlm-server                | image + video; `enable_thinking=false` by default. Video requires vlm-server's `max_videos_per_prompt >= 1` |
| `llama_nemotron` | llama-nemotron-llm-server | OpenAI tool calling via llama3_json (server-side) |
| `nemotron3_nano` | nemotron3-nano-llm-server | reasoning field: `reasoning` |
| `nemotron_omni`  | nemotron-omni-llm-server  | reasoning field: `reasoning_content`, vision + video |
| `parakeet_stt`   | stt-server                | |
| `piper_tts`      | tts/piper                 | |
| `magpie_tts`     | tts/magpie                | |
| `nemotron_embedding` | embedding-server      | dense text embeddings |

### Explicit (no-preset) specification

```yaml
agent_llm:
  kind:       openai_compat
  category:   llm
  base_url:   http://localhost:8107
  model_name: llm
  capabilities: { tool_calls: true, reasoning: true }
  reasoning_field: reasoning
  default_extras:
    chat_template_kwargs: { enable_thinking: false }
  timeout: 60.0
```

`category:` is required when not using a preset.

### Deployment profiles

A structured profile can keep model behavior, endpoint connectivity, and
process ownership in one JSON file consumed by both the worker and a
stdlib-only orchestrator:

```json
{
  "models": {
    "vlm": {
      "adapter": {"preset": "cosmos_vlm"},
      "endpoint": {"base_url": "http://localhost:8100", "readiness": "health"},
      "deployment": {"ownership": "managed", "service": "vlm"}
    }
  }
}
```

The worker passes the file to `load_models_config()`. The orchestrator calls
`load_model_deployment(worker_config)` to map `managed` to an owned process,
`reused` to `launch_mode="reuse"`, and `external` to no local process. Launcher
profiles must use the wrapped nested JSON shape and declare credentials as
`endpoint.api_key_env`; the launcher rejects non-`.json` profiles before
parsing, while flat YAML remains supported for worker-only configs.

Model roles compose `AdapterSpec`, `EndpointSpec`, and `DeploymentSpec`; their
legacy flat attributes remain available as read-only compatibility aliases.

### Protocols

```python
class LLMService(Protocol):
    capabilities: Capabilities
    async def chat(self, messages, *, tools=None, max_tokens=None,
                   temperature=None, enable_thinking=False,
                   thinking_budget=None, timeout=None,
                   headers=None) -> ChatResponse: ...
    def stream(self, messages, *, ...) -> AsyncIterator[str]: ...
    async def health(self) -> bool: ...
    async def close(self) -> None: ...

class VLMService(Protocol):
    capabilities: Capabilities
    async def ask_image(self, image, question, *, system_prompt="",
                        max_tokens=None, temperature=None,
                        timeout=None, headers=None) -> ChatResponse: ...
    async def ask_video(self, video, question, *, system_prompt="",
                        max_tokens=None, temperature=None,
                        timeout=None, headers=None) -> ChatResponse: ...
    def stream(self, image, question, *, system_prompt="",
               max_tokens=None, temperature=None,
               timeout=None, headers=None) -> AsyncIterator[str]: ...
    async def health(self) -> bool: ...

class STTService(Protocol):
    async def transcribe(self, audio: bytes, *, sample_rate=None,
                         channels=1, timeout=None) -> str: ...
    async def health(self) -> bool: ...

class TTSService(Protocol):
    async def synthesize(self, text: str, *, response_format="wav",
                         timeout=None) -> bytes: ...
    async def health(self) -> bool: ...
```

`ChatResponse.reasoning` is the canonical reasoning field — the
`reasoning_field` knob normalizes `reasoning_content` (the nemotron_v3 parser)
into the same surface.

### Remote and hosted-NIM endpoints

Cloud and remote endpoints (e.g. hosted [NVIDIA NIM](https://build.nvidia.com))
are a configuration change — point `base_url` at the OpenAI-compatible URL and set
`api_key_env`:

```yaml
vlm:
  kind:        openai_compat
  category:    vlm
  base_url:    https://integrate.api.nvidia.com
  model_name:  nvidia/cosmos-reason1-7b
  api_key_env: NGC_API_KEY    # → Authorization: Bearer <env value>
  health_check: false         # remote endpoints have no local /health route
```

`api_key_env` names the environment variable holding the API key; its value is
sent as an `Authorization: Bearer <value>` header on every request.

`health_check` (default `true`) gates whether `health()` probes
`base_url/health`. Remote endpoints don't expose that route, so `false` makes
`health()` return `True` without a request — otherwise a worker's readiness
gate would block forever.

Non-OpenAI-compatible backends can be added as new `kind`s without changing the
protocols or callers.

### Tests

The clients can be exercised without a GPU.

---

## Native tools and model tool calls

`xr-ai-tools` is the native migration target for model-driven XR composition.
`Tool` declares Pydantic request and response boundaries and executes its
handler through NeMo Relay. `tool_definitions(...)` adapts a native catalog to
`xr-ai-models` definitions; `handle_tool_call(...)` invokes one model-selected
call and returns the tool-role message.

The package does not implement an agent, model loop, Hub transport, or implicit
conversation store. Applications own prompts, LLM calls, history, iteration,
foreground selection, workflow state, and background work.

`xr_ai_tools.live_vision.LiveVisionTool` is the finite current-frame tool; it
returns one complete observation through Relay's managed tool and LLM
boundaries. `xr_ai_tools.streaming_vision.StreamingVisionTool` is a separate
`AsyncTool` that yields typed chunks around Relay's managed streaming LLM
boundary. It has no voice or output-transport dependency. Both tools own their
own frame sources and redact inline camera data from Relay events without
changing provider input.

Applications place native tools and model loops inside an agent to group tools
with their state, and register the agent with `xr-ai-agent-runtime` when they
need pub/sub. Relay remains responsible for tool and model execution; the
runtime does not duplicate the tool loop or model boundary.

## xr-ai-nat model bridge

Unmigrated workflows install `xr-ai-nat[agents]` when they use NAT's built-in
agent graphs.
`ModelsLLMConfig` adapts an `xr-ai-models` `LLMService` to NAT's LangChain
client contract:

```python
from nat.plugin_api import LLMRef
from xr_ai_nat.llm import ModelsLLMConfig

llm_ref = LLMRef("agent_llm")
await builder.add_llm(llm_ref, ModelsLLMConfig(service=llm))
```

Applications may instead set `profile_path` and `role` for configuration-led
construction. Exactly one source is required. The provider closes clients it
constructs from a profile and leaves injected services under the caller's
lifecycle ownership.

---

## xr-ai-voice

Native voice applications work with participant-aware runtime topics rather
than Pipecat processors. `VoiceAgent` owns `VoiceSession`, publishes accepted
input and lifecycle events with voice-owned schemas on application-named
topics, and subscribes to `voice.output`:

```python
from xr_ai_runtime import Topic
from xr_ai_voice import (
    UserQuery,
    VadConfig,
    VoiceAgent,
    VoiceParticipantLeft,
    VoiceSession,
)

session = VoiceSession(
    stt=stt,
    tts=tts,
    vad=VadConfig(),
    voice_gate=voice_gate_config,
    probes={"vlm": vlm.health},
    ready_file=ready_file,
    closeables=(vlm,),
)
queries = Topic("my-sample.user-query", UserQuery)
participant_left = Topic("my-sample.participant-left", VoiceParticipantLeft)
voice = VoiceAgent(
    session,
    query_topic=queries,
    participant_left_topic=participant_left,
)
runtime.register("voice", voice)
async with runtime:
    await voice.run(runtime)
```

The default hub transport is opened only after readiness probes succeed; failed
readiness closes the session's model clients without opening hub sockets. The
ready file is touched only after the input transport enters its hub IPC receive
loop. `VoiceSession` preserves participant routing, cancels superseded or
interrupted output, installs signal handlers, and closes its transport and
model clients. `VoiceAgent` turns transport lifecycle callbacks into typed
runtime events on agent-owned tasks, so application cleanup cannot block the
shared media processor. It cancels and awaits those tasks during shutdown.
Application agents subscribe and clean up their own state. A pid-less
interruption is a global event.

## xr-ai-pipecat

The unified [Pipecat](https://github.com/pipecat-ai/pipecat) voice pipeline for
xr-ai agents that still use the direct processor API. The top-level entry point
is `make_voice_pipeline`; those workers subclass `BrainProcessor`, hand the
instance to the factory, and run it with `run_voice_pipeline`. Everything else
— VAD/STT, voice gate, streaming TTS — is provided.

### make_voice_pipeline

One call composes the chain and returns the assembled pipeline plus a
`PipelineWorker` ready to run:

```python
from xr_ai_pipecat import make_voice_pipeline, VadConfig

pipeline, worker = make_voice_pipeline(
    transport      = transport,        # XRMediaHubTransport
    stt            = stt,              # STTService  (from xr-ai-models)
    tts            = tts,              # TTSService  (from xr-ai-models)
    brain          = my_brain,         # BrainProcessor subclass
    vad_cfg        = VadConfig(),
    voice_gate_cfg = voice_gate_cfg,   # xr_ai_voicegate.VoiceGateConfig
    text_topic     = "agent.response",
    idle_timeout_secs = None,
)
```

The resulting pipeline is:

```text
input → VadStt → VoiceGate → brain → StreamingTts → output
```

### run_voice_pipeline

Run the returned worker with its transport. For launcher-managed workers, pass
the ready-file callback so it runs only after the input transport has started
the hub IPC receive loop. Participant roster catch-up remains asynchronous;
the worker re-announces its current status periodically so clients that join
or reconnect later converge on the same state. If the callback fails, the
worker is cancelled and the error propagates so the launcher reports startup
failure rather than waiting on a process that cannot signal readiness:

```python
from xr_ai_pipecat import run_voice_pipeline

await run_voice_pipeline(worker, transport, on_ready=ready_file.touch)
```

| Stage | Processor | Role |
|---|---|---|
| input        | `transport.input()`     | inbound microphone audio frames from the hub |
| VAD/STT      | `VadSttProcessor`       | Silero-VAD utterance detection → `STTService.transcribe` → `TranscriptionFrame`; emits start and stop speech frames and a fast-path STOP probe |
| voice gate   | `VoiceGateProcessor`    | wraps `xr_ai_voicegate.VoiceGate`; wake-phrase and stop gating, chime and stop-ack audio |
| brain        | `BrainProcessor`        | the sample-specific reasoning (you subclass this) |
| streaming TTS| `StreamingTtsProcessor` | sentence-batched parallel `TTSService.synthesize`, monotonic playback, per-turn data echo |
| output       | `transport.output()`    | return audio + data back to the hub |

`text_topic` controls the per-turn data-channel echo emitted by the streaming
TTS processor. Set it to `""` to opt out — samples whose brain pushes its own
response data message (e.g. xr-render-demo) want this off to avoid duplicate
sends.

#### The idle-timeout knob

`idle_timeout_secs` controls Pipecat's idle-timeout auto-cancel and is
**disabled by default** (`None`): the pipeline is *never* cancelled for
inactivity, so a quiet session stays connected indefinitely — important for XR
sessions where the user may simply not be speaking. This deliberately overrides
Pipecat's upstream default (`cancel_on_idle_timeout=True`), which would
silently drop idle sessions. Set a positive number of seconds to opt in: the
worker then cancels the pipeline (and its runner) after that long with no
user or bot speech.

### Writing a brain

Subclass `BrainProcessor` and implement `handle_query`. It is a coroutine that
*returns* either a single string (one downstream `TextFrame`) or an async
iterator of strings (one `TextFrame` per chunk — this is how token streaming
reaches TTS). Note it returns the iterator; it is not itself a generator:

```python
from xr_ai_pipecat import BrainProcessor

class MyBrain(BrainProcessor):
    def __init__(self, *, llm, **kw):
        super().__init__(**kw)
        self._llm = llm          # the sample injects its own LLMService

    async def handle_query(self, pid, text, fresh_match):
        # Return the AsyncIterator[str]; the base class consumes it and
        # pushes one TextFrame per chunk. For a non-streaming brain,
        # `return resp.content` (a single string) instead.
        return self._llm.stream([...])
```

The base class owns the per-participant in-flight task, cancellation, and the
lifecycle hooks. Key semantics:

- A new `GatedQueryFrame` supersedes any prior in-flight response for the same
  participant — the prior brain task is cancelled automatically. You cannot
  have two queries in flight for one participant.
- `UserStartedSpeakingFrame` is a **hook only**; it does *not* cancel in-flight
  work. Cancelling on every speech onset would interrupt the agent mid-sentence
  on a follow-up, and any acoustic-echo leak of the agent's own TTS would make
  it cancel itself. The voice gate emits an explicit `InterruptionFrame` when
  the user actually says "stop"; that is the real cancel signal.

Optional overrides (all default to no-op):

| Hook | Fires when | Typical use |
|---|---|---|
| `on_user_started_speaking(pid)` | speech onset | speculative warmup (camera, image fetch) |
| `on_query_superseded(pid)`      | every non-first query for a pid | drain in-flight TTS audio (push `InterruptionFrame`) |
| `on_participant_joined(pid)`    | participant joins | per-pid setup |
| `on_participant_left(pid)`      | participant leaves | per-pid teardown |

### VAD configuration

`VadConfig` mirrors the constructor of `xr_ai_vad.VadDetector`:

| Field | Default | Meaning |
|---|---|---|
| `silence_duration`   | `0.8`  | seconds of silence that finalize an utterance |
| `min_speech`         | `0.15` | minimum speech duration to count as an utterance |
| `silero_threshold`   | `0.5`  | Silero VAD speech-probability threshold |
| `stop_probe_after_s` | `0.4`  | seconds after speech-start to run an early STT pass and check for a STOP phrase; `0` or negative disables the probe |

The early STOP probe lets brief commands ("stop", "be quiet") interrupt the
agent without waiting for the full `silence_duration` finalize window. On a
STOP match the processor pushes an `InterruptionFrame` immediately and lets the
gate handle the canned acknowledgement; the eventual VAD-finalize for the same
utterance is suppressed so the stop-ack does not double.

### Dependencies

`xr-ai-pipecat` builds on `xr-ai-hub-client`, `xr-ai-models`, `xr-ai-vad`,
`xr-ai-voicegate`, and `pipecat-ai`.

---

## xr-ai-hub-client

The lightweight, agent-side IPC library for the XR-Media-Hub. Agents only need
this package — its sole runtime dependencies are `pyzmq` and `msgpack`. The
heavy server runtime (LiveKit, FastAPI, uvicorn) is **not** a dependency, so an
agent process stays small.

### ProcessorEndpoint

`ProcessorEndpoint` connects to the hub's PUB socket to receive real-time video
signals, audio, data, and participant events, and connects a PUSH socket to
send return-data, return-audio, and frame requests back. It works for any
downstream workload — analytics, ML inference, transcription, echo, recording
— not just agentic pipelines.

```python
from xr_ai_hub import ProcessorEndpoint, Subscribe

ep = ProcessorEndpoint(
    sub_addr  = "ipc:///tmp/xr_hub_pub",
    push_addr = "ipc:///tmp/xr_hub_in",
)
ep.on_frame(handle_frame_signal)   # metadata — fires at full frame rate
ep.on_audio(my_audio_handler)
ep.on_data(my_data_handler)
ep.on_participant(handle_participant)  # optional — set is auto-maintained
await ep.run()
```

#### Subscription model

Participants are the unit of subscription. By default the endpoint subscribes
to every participant who joins (and unsubscribes on leave), giving each agent
the full inbound stream — data, audio, and video — for every client. Two knobs
control this:

- `filter` — a `Subscribe` flag that drops whole categories
  (`DATA`, `AUDIO`, and `VIDEO`) at the ZMQ kernel level for efficiency. Default
  is `Subscribe.ALL`. Combine flags with `|` to scope down:

  ```python
  # Audio-only processor; ignores data + video on every pid.
  ep = ProcessorEndpoint(..., filter=Subscribe.AUDIO)
  ```

- `auto_subscribe` — when `True` (default), the endpoint subscribes on join and
  unsubscribes on leave. Set to `False` for agents that service a fixed set of
  participants, then call `subscribe(pid)` yourself (it may be called before
  that participant has even joined — ZMQ holds the subscription until matching
  traffic arrives).

Endpoints created mid-session issue a roster request so they learn about
participants who joined before they did: the hub re-publishes a "joined" event
for every current pid, so already-connected pids are auto-subscribed
retroactively. Because the replays go on the regular `participant` topic, keep
your `on_participant` callbacks idempotent.

#### On-demand frame pixels

Video frame access is two-step, so an agent only pays for the pixels it
actually uses:

1. The `on_frame` callback receives `FrameSignal` metadata (always, at full
   frame rate).
2. Call `await ep.request_frame(signal)` to pull pixel data on demand. The hub
   serves from a small cache and copies pixels only when a request arrives;
   returns `None` if the frame has expired or on timeout. Concurrent requests
   for the same `(participant, track)` are coalesced into one `FRAME_REQUEST`.

#### LiveFrameSource

`LiveFrameSource` is the higher-level live-camera helper for a caller that
needs one fresh raw frame for a participant. It waits for a frame within its
configured freshness window, then returns `FrameData` or raises
`FrameUnavailable`. It deliberately stops at raw pixels: image conversion,
model calls, and PNG export remain the consumer's responsibility. Participant
departure events automatically discard that participant's cached frames and
wake pending requests; wait state is removed when each request completes.

```python
from xr_ai_hub import LiveFrameSource

frames = LiveFrameSource(ep, max_age_s=2.0, timeout_s=5.0)
frame = await frames.get("participant-1")
```

#### Return path

| Method | Sends |
|---|---|
| `send_return_data(msg)`              | a `DataMessage` back to a client (text or binary on a topic) |
| `send_return_audio(chunk)`           | an `AudioChunk` of agent or TTS audio to a client |
| `flush_return_audio(pid)`            | drops audio queued at the hub for `pid` — interrupts the agent's own playback |
| `set_status(status, pid=None)`       | records and publishes *this agent's* status (`"loading"`, `"processing"`, `"idle"`, `"ready"`) on the reserved `_agent.status` channel; broadcasts when `pid` is omitted |
| `mark_ready()`                       | shorthand for `set_status("ready")` — declares this agent available to serve requests |
| `republish_statuses()`               | re-sends each connected participant's current agent status so a missed one-shot update self-heals |
| `wait_for_subscriptions(timeout=5.0)`| blocks until the hub has applied every subscription issued so far; returns `False` on timeout |
| `request_roster()`                   | asks the hub to replay "joined" events for all current pids |

### IPC message types

The codec is msgpack with a small `MsgType` tag. New types can be appended
without breaking existing code.

| `MsgType` | Direction | Meaning |
|---|---|---|
| `FRAME_SIGNAL`       | connector → hub | a decoded frame was written to the shared-memory ring buffer |
| `AUDIO_CHUNK`        | connector → hub | raw PCM audio chunk |
| `CONTROL`            | connector → hub | extensible key/value control message |
| `DATA_MESSAGE`       | connector → hub | LiveKit data-channel payload (routed by topic) |
| `RETURN_AUDIO`       | hub → connector | agent or TTS audio for a specific client |
| `RETURN_DATA`        | hub → connector | agent text or binary for a specific client |
| `PARTICIPANT_EVENT`  | bidirectional   | participant joined or left the room |
| `CONNECTOR_REGISTER` | connector → hub | connector announces itself + its shm name |
| `FRAME_REQUEST`      | processor → hub | request pixel data for a frame |
| `FRAME_DATA`         | hub → processor | pixel data delivered to the requester |
| `RETURN_AUDIO_FLUSH` | processor → hub | drop audio queued for a participant's return track |
| `ROSTER_REQUEST`     | processor → hub | replay joined-events for the current roster |
| `SUBSCRIPTION_PROBE` | processor → hub → processor | round-trip token proving pending SUBSCRIBEs are live |
| `AGENT_PRESENCE`     | processor → hub | readiness participation, plus the participants the agent answers for |

### Readiness contract

`_agent.status` carries one scalar per client, and the hub owns it. Each agent
reports only its own state, tagged with its `agent_id`; the hub folds every
*responsible* agent's state into the status a client sees, taking the least
available: `loading` > `processing` > `idle` > `ready`. An agent that has
registered but not yet reported counts as `loading`, so one ready agent cannot
make the room look ready while another is still starting up or busy.

Two things bound who gets a say:

- **Participation is opt-in.** Only endpoints constructed with
  `announces_readiness=True` register. `ProcessorEndpoint` is the generic
  downstream endpoint — analytics, recorders, MCP servers — and a processor
  that never reports status must not hold clients at `loading`. The voice
  transports opt in; passive processors leave it off, and `set_status` is a
  logged no-op for them.
- **Scope follows subscription.** An endpoint answers for exactly the
  participants it has a live subscription for: every participant when
  `auto_subscribe=True`, otherwise the pids passed to `subscribe()`. Scope
  travels to the hub on `AGENT_PRESENCE` and is re-announced when it changes.
  An agent pinned to one pid neither marks another client ready nor holds that
  client back.

Availability also implies routability. `set_status` publishes only for
participants in scope, and waits behind `wait_for_subscriptions` before doing
so, so a client is told `ready` only once the hub has applied this endpoint's
SUBSCRIBE for it — otherwise the client's first request would land in the
PUB/SUB slow-joiner window and be dropped. The barrier proves issued
subscriptions are live; the scope check proves one was issued for that
participant. Agents that need the barrier for their own purposes can await
`wait_for_subscriptions()` directly.

`agent_id` defaults to `$XR_AI_AGENT_ID`, falling back to a per-process value;
pass `agent_id=` to `ProcessorEndpoint` to set it explicitly.

### Shared memory

`ShmRingBuffer` and `SlotView` give agents that read raw pixels a zero-copy view
into the hub's shared-memory ring buffer. The codec is extensible via
`register_encoder` and `register_decoder` for custom payload types.
