<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Agent SDK

The `agent-sdk/` workspace holds the three libraries an xr-ai agent is built
from:

- **`xr-ai-models`** — unified service protocols (`LLMService`, `VLMService`,
  `STTService`, `TTSService`) plus OpenAI-compatible HTTP clients, driven by a
  `models.yaml` preset configuration. Swapping a backend is a configuration
  edit, not a code edit.
- **`xr-ai-pipecat`** — the unified voice pipeline. One call,
  `make_voice_pipeline`, composes input → VAD/STT → voice gate → brain →
  streaming TTS → output. Sample workers subclass one class (`BrainProcessor`)
  and hand it to the factory.
- **`xr-ai-agent`** — the minimal pyzmq + msgpack IPC library every agent uses
  to talk to the XR-Media-Hub (refer to {doc}`server-runtime`). No LiveKit or
  FastAPI dependency.

---

## xr-ai-models

Worker code depends on the four service protocols and constructs concrete
clients from a `models.yaml` configuration — no hand-rolled `httpx` calls in callers,
no model quirks leaking out of this package.

Each sample's `models.yaml` names the logical models the worker needs;
`make_llm(config, "llm")` / `make_vlm` / `make_stt` / `make_tts` return an
object satisfying the matching service protocol regardless of backend or
model-specific quirks (such as reasoning-field naming). Swapping a model is a
config edit, not a code change.

### Quickstart

```python
from xr_ai_models import load_models_config, make_llm, ChatMessage

config = load_models_config("yaml/models.yaml")
async with make_llm(config, "agent_llm") as llm:
    resp = await llm.chat(
        [ChatMessage(role="user", content="hello")],
        max_tokens=128,
        enable_thinking=True,
    )
    print(resp.content, resp.reasoning)
```

`models.yaml`:

```yaml
agent_llm:
  kind:     preset:nemotron3_nano
  base_url: http://localhost:8107

vlm:
  kind:     preset:cosmos_vlm
  base_url: http://localhost:8100

stt:
  kind:     preset:parakeet_stt
  base_url: http://localhost:8103

tts:
  kind:     preset:piper_tts
  base_url: http://localhost:8105
```

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

### Protocols

```python
class LLMService(Protocol):
    capabilities: Capabilities
    async def chat(self, messages, *, tools=None, max_tokens=None,
                   temperature=None, enable_thinking=False,
                   thinking_budget=None, timeout=None) -> ChatResponse: ...
    def stream(self, messages, *, ...) -> AsyncIterator[str]: ...
    async def health(self) -> bool: ...
    async def close(self) -> None: ...

class VLMService(Protocol):
    capabilities: Capabilities
    async def ask_image(self, image, question, *, system_prompt="",
                        max_tokens=None, temperature=None,
                        timeout=None) -> ChatResponse: ...
    async def ask_video(self, video, question, *, system_prompt="",
                        max_tokens=None, temperature=None,
                        timeout=None) -> ChatResponse: ...
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

## xr-ai-pipecat

The unified [Pipecat](https://github.com/pipecat-ai/pipecat) voice pipeline for
xr-ai agents. The top-level entry point is `make_voice_pipeline`; sample
workers subclass `BrainProcessor` and hand the instance to the factory.
Everything else — VAD/STT, voice gate, streaming TTS — is provided.

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

`xr-ai-pipecat` builds on `xr-ai-agent`, `xr-ai-models`, `xr-ai-vad`,
`xr-ai-voicegate`, and `pipecat-ai`.

---

## xr-ai-agent

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
from xr_ai_agent import ProcessorEndpoint, Subscribe

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

#### Return path

| Method | Sends |
|---|---|
| `send_return_data(msg)`              | a `DataMessage` back to a client (text or binary on a topic) |
| `send_return_audio(chunk)`           | an `AudioChunk` of agent or TTS audio to a client |
| `flush_return_audio(pid)`            | drops audio queued at the hub for `pid` — interrupts the agent's own playback |
| `set_status(status, pid=None)`       | publishes agent status (e.g. `"idle"`, `"processing"`) on the reserved `_agent.status` channel; broadcasts when `pid` is omitted |
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

### Shared memory

`ShmRingBuffer` and `SlotView` give agents that read raw pixels a zero-copy view
into the hub's shared-memory ring buffer. The codec is extensible via
`register_encoder` and `register_decoder` for custom payload types.
