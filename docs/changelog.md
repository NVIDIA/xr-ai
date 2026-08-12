<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Decisions & change log

Significant decisions, in reverse-chronological order. Update this whenever a
non-trivial architectural or design decision is made so the rationale is
preserved and not re-litigated.

### 2026-08-12 — Streaming tools isolate Relay scopes in producer tasks

`AsyncTool` runs each handler in a forked producer task that exclusively owns
its Relay tool scope and closes the handler in that same task and context.
Chunks cross a one-item queue and are yielded in the consumer's context. The
consumer races the queue against producer completion so normal completion,
exceptions (including `BaseExceptionGroup`), and cancellation cannot leave it
waiting forever. Abandoning or cancelling the consumer cancels and awaits the
producer. Voice handlers explicitly close streaming responses, and the simple
VLM adapter also closes its nested tool stream, so cleanup does not depend on
async-generator garbage collection.

This boundary deliberately means callers do not run inside the tool scope and
parent scope-local registrations are not copied into the producer context.
Cancellation records the tool scope as an error; the producer may stay one
chunk ahead of the consumer, and its span may end before the final buffered
chunk is consumed. Producer cleanup remains unbounded: a timeout and detached
cleanup policy requires a separate decision because abandoning cleanup could
leave Relay state or participant status unfinished.

### 2026-08-12 — Voice output is a subscriber, not a privileged side channel

`VoiceAgent` owns `VoiceSession`, subscribes to the typed `voice.output` topic,
and serializes output through its participant-aware delivery path. Producers
may publish one finite message or a sequence of chunks sharing a response ID;
both use the same TTS aggregation, data echo, turn observation, and interruption
path. Voice delivery is one runtime agent among the other application tasks
rather than a privileged dispatcher side channel.

Participant departure and interruption follow the same rule: `VoiceAgent`
publishes voice-owned schemas on application-named topics, and each application
agent handles its own tasks and resources. `app.py` remains
the composition root and contains no transport callbacks or resource logic.

### 2026-08-12 — Agents own existing tools and their concurrency policy

`xr-ai-agent-runtime` defines an `Agent` as a plain object containing private
state and ordinary `Tool` or `AsyncTool` instances from
`xr-ai-tools`. There is no separate function contract, agent address, runtime
call API, or tool adapter. Direct callers and other agents use `execute()` or
`stream()`; model-selected calls use the same `ToolSet` and
`handle_tool_call()` path as standalone tools.

The runtime owns typed, participant-scoped `publish()` fan-out and quiesces only
the in-flight subscription deliveries it creates. Agent resources, background
tasks, and lifecycle remain entirely agent-owned. Tool handlers and subscription
callbacks may run concurrently; each agent owns any lock or private queue needed
for its state instead of receiving a mandatory mailbox. This avoids implicit
head-of-line blocking, especially for streams.

Unary tools now permit `None` for acknowledged side effects. Streaming tools
remain `AsyncTool` instances and are never silently buffered into model tool
results. Agent lifetime is not a model tool; domain controls such as starting
or stopping monitoring remain ordinary tools. Raw media stays on the hub path,
and Relay remains the execution and telemetry boundary around model calls and
tool execution.

### 2026-08-12 — Tool-call handling is not an agent runtime

`agents.py`, `agent_runner.py`, `Agent`, and `AgentRunner` are removed.
`tool_calling.py` only adapts native schemas to model `ToolDef` values and
handles one model-produced `ToolCall` at a time. Applications own prompts, model
calls, history, iteration, and concurrency.

The unused NAT `StreamingVisionConfig` and its schemas are also removed because
`StreamingVisionTool` replaces that surface. The still-used `VisionToolsConfig`
remains in NAT because its recorded-frame tool has no native replacement yet.

### 2026-08-12 — Native tools live outside the NAT compatibility package

The Relay-managed tool modules moved from `xr-ai-nat` to the
dedicated `xr-ai-tools` package. Live vision is split into independent finite
and streaming tool modules. `xr-ai-nat` now remains only as the NeMo Agent
Toolkit compatibility surface during migration.

### 2026-08-12 — Agentic vision is finite; streaming vision is asynchronous

Relay's managed tool API accepts completed JSON results; it does not define a
streaming tool execution contract. XR AI does not reproduce Relay's guardrail
and intercept pipeline around an async generator. `LiveVisionTool` therefore
returns one complete observation through `Tool.execute()` for normal agentic
planning. `StreamingVisionTool` is a separate, transport-independent
`AsyncTool` that yields typed chunks while its provider call uses Relay's
managed streaming LLM API. Applications may adapt that stream to voice or any
other consumer; the tool itself has no voice behavior.

Live camera frames remain provider input but are replaced in emitted Relay
events by a scope-local sanitizer. Relay request-intercept headers cross the
`VLMService` boundary as controlled per-call headers; model-profile credentials
remain non-overridable.

### 2026-08-11 — Native tools own agent composition; Relay owns their execution

NeMo Agent Toolkit is being retired from XR AI in focused migrations rather
than a framework-wide swap. `xr-ai-nat` is the public, toolkit-independent
native tools layer: typed tools, an `AgentRunner` async-turn protocol, and a
bounded default tool loop. A model is a private implementation dependency of a
model-backed tool or agent, reached only through `xr-ai-models`; voice, text,
and background triggers invoke native tools or responders rather than model
clients. NeMo Relay runs finite tool and model lifecycles, supplying middleware,
guardrails, and telemetry. `as_agent_tool` lets a custom or future Fabric-backed
runner use the same registered-tool path without making a framework part of the
public trigger boundary.

The existing NeMo Agent Toolkit function groups remain behind legacy extras
while they migrate. Relay does not own XR application routing, participant
state, media IPC, or deployment. The tea-making sample defines the required
foreground/background application behavior for the migration, while NeMo
Platform and NeMo Fabric remain optional evaluation and harness deployment
targets outside the local worker dependency graph.

### 2026-08-10 — Client readiness is hub-owned and routability-gated

Process readiness and client readiness were the same signal, and both were
wrong at the protocol level.

A worker touched its ready file and broadcast availability from the same point,
before the hub had applied the SUBSCRIBE for a joining participant. A client
acting on that signal could have its first request dropped in the PUB/SUB
slow-joiner window. `set_status` now publishes behind `wait_for_subscriptions`,
a `SUBSCRIPTION_PROBE` round trip through the hub; because ZMQ applies one
socket's subscription commands in order, the echo proves the preceding
SUBSCRIBEs are live. Process readiness is unchanged and still releases on IPC
start — it is a launcher concern, not a client-facing one.

Availability was also per-agent while the client's `_agent.status` is a single
scalar, so in the supported one-hub/many-agents topology a ready agent's
periodic re-announcement could overwrite a busy peer, and a hub-generated
global `loading` could not disambiguate. Ownership moved to the hub: agents
report only their own state tagged with an `agent_id`, announce presence with
`AGENT_PRESENCE`, and the hub folds the room down to the least available state
(`loading` > `processing` > `idle` > `ready`). An attached agent that has not
reported counts as `loading`. Clients keep a scalar and gain no aggregation
logic; payloads without an `agent_id` are forwarded verbatim for older SDKs.

Readiness participation is opt-in (`announces_readiness=True`) and scope
follows subscription. `ProcessorEndpoint` is the generic downstream endpoint,
so registering every instance would let a passive processor — video-MCP,
analytics, a recorder — pin every client at `loading` forever; and an endpoint
pinned to one pid would otherwise mark unrelated clients ready, since the probe
proves only that *issued* subscriptions are live, not that one was issued for
that participant. An endpoint answers for exactly the participants it
subscribes to, and the hub aggregates only those agents.

### 2026-08-11 — Changelog edits don't trigger a docs preview

`docs/changelog.md` is excluded from the `Docs` workflow's path filters. It sits
outside the Sphinx source tree (`docs/source/`) — `reference/changelog.md` only
links out to it on GitHub — so a changelog-only edit rebuilt the whole versioned
site and published a per-PR preview that was byte-identical to the last one.
Since AGENTS.md asks for a changelog entry alongside most changes, that fired on
a large share of PRs. `Build (strict)` is not a required status check, so
skipping it does not gate merges.

### 2026-08-11 — Nightly GPU test runs on an always-on `gpu` runner

The nightly workflow no longer brackets its pytest job with Brev lifecycle
management. The `start` and `stop` jobs, the `brev-controller` runner, the
`BREV_INSTANCE_NAME`/`BREV_ORG` secrets, and the `keep_running` dispatch input
are gone; `pytest` runs directly on `runs-on: gpu` and `notify` depends on it
alone. This supersedes the 2026-06-11 entry below: the GPU host is no longer a
per-run billed instance, so the start/stop bracket bought nothing and added two
failure modes (a start timeout that queued pytest indefinitely, and a stop
failure that had to be loud to avoid a cost leak).

### 2026-08-10 — Voice-worker ready files wait for inbound IPC

`VoiceSession` and the direct `run_voice_pipeline` compatibility path release a
managed worker's ready file only after the input transport has started its hub
IPC receive loop. Participant roster catch-up remains asynchronous, so process
readiness stays a launcher concern rather than a per-client discovery barrier.
The endpoint stores each agent's current status and the pipeline re-announces
that state periodically; a late or reconnecting client therefore converges
without relying on a one-shot event.

### 2026-08-05 — Docker vLLM setup owns the image entrypoint

The shared vLLM Docker launcher explicitly selects `/bin/bash` before installing
model-specific wheels and executing `vllm serve`. The Omni profile's
`vllm/vllm-openai:v0.20.0` image otherwise interprets the setup command through
its default `vllm serve` entrypoint. Failed stopped containers are recreated
rather than restarted because Docker cannot update their recorded entrypoint or
command.

### 2026-08-04 — Model servers select one multimodal stack

`model_servers` defaults to the separate Nemotron-3 Nano and Cosmos services,
with `--omni-stack` selecting Nemotron-3 Nano Omni instead. The launcher keeps
STT and embeddings in both layouts and persists Omni like the other shared
vLLM services, so the launcher can exit after readiness without unloading
weights. `--stop` cleans every stack-specific port without requiring the
original selection.

### 2026-08-03 — Spoken text must carry terminal punctuation

The TTS stage batches on sentence-final punctuation and flushes trailing
fragments only at end of turn, so any mid-turn utterance yielded without
terminal punctuation plays late, concatenated with the final response. The
render demo's spoken quick-ack is punctuation-normalized at the yield site;
whole-message JSON (an echoed tool result) is sanitized before TTS.

### 2026-08-03 — Thinking is reserved for requests the tools can't settle

The quick-ack classifier routes every positional operation — placement,
movement, resizing, recoloring, removal, camera lookups — to the
non-thinking fast path: the spatial-math tools compute exact answers, so
reasoning over them adds latency without improving results. `think: true`
is reserved for vague corrections and free-form compositions no tool
pattern covers. Live-verified on Nemotron-3-Nano-30B: relative placements,
displacement with unit conversion, and midpoint placement all execute
correctly without thinking. The still-working panel ticker is purely
time-gated so slow non-thinking turns get progress updates too.

### 2026-08-03 — xr-render-demo drops the dedicated quick-response LLM

The Llama-3.1-Nemotron-Nano-8B server (port 8106) is no longer part of the
`model-servers` or xr-render-demo stacks. Nemotron-3-Nano-30B is fast enough
for the quick-ack and still-working calls, so the `llm` logical model now
points at the same server as `agent_llm` (port 8107) and the ~16 GiB of VRAM
the 8B held is freed. The standalone `ai-services/llm/llama_nemotron` server
and its `xr-ai-models` preset remain available for samples that want a small
dedicated model.

### 2026-08-03 — Native RAG uses a typed service boundary

Dense document retrieval is a reusable `rag-service` capability exposed to
agents as the native `xr_rag` NAT function group. The service owns document
loading, chunking, content-addressed embedding caches, and retrieval; its
private msgpack/ZMQ transport remains behind typed NAT contracts. Embedding
HTTP is added to `xr-ai-models` alongside the existing model protocols, and a
small persistent vLLM embedding server joins the shared model-server stack.
This replaces the prototype's FastMCP boundary and synchronous HTTP client
without coupling samples to the retrieval implementation.

### 2026-07-31 — GitHub Pages publishes immutable release documentation

The documentation site now uses `sphinx-multiversion` to render `main` as
development documentation and every semantic `v*` tag as a separate release
site subtree. The root URL redirects through `latest/` to the most recently
released stable SemVer tag, falling back to the highest prerelease only when no
stable release exists (or to `main/` until the first release); the version
selector identifies the latest release, links to it from older versions, and
preserves the current page when it exists in the selected version.
Pull requests strictly build the checked-out source and expose an HTML artifact
for review; protected pushes fetch full Git history and strictly render every
published version. Repository links are rewritten while each documentation
version is read, so historical release pages point to their matching tag. Only
`main` and `v*` tags are eligible to deploy, so pull requests remain build-only.

### 2026-07-31 — XR-Media-Hub completes the services layout

XR-Media-Hub now lives at `services/xr-media-hub/` with the repository's
other reusable processes. The move preserves its Python package, distribution,
command, ports, configuration schema, IPC protocol, LiveKit transport, and
recorder behavior. The standalone reference configuration still serves the
repository web client after accounting for the service's additional directory
depth.

Sample orchestrators adopt only the mechanical hub project-path replacement.
The render demo's worker, scene, prompts, configuration, evaluations, and
lifecycle remain unchanged.

### 2026-07-31 — CloudXR runtime joins the services root

The shared CloudXR OpenXR runtime now lives at
`services/cloudxr-runtime/`, beside the other reusable launched services. The
move preserves the `cloudxr-runtime` package, `cloudxr_runtime` command,
configuration, ports, environment-file contract, and runtime behavior while
giving service projects a consistent direct-child layout.

The render demo adopts only the mechanical CloudXR project path in its
orchestrator. Its scene and worker remain unchanged.

### 2026-07-31 — Model-serving projects share the services root

The VLM, STT, LLM, TTS, and embedding server projects now live as direct
children of `services/`. A common project depth makes process declarations,
editable dependency paths, and operational ownership predictable while
preserving each package name, command, port, and serving behavior.

The move normalizes the standalone VLM, STT, Nemotron 3 Nano, Nemotron Omni,
Magpie TTS, and Piper TTS reference configurations to the repository-level
`models/` directory. Their previous ignored caches remain under
`ai-services/models/` or `ai-services/tts/models/` because Git cannot move
untracked weights. Operators must merge those directories before an offline
start; `docs/ai-services.md` documents a non-clobbering migration procedure.
Sample-owned model configurations and the standalone Llama Nemotron and
embedding configurations already resolved to the repository-root cache.

MCP compatibility adapters remain under `agent-mcp-servers/`. The render demo
adopts the mechanical model-service paths in its orchestrator and corrects one
stale path in a YAML comment; its worker, scene, prompts, evaluations, and
architecture are unchanged.

### 2026-07-31 — Separate model behavior, endpoints, and deployment ownership

Model configuration accepts a nested profile with independent `adapter`,
`endpoint`, and `deployment` sections while preserving the flat YAML format.
Workers continue to construct clients through `xr-ai-models`; stdlib-only
orchestrators read only service ownership and credentials through
`load_model_deployment()`. Launcher-visible profiles intentionally use wrapped,
structured JSON: the launcher does not depend on PyYAML or resolve model
presets. The simple VLM sample consumes bundled local and hosted profiles
end-to-end, replacing its separate `model_backend` and `models_yaml` switches.
This keeps endpoint selection and process lifecycle in one profile without
coupling the launcher to the model SDK.

The loader accepts nested JSON or YAML profiles, an optional `models` root,
direct role mappings, and existing flat entries. Legacy flat constructors and
read-only attribute aliases keep current callers compatible. Render profiles
remain unchanged pending their owning refactor.

### 2026-07-30 — Simple VLM adopts the native voice runtime

`simple-vlm-example` is the first sample migrated from direct
`xr-ai-pipecat` assembly to the public `xr-ai-voice` runtime. Its worker is now
a named package with separate entry-point, configuration, application, and
prompt resources. `VoiceSession` owns readiness, ready-file creation, hub
transport, voice-gate processing, signals, turn cancellation, and cleanup,
while the application composes `StreamingVisionConfig` through
`xr_ai_nat.adapters.as_voice_handler`.

Voice, typed text, and `ping` continue through one streaming VLM path;
participant departure releases live-frame state and newer turns interrupt
superseded speech. `VoiceSession` defers its default hub transport until model
readiness succeeds and closes model clients if readiness fails; typed data
outside an active session is ignored. Transcript persistence is not added
because this sample has no conversation-memory behavior to preserve.
`xr-ai-pipecat` remains available for `xr-render-demo` and other unmigrated
consumers.

### 2026-07-30 — Retire the superseded xr-ai-capabilities package

`xr-ai-capabilities` and its unused `VisionModule` are removed after production
samples migrated live and recorded vision behavior to the typed
`StreamingVisionConfig` and `VisionToolsConfig` functions in `xr-ai-nat`.
Keeping both surfaces duplicated frame acquisition and VLM orchestration without
an active consumer. This removal does not add a replacement abstraction:
reusable agent functions remain NAT-first, while application-specific
capabilities stay with their application.

### 2026-07-30 — Adopt the PyNvVideoCodec 2.2 packet contract

`xr-media-hub` and `video-memory-service` now require PyNvVideoCodec 2.2 or
newer. Version 2.2 changed `Encode()` and `EndEncode()` from returning one byte
string to returning a list of packet dictionaries whose `data` values hold the
bitstream. The recorder consumes that packet contract directly; passing the new
result to `bytearray.extend()` dropped every frame with
`TypeError: 'dict' object cannot be interpreted as an integer`. The
historical-video GPU fixture follows the same contract and no longer masks type
errors as unavailable NVENC hardware.

### 2026-07-29 — Hub client renamed: `xr_ai_agent` → `xr_ai_hub` (`xr-ai-hub-client`)

The name `xr_ai_agent` had drifted from what the package is: an agent-side
*client for the hub*, not the agent. It also occupied the `agent-sdk/` root, so
the directory that holds every SDK package was itself one of them — leaving
nowhere to state a dependency rule about the hub client specifically, and
forcing `agent-sdk/pyproject.toml` into unrelated CI cache keys.

The module `xr_ai_agent` becomes `xr_ai_hub`, and its distribution moves from the
`agent-sdk/` root (`xr-ai-agent`) into its own package directory
`agent-sdk/xr-ai-hub-client/` (`xr-ai-hub-client`), leaving `agent-sdk/` a plain
container. In-tree consumers import `xr_ai_hub` and depend on `xr-ai-hub-client`.

**Breaking: the distribution name changed.** `xr-ai-agent` no longer resolves —
anything declaring that dependency must switch to `xr-ai-hub-client`. Every
in-tree consumer is an editable path dependency and is updated in this same
change, and the package is not published, so no compatibility distribution is
shipped. The *import* API is a different matter and is preserved: because
`xr_ai_agent` was a public top-level import, the `xr-ai-hub-client` distribution
ships a deprecated `xr_ai_agent` package that forwards to `xr_ai_hub` and emits a
`DeprecationWarning` on import. That alias only helps once a consumer has already
switched its dependency declaration; it will be removed in a future version.

Two small type-checking tweaks ride along: `cast(bytes, …)` in `_codec` and
`memoryview` None-narrowing in `_shm`. `LiveFrameSource` is carried over
unchanged — it keeps multi-waiter `get()`, `participants()`, and participant-leave
auto-release because the video-mcp live-frame exporter calls `participants()` in
two places, so slimming it here would break a live consumer.

### 2026-07-29 — xr-ai-models internal modules are privatized

The `xr-ai-models` implementation modules `config.py`, `factory.py`,
`openai_compat.py`, and `protocols.py` moved to underscore-private names
(`_config.py`, `_factory.py`, `_openai_compat.py`, `_protocols.py`). The
package's public API is unchanged: every name previously re-exported from
`xr_ai_models` is still importable from the package root
(`from xr_ai_models import VLMService, load_models_config, make_vlm, …`).
In-tree consumers now import from the package root (and tests reach into the
private modules directly for internal helpers). Because the four old module
paths were documented public imports, each is retained as a deprecated
forwarding alias that emits a `DeprecationWarning` on import and re-exports the
canonical objects — including the config names not exposed at the package root
(`KIND_OPENAI_COMPAT`, `ModelKind`, `Category`, `Spec`), which stay importable
from `xr_ai_models.config`; the aliases will be removed in a future version.

### 2026-07-28 — Voice adapters and conversation recall

Added `xr_ai_nat.adapters.voice` (`as_voice_handler`, `record_voice_transcripts`)
— the bridge between native NAT functions and an `xr-ai-voice` `VoiceSession`.
`adapters/voice` imports `xr-ai-voice`, so it is gated behind a new
`xr-ai-nat[voice]` optional extra rather than a hard dependency.

`record_voice_transcripts` is the producer that stores completed turns under
`{participant_id}:user` / `{participant_id}:agent` sources. With that producer
now landing, the `xr_conversation_memory` function group (`recall_conversation`)
— deferred out of the text-memory PR because it had no producer on main — is
re-introduced here, alongside an end-to-end record→recall test. It reads those
role-scoped sources back through the existing typed `query_transcripts` and
returns timestamp-ordered `ConversationEntry` turns for one participant. A real
exchange gives the user turn and the agent turn the same timestamp (both carry
the originating query's time), so recall orders that tie user-before-agent.

Because agents consume these schemas, the whole recall surface is described:
every `ConversationEntry` field and `RecallConversationResult.entries` carry
descriptions, `role` is constrained to `Literal["user", "agent"]` — the only two
values the producer writes — and the config's `text_memory` reference documents
which group recall reads from. A regression asserts the generated request/result
contract keeps its descriptions and the `role` enum.

Both adapters are re-exported from the `xr_ai_nat.adapters` package namespace,
which is the documented import path. The re-export is lazy (PEP 562
`__getattr__`) because the adapters need the optional extra: importing
`xr_ai_nat.adapters` without it still succeeds, and only attribute access
raises — with an error naming the extra to install. Lazy access also keeps the
deprecated `adapters.mcp` alias from emitting its warning on an unrelated import
of the package. The `xr-ai-nat` README documents the `[voice]` extra, both
adapters, `xr_conversation_memory`, and how transcript recording feeds recall.

### 2026-07-28 — Introduce `xr-ai-voice` alongside `xr-ai-pipecat`

Added the `xr-ai-voice` SDK package (`agent-sdk/xr-ai-voice`), a voice runtime
exposing `VoiceSession` plus the `VoiceHandler`/`VoiceQuery`/`VoiceResponse`/
`VoiceTurn` handler surface and `HubVoiceTransport`. It is introduced alongside
the existing `xr-ai-pipecat`; neither package is removed and no sample migrates
onto voice yet (there are no consumers). Readiness is health-based (split across
`_readiness`/`_session`); if #300's request-readiness lands, it folds into
`_readiness` when samples migrate.

The voice runtime anchors every turn to when the participant actually spoke: the
hub's ``AudioChunk.pts_us`` is carried forward on pipecat's presentation
timestamp, ``VadSttProcessor`` captures the value at speech onset and stamps it
onto the transcript, and ``VoiceGateProcessor`` uses it for the dispatched
query's ``pts_us`` (falling back to wall clock only for a transcript with no
originating audio). Stamping wall clock after STT instead baked VAD hangover plus
transcription latency into the timestamp, and that error would persist into
stored transcripts and time-relative recorded-frame lookups. Pipeline shutdown is
also complete: ``EndFrame``/``CancelFrame`` cancels and awaits in-flight handler
turns and tears down pending early-STT probe tasks, so neither a turn nor a probe
can emit text or write a transcript after the session has ended.

The voice runtime is participant-scoped for the repository's one-hub/many-clients
model, and every teardown path is scoped the same way: a ``ParticipantLeftFrame``
releases just the departing participant's synthesis state (before the frame
reaches the output transport, which drops that pid's media sender — otherwise a
lingering synth task could emit audio afterwards and the transport's lazy routing
would recreate the sender it had just released), and a pid-less interruption
drains every participant *and* flushes each of their hub audio rather than only
the fallback target's. `StreamingTtsProcessor` keys its pending text, synthesis/order queue,
sender task, interruption, and hub flush by participant id, so concurrent
participants never share a buffer (which would splice their words) or a sender
(which would misroute audio); an `InterruptionFrame` scopes to its
`transport_source` pid — the handler's supersede interrupt now carries that pid.
Per-participant transport `MediaSender`s are released on `ParticipantLeftFrame`
and every sender task is torn down on pipeline `EndFrame`/`CancelFrame`, so
join/leave churn and shutdown retain no state. The `on_query_superseded` callback
fires only when a new query actually replaces a still-in-flight turn (not for a
queued follow-up or a query after the previous turn finished).

The voice runtime's `VoiceGateProcessor` relies on the wake-gate's partial-wake
/ early-chime helpers (`VoiceGate.begin_utterance`, `wake_ack_enabled`,
`matches_magic_phrase`, `could_match_magic_phrase`, and `play_chime` returning
whether audio was emitted). These are additive superset helpers not yet present
in `xr-ai-voicegate` on main, so `utils/xr-ai-voicegate/{gate,config}.py` are
extended with them in the same change. The additions are backward-compatible:
existing `xr-ai-voicegate`/`xr-ai-pipecat` callers ignore the `play_chime`
return value, and their tests continue to pass.
### 2026-07-27 — Native vision exposes current- and recorded-frame tools

The `xr_ai_nat` vision capability drops the path-based `xr_vision`/`ask_image`
function group in favour of the native `xr_vision_tools` group
(`VisionToolsConfig`), which exposes `look_at_current_frame` over the always-on
live frame source and `look_at_past_frame` over the `video_memory` function
group. `StreamingVisionConfig` (`xr_streaming_vision`) is retained for
simple-vlm's streaming Q&A.

**Breaking (public API):** `VisionFunctionsConfig` — previously exported from
`xr_ai_nat.functions.vision` — is removed and semantically replaced by
`VisionToolsConfig`; the two have different config fields and tool surfaces, so
this is a versioned breaking change rather than a rename (no forwarding alias).
Callers on `VisionFunctionsConfig`/`xr_vision`/`ask_image` must migrate to
`VisionToolsConfig`/`xr_vision_tools`. The `vision/_images.py` → `vision/_pixels.py`
move is an internal (underscore-private) helper rename with no compatibility
surface.

The xr-render worker now consumes the native perception
tools directly: the local `look_at_current_frame` `ToolDef` wrapper and the
`ask_image` two-step were removed from `processors.py`, with the processor
injecting participant identity (and the utterance timestamp for recorded
lookups) that the model never supplies. The no-frame path still ends the turn
with a short spoken message. Camera capture remains always-on streaming — no
camera-on-demand path was reintroduced. vlm-mcp keeps its file-based `ask_image`
MCP tool, now self-contained in the server since the native surface no longer
offers a file-path tool.

### 2026-07-27 — Video-memory contracts inline into the typed client

The model definitions move into `video_memory/_client.py`, aligning the model
surface with the render-subagents branch: request models now extend the shared
`_StrictRequest` base and a `VideoMemoryClient.health()` helper is added. The
client imports `RPCClient` from the canonical `xr_ai_nat.functions._service.rpc`
(#303, merged), and `VideoHealthResult` retains `recording_enabled` so the Video
MCP shim's conditional tool sets and live-only outage fallback stay intact.
`VideoMemoryClient.list_recorded_participants()` keeps a no-argument form
(building the typed request internally, like `get_health()`), so no-argument
callers continue to work.

`video_memory/schemas.py` is retained as a deprecated forwarding alias (emits a
`DeprecationWarning` on import). It preserves the full legacy public surface:
the models whose names are unchanged, plus the renamed ones as aliases — because
their data contracts are unchanged. `ParticipantsResult` →
`ListRecordedParticipantsResult`, `VideoMemoryHealth` → `VideoHealthResult`, and
the legacy no-argument `EmptyRequest` → `ListRecordedParticipantsRequest` (both
field-less strict requests). The package-level `ParticipantsResult` export is
likewise kept as a deprecated alias. Import from
`xr_ai_nat.functions.video_memory` (or its `._client` submodule) going forward;
the aliases will be removed in a future version.

### 2026-07-27 — Text-memory adopts typed request/result models

The native `text_memory` capability now speaks explicit `_StrictRequest`
request models and typed result models instead of returning errors as data.
`add_transcript` rejects blank text at the request boundary
(`Field(min_length=1)` plus a `field_validator`) rather than emitting a
`TextMemoryError`, and `get_transcript_stats` for an unknown source returns a
`TranscriptStatsResult` with `count=0` and null `earliest_us`/`latest_us`
rather than an error. `query_transcripts` and `list_sources` return objects
(`{"segments": …}`, `{"sources": …}`) in place of bare lists. The store and
schemas modules fold into `functions.py`: the private `_store.py` is removed
outright, while `schemas.py` survives as a deprecated forwarding alias (warns on
import). It keeps `TranscriptSegment` (unchanged) and `TranscriptStats` — which
was renamed to `TranscriptStatsResult` but kept the same fields (only
`earliest_us`/`latest_us` widened `int` → `int | None`), so it stays as a
deprecated alias. **Genuine removals** (no alias): `OperationResult` and
`TextMemoryError`, because the typed API validates input and returns typed
results instead of error-as-data. Import from
`xr_ai_nat.functions.text_memory` going forward. The
path-escape guard is preserved: every `.identity`/`.jsonl` path is wrapped in
`_check()` (`resolve()` + `is_relative_to(root)`) before it is read or used,
including the `glob("*.identity")` loops, so a symlinked identity file is never
followed. The `transcript-mcp` compatibility shim keeps republishing the four
legacy tools over the new typed surface; because it no longer applies the
`untyped_outputs` unwrapping, its `query_transcripts` and `list_sources` MCP
outputs are now the typed objects (`{"segments": …}`, `{"sources": …}`) rather
than the bare lists the earlier shim emitted — a deliberate wire-shape change
that aligns the shim with the typed native API. Participant-oriented
`recall_conversation` is deferred to land with its producer (the future
`record_voice_transcripts` writer that stores `{participant_id}:{role}`
sources); the render worker currently writes transcripts under the bare
`participant_id`, so wiring recall now would return empty history.

### 2026-07-27 — MCP export lives under `xr_ai_nat.mcp`

The generic native-function → MCP publisher moved from
`xr_ai_nat.adapters.mcp` to a top-level `xr_ai_nat.mcp` package. The canonical
import is now `xr_ai_nat.mcp.create_mcp_server`; `mcp/` owns exposing native
capabilities to MCP-only agents, leaving `adapters/` for framework adapters.
Public signature is unchanged; only the import path moved. Since
`xr_ai_nat.adapters.mcp.create_mcp_server` was a documented public import, it
remains as a deprecated forwarding alias (it emits a `DeprecationWarning` on
import) to avoid a silent breaking change; the alias will be removed in a
future version.

### 2026-07-27 — Shared service transport and value models under `xr-ai-nat`

The private correlated msgpack/ZMQ transport shared by service-backed functions
moved from `xr_ai_nat.functions._rpc` (a four-module package) to a single
`xr_ai_nat.functions._service.rpc` module; `_service/` owns the private RPC
transport. The shared value models live separately in a capability-neutral
`xr_ai_nat.functions.types`: the coordinate models `Vector3` and `SpatialFrame`
moved there out of `spatial_math/schemas.py`, drawn from a shared, exported
`ServiceResult` base. `spatial_math/schemas.py` is retained as a deprecated
re-export alias (with a `DeprecationWarning`) so existing `spatial_math.schemas`
imports keep working; it will be removed in a future version. `Color` is added
to `functions.types` too as a preparatory shared home; the render scene still
defines and uses its own `Color`/`Vector3`, so migrating that scene schema onto
these types is deferred to a later change.

The RPC wire format and the models' JSON string rendering (`__str__` →
`model_dump_json`) are identical. One deliberate semantic change: the shared
`ServiceResult` base sets `extra="allow"`, so unknown fields are now retained
rather than dropped (the previous coordinate base used Pydantic's default
`extra="ignore"`). This matches the target's intent and only affects inputs
that carry fields outside the model — the spatial-math/tracking call sites pass
exactly the declared fields, so their behaviour is unchanged.

### 2026-07-27 — STT transcription failures are 5xx, not empty 200s

The STT endpoint lets backend exceptions propagate to an HTTP 500 carrying a
stable generic detail ("transcription failed"); the full exception is logged
server-side only, so backend paths and runtime state stay out of responses.
The endpoint used to catch every backend exception and return
200 with an empty transcript, a deliberate guard against NeMo throwing on
very short audio; in practice that guard made a fully broken backend look
healthy while every transcription failed. The voice pipeline catches the
resulting client error, logs it, and drops the utterance, so a session
survives individual failures. Successful transcriptions of silent or
unintelligible audio still return 200 with an empty transcript.

### 2026-07-27 — vLLM container logs are streamed by a supervisor

The docker log streamer waits for the container to exist before attaching
`docker logs -f` and re-attaches (with `--since`) if the stream exits while
the container is still expected. A single unsupervised attach races
`docker run`: attaching before dockerd registers the container writes one
"No such container" line and never recovers, leaving the advertised log
file empty for the whole run.

### 2026-07-27 — HF_TOKEN is required by checkpoint-downloading samples

`model_servers` and `simple_vlm_example` now call
`require_credentials("HF_TOKEN")` and exit with instructions when no token is
found, instead of warning and continuing. Unauthenticated HuggingFace
downloads are rate-limited to the point of stalling indefinitely on the
multi-GB checkpoints, with no error and no progress output, so the previous
one-line warning turned a missing token into an apparent hang on first
launch. The `--allow-anonymous` flag restores the old warn-and-continue
behavior for already-cached weights or deliberate anonymous runs.

### 2026-07-21 — Video memory is recorded history, not live capture

`video-memory-service` reads the H.264 chunks written by XR Media Hub and no
longer subscribes to hub IPC. Callers that need a current frame use
`LiveFrameSource`; Video MCP keeps its legacy live tools by owning that source
until the compatibility adapter is retired. Recorded-frame requests use an
absolute Unix-microsecond reference timestamp plus a whole-second offset so an
agent can reason coarsely while receiving the precise selected timestamp.

### 2026-07-20 — NAT agents retain the xr-ai-models service boundary

`ModelsLLMConfig` registers an `xr-ai-models` `LLMService` as a NAT LLM
provider. Its LangChain client translates NAT agent messages and tools but does
not own model transport, so built-in NAT agents preserve the same deployment
profiles and OpenAI-compatible service seam as direct callers. The provider is
independent of the current XR render loop so future NAT agent workflows can use
the same model-service boundary without a parallel client implementation.

### 2026-07-20 — XR render composes native capabilities directly

The XR render worker builds scene, tracking, spatial-math, vision,
video-memory, and text-memory Functions in one NAT workflow. Runtime-backed
Functions retain typed process boundaries, but the sample no longer launches
or calls MCP adapters. Its existing model tool names and prompt remain stable
while schemas are derived from Functions, keeping this transport migration
independent from the later agent-loop migration. The prompt eval derives its
tool schemas from the same native toolbox assembly and executes effects against
fixtures, so it no longer requires compatibility MCP processes.

### 2026-07-20 — Simple VLM invokes native live vision

`simple-vlm-example` now builds `StreamingVisionConfig` with a NAT
`WorkflowBuilder` and adapts that function to its existing Pipecat voice
pipeline. Live-frame tracking, conversion, and VLM streaming therefore have a
native invocation boundary while voice behavior and model deployment profiles
remain unchanged.

### 2026-07-20 — XR render scene and LOVR stay sample-local

The XR render demo owns its scene state, typed native function groups, LOVR
lifecycle, and Lua app as one vertical component under `scene/`. The component
exposes typed msgpack/ZMQ on port 8320. Render MCP retains port 8220 and its
legacy tool schemas, but is only an adapter; no generic render or scene API is
added to `xr-ai-nat` or the repository-wide services directory.

### 2026-07-20 — Live and recorded video become native video-memory functions

`xr_video_memory` introduces recorded-video queries and timestamp-anchored
frame extraction through `services/video-memory-service`. The service owns
filesystem access and NVDEC; the later 2026-07-21 entry separates live hub
frames into the caller-owned `LiveFrameSource`. Video MCP keeps port 8210 and
its conditional legacy tool list as a compatibility adapter over the private
service at port 8310.

### 2026-07-20 — XR tracking becomes a native function backed by one OpenXR service

`xr_tracking` returns the current user coordinate frame through a private
msgpack/ZMQ service boundary. The service speaks plain dictionaries and the
native client owns typed request and result contracts, keeping hardware code
independent of NAT result models. The long-running OpenXR session lives in
`services/openxr-service`; native agents call the NAT function directly and
the existing OXR MCP server preserves its port and tool schemas as a
compatibility adapter. Port 8330 is reserved for the private service while
port 8230 remains the external MCP endpoint.

### 2026-07-17 — Image question answering becomes a native vision function

The `xr_vision` NAT function group owns local-image normalization and the
`xr-ai-models` VLM call. It accepts an already acquired image path so live and
recorded frame sources remain independently composable. VLM MCP keeps its
existing `ask_image` tool and configuration, but now republishes the native
function through the generic MCP adapter.

### 2026-07-17 — Text history becomes native with optional MCP export

The `xr_text_memory` NAT function group now owns persistent transcript JSONL
storage. Transcript MCP republishes an explicit selection of those functions
under its existing tool names through the generic `xr-ai-nat[mcp]` adapter.
Native agents invoke text memory in-process; the MCP process exists only for
agents that require that protocol.

### 2026-07-17 — Spatial calculations become native NAT functions

`agent-sdk/xr-ai-nat` introduces an `xr_spatial_math` NAT function group whose
inputs include an explicit spatial frame. This keeps deterministic coordinate
math in-process and independent of tracking or transport. Vec MCP and the
spatial tools in OpenXR MCP retain their existing commands and schemas, but now
delegate to the same pure math core; they are compatibility boundaries rather
than owners of the capability. Native function names use calculation verbs,
identify anchors and reference frames explicitly, and include `_meters` on
distance arguments. The six operations all return `Vector3`; containment and
scene mutation remain application responsibilities rather than spatial math.
Each capability module is also its own NAT plugin entry point, avoiding a
package-wide import aggregator as more native capabilities are added.

### 2026-07-02 — Apple client: CloudXR streaming (visionOS-only)

`client-samples/ios-visionos/` runs CloudXR as a second transport alongside its
LiveKit (StreamKit) agent channel — the same dual-plane pattern as the web-xr
client. XR support is currently limited to Apple Vision Pro: the CloudXR
surface on `AppModel` and its `ContentView` controls are `#if os(visionOS)`, so
iOS/iPadOS builds run every non-XR feature (agent, mic, camera, data) with no XR
code compiled in.

- **One CloudXR session, reused across connects.** `AppModel` creates a single
  `CloudXRKit.Session` and reuses it across connect/disconnect cycles (avoiding
  re-`configure()` churn); it is never nilled. `startXR()` pins
  `resolutionPreset = .standardPreset` — the framework default fallback is 4096²
  per eye and OOMs the server compositor. CloudXR connects to the same host as the
  LiveKit hub (`cloudxr_runtime` always runs on that machine in this stack).
- **State is observed in the model, not a view.** `beginObservingXRState()` tracks
  `Session.state` with `withObservationTracking`, re-arming after each change, and
  the UI's computed `xrState` is derived from that, so transitions are caught even
  on the reused session (a view `.onChange` can miss the flip).
- **The render target is the `ImmersiveSpace`.** The `CloudXRSessionComponent`
  lives in `ImmersiveView`'s `RealityView`; if that view disappears while the
  session is connected the SDK has nowhere to deliver frames and never re-binds.
  So the ImmersiveSpace `.onDisappear` stops XR when live, `disconnect()` stops XR
  before tearing down LiveKit, and Stop clears `sessionEntity.children` so the
  framework-parented streaming mesh doesn't linger in the still-open space.
- **`xr.session.started` gates the worker.** Once CloudXR reaches `.connected`,
  `AppModel` publishes on the `xr.session.started` LiveKit topic so `render-mcp`
  launches LOVR exactly as it does for the web client.
- **Entitlement.** `CloudXRKit` wraps NVIDIA's
  [`cloudxr-framework`](https://github.com/NVIDIA/cloudxr-framework) SPM package.
  The visionOS `com.apple.developer.low-latency-streaming` entitlement is required
  for the low-latency path and signs cleanly only under Apple Developer Program
  enrollment; non-ADP teams can strip it locally for higher latency.
- **Server side is untouched.** `auto-webrtc` stays the committed default (web-xr
  is the out-of-box client); Apple-client users flip `NV_DEVICE_PROFILE:
  auto-native` for their run. The two profiles are mutually exclusive per run (a
  CloudXR-side constraint).

### 2026-07-02 — Apple client: CloudXR/LiveKit lifecycle invariants

The XR/LiveKit lifecycle stays correct across an intentional disconnect, a
transient LiveKit reconnect, a coalesced CloudXR state observation, and double
taps. The load-bearing invariants in `AppModel`:

- **Teardown gate.** `disconnect()` sets `isTearingDown` for the whole teardown,
  cancels the mic-recovery task, and `await`s `stopXR()` before closing the
  LiveKit session; `handleCloudxrStateChange` skips mic recovery while
  `isTearingDown` is set, so the asynchronous CloudXR `.disconnected` can't
  republish against a session being nilled. The XR-only stop path leaves the flag
  false and still recovers the mic.
- **`stopXR()` awaits teardown.** It issues `Session.disconnect()` and suspends
  until the CloudXR `.disconnected` transition (via a continuation resumed in
  `handleCloudxrStateChange`), so `disconnect()`'s "release the render target
  before closing the agent channel" ordering actually holds. An already-idle or
  `.error` session returns immediately.
- **`startXR()` is single-flight.** `xrState` stays `.idle` until CloudXRKit flips
  state, so an `isStartingXR` guard stops a double **Launch XR** tap (or a retry
  while `connect()` is suspended) from re-running `configure`/`connect` on the
  reused session.
- **The two transports can't orphan each other.** An unexpected LiveKit
  `.disconnected` (not an intentional `disconnect()`) also tears XR down, so XR
  can't keep streaming with an orphaned CloudXR session and a stale started latch.
- **`xr.session.started` is published robustly.** The latch
  (`hasPublishedXRStarted`) is set only after a `send` succeeds; an
  `isPublishingXRStarted` guard stops overlapping observation ticks from launching
  concurrent publishers. Because `.connected` won't re-fire while the session
  stays up, a bounded retry loop republishes `xr.session.started` until
  `hasPublishedXRStarted` is set. A LiveKit return to `.connected` also re-arms
  the publish, so a transient LiveKit reconnect overlapping XR connect can't
  strand the signal. The latch resets on `.disconnected` so a second XR session
  republishes. Duplicates are safe (the worker and `render-mcp` treat the signal
  idempotently); a missed signal is the real failure.

### 2026-07-02 — Apple client: agent microphone and camera reliability

The agent microphone has to survive two Apple-client hazards: an unconfigured
audio session at first connect, and an interruption (OS or CloudXR) that leaves
capture dead while the UI still shows the mic "on". The camera track has to
survive a transient LiveKit reconnect.

- **Pre-configure the session at launch.** `LiveKitBackend.prepareAudio()` (a new
  public static, called early from `StreamKitSampleApp.init()`) sets
  `AVAudioSession` to `.playAndRecord` / `.voiceChat` and requests mic permission,
  so the first `StreamSession.connect(...)` doesn't publish a silent track against
  `.soloAmbient`.
- **`recoverMic()` is the single recovery entry point.** It runs whenever mic
  intent (`micEnabledByUser`) is on, triggered by an `AVAudioSession` `.ended`
  interruption, a `mediaServicesWereReset` notification, and (on visionOS) the
  CloudXR `.disconnected` transition. The XR-exit case is the hard one: CloudXR's
  NSK layer owns the shared `AVAudioSession` while streaming and tears it (and
  LiveKit's recording engine) down on exit, posting an `AVAudioSession` `.began`
  interruption with NO matching `.ended`, so recovery can't be driven off
  `.ended`.
- **Toggle-only.** Recovery cycles the mic through LiveKit's `AudioManager`
  (`stopAudio()` → `startAudio()`) and never calls `AVAudioSession.setActive`: NSK
  kills LiveKit's recording engine, not just the category, so only a real
  republish rebinds the published track to a live engine. Single-flight via
  `micRecoveryTask`.
- **Bounded settle → stop → start → verify retry loop.** A single fixed
  stop/start can land just as a `.began` arrives and report success while the OS
  still holds capture suspended, so the loop retries past each `.began` until
  teardown finishes.
- **`.began` is a suspension signal only.** It bumps `interruptionBeganGeneration`
  and never triggers recovery (recovering on `.began` would cycle the mic
  mid-interruption). Genuine OS interruptions also post `.ended`; the `.ended`
  path is gated to `micEnabledByUser && xrState == .idle`.
- **An attempt succeeds only when `isAudioActive` is true AND
  `interruptionBeganGeneration` is unchanged across the verify window.** LiveKit
  reports the track unmuted (`didUpdateIsMuted`) even when capture is dead, so the
  interruption generation is the only honest "capture suspended" cue. A changed
  generation means NSK re-suspended capture and the loop retries; an exhausted
  budget marks the mic off rather than claiming a live mic.
- **Intent is split from live state.** `micEnabledByUser` (intent) is separate
  from `isAudioActive` (live) so the UI stays honest when recovery is exhausted.
- **Camera intent survives a reconnect.** A transient LiveKit `.reconnecting`
  drops the camera track to a known-off state but records intent; the track is
  restored when the room reconnects, mirroring how mic intent survives an
  interruption.

### 2026-06-29 — cloudxr_env yaml values default rather than override the environment

cloudxr-runtime applies the `cloudxr_env` block from `cloudxr_runtime.yaml` with
`os.environ.setdefault`, so the yaml supplies defaults and an explicit
environment value (e.g. an inlined `NV_DEVICE_PROFILE=auto-native`) overrides it.
This matches the orchestrator and cloudxr-runtime both resolving the profile
env-first via `read_device_profile`.

### 2026-06-26 — Web client page + vendor build are WebRTC-only

The xr-render-demo serves the static web page and runs the npm web-vendor
build only for WebRTC device profiles. Native CloudXR profiles (`auto-native`,
`apple-vision-pro`, `ipad-pro`) skip both: the page's "Launch XR" uses the WSS
proxy that native already skips, so for native it is a dead WebRTC-only UI that
native clients (the AVP Swift app) never load. Skipping the build drops
the npm dependency for native-only users.

The orchestrator reads `NV_DEVICE_PROFILE` from `cloudxr_runtime.yaml` (stdlib
regex, same as the existing `model_backend`/`lovr_bin` reads), gates
`_ensure_web_vendor()`, and signals the hub via a generic
`XR_MEDIA_HUB_NO_WEB_CLIENT` env var. The hub stays free of CloudXR-profile
knowledge: its config loader honors that transport-agnostic var by clearing
`web_client_dir` (no static mount) while keeping `/token`, `/cert`, and `/rtc`
live. The native AVP app still fetches its LiveKit token from
`https://<host>:8080/token`.

### 2026-06-25 — CloudXR WSS signaling proxy is WebRTC-only

The WSS proxy (port 48322) is started only for WebRTC device profiles.

### 2026-06-11 — iOS/visionOS: pre-warm the LiveKit recording engine on mic start

`LiveKitBackend.startAudio` now calls
`AudioManager.shared.setRecordingAlwaysPreparedMode(true)` (after resetting
`setEngineAvailability(.default)`) *before* `setMicrophone(enabled: true)`.

The symptom was an intermittent mic-publish timeout
(`io.livekit.swift-sdk Code=101 "Timed out"`) on Apple Vision Pro: the publish
arms an `AsyncCompleter` that waits ~5s for the first captured audio frame, but
the `AVAudioEngine` recording path sometimes never started, so no frame ever
arrived and the publish failed. Pre-warming the recording path makes a frame
available immediately, so the completer resolves and the publish succeeds.

Because prepared mode keeps the engine input hot for fast re-enable, the OS mic
indicator (orange dot) stayed lit after the user stopped audio. So `stopAudio`
now drops prepared mode and pins
`setEngineAvailability(isInputAvailable: false, isOutputAvailable: true)` —
input goes down so the dot clears, output stays up for agent playback. (The
disconnect path already clears the dot on its own — `room.disconnect()` releases
the engine — verified on device by disabling a defensive `tearDown` reset and
confirming the dot still cleared, so no extra teardown handling was needed.)
A failed start also rolls the recording engine back (drop prepared mode, pin
input down) so the mic indicator doesn't linger after a failed start, and
`stopAudio` runs that same rollback even if the unpublish throws.
App-side, `AppModel.startAudio` gained an
`isAudioStarting` re-entrancy guard (surfaced as a "Starting…" state and a
locked mic-mode picker in `ContentView`) and reports an honest failure message
instead of silently leaving the UI in a half-started state.

### 2026-06-11 — nightly GPU host paused/restarted via Brev CLI, fail-loud on cost leaks

The `gpu` runner sits on a billed Brev instance, so leaving it up between
the 04:00 UTC nightly runs wastes GPU-hours. The workflow now brackets the
pytest job with Brev lifecycle management on an always-on `brev-controller`
runner: a `start` job idempotently boots the instance and polls `brev ls`
until `RUNNING` before pytest lands on the `gpu` runner, and a `stop` job
(`if: always() && inputs.keep_running != true`) pauses it afterward. A
`keep_running` `workflow_dispatch` input holds the box up for debugging; a
`notify` job opens/updates a single `nightly-failure` tracking issue when
any prior job fails or is cancelled.

The cost-leak guard is the point, so the `stop` job fails loudly rather than
silently skipping: `brev ls` is captured outside the grep pipe (a failing
CLI/auth/network call aborts the step instead of reading as "nothing to
stop"), and any non-`STOPPED` state — including a transient
`STARTING`/`STOPPING` left by a timed-out `start` — triggers `brev stop` and
a re-poll until `STOPPED`. Either failure surfaces through `notify` instead
of leaving the GPU billing unnoticed. `BREV_INSTANCE_NAME`/`BREV_ORG` are
repository **secrets** (not variables); the guards abort if they resolve
empty. The top-level token is `contents: read`, with `notify` overriding to
`issues: write`.
### 2026-06-10 — Sphinx documentation site (isaacteleop-style), scaffold

Stood up a Sphinx documentation site under `docs/source/`, mirroring the
NVIDIA/IsaacTeleop docs setup (NVIDIA Sphinx theme, GitHub Pages publish). It
uses **MyST** so the existing `docs/*.md` content is reused as Markdown pages
rather than rewritten as reStructuredText. Organized into five sections —
Overview, Getting Started, Components, Guides, Reference — each section index
uses a **`:glob:` toctree** so new pages self-register by simply being dropped
into the section directory (this is what lets documentation pages be authored as
independent, individually-mergeable PRs without all touching a shared nav file).

`.github/workflows/docs.yaml` runs a strict build (`sphinx-build -W`) on every
docs PR and deploys to GitHub Pages on push to `main` of the canonical repo
(deploy is a no-op until Pages is enabled in repo settings). `docs/requirements.txt`
pins the toolchain to IsaacTeleop's versions. The existing `docs/*.md` files are
intentionally **kept in place** (the README and in-flight PRs link to them); the
site ingests/ports their content, and a later cleanup can collapse the
duplication once the site is the source of truth. `sphinx-multiversion` is a
deliberate follow-up — single-version first to de-risk CI.

### 2026-06-09 — render-mcp: scene resync survives a LOVR respawn (blocking send, not NOBLOCK)

After a LOVR (re)start, `render-mcp` re-pushed every stored primitive via
`_resync_scene` → `forward()`, which sends with `zmq.NOBLOCK`. But LOVR has not
yet connected its PULL socket at that instant, and a PUSH with zero peers does
not buffer up to `SNDHWM` — it returns `EAGAIN` immediately under `NOBLOCK` — so
the entire resync was silently dropped and the scene came back empty after a
crash/respawn even though `_objects` still held everything. `_resync_scene` now
sends the restore as *blocking* sends (the first waits for LOVR's PULL to
attach, the rest queue up to `SNDHWM`), bounded by `_RESYNC_TIMEOUT_S` so a LOVR
that never connects can't wedge the spawn path. `_lovr_started` is now flipped
True only *after* resync completes (and only if LOVR is still alive), so live
`forward()` ops fast-drop as `not_started` during the resync window instead of
contending on the shared PUSH socket behind the parked send. Fixes #198.

### 2026-06-09 — render-mcp: per-launch context for LOVR respawn (no leaked pipe tasks / log handles)

`render-mcp` parked every LOVR launch's `ManagedProcess` in the
process-lifetime `AsyncExitStack`, so on each respawn the previous context's
teardown (its two `_forward` pipe tasks + open log-file handle) never ran until
whole-process shutdown — N restarts leaked N-1 dead contexts. Each launch now
gets its own `AsyncExitStack`, closed inside `_watch` as soon as the child
exits (before a respawn is allowed); a single `_aclose_live_launch` callback on
the app-lifetime stack covers the shutdown-while-LOVR-running case, so it never
accumulates per launch. Fixes #196.

### 2026-06-09 — Ctrl-C during startup tears down everything, incl. persist + docker containers

Pressing Ctrl-C while `model-servers` was launching (slow image pull / weight
download) left half-started vLLM docker containers running: the user could not
simply abort and retry (e.g. after forgetting `HF_TOKEN`). Two coupled causes,
two coupled fixes.

`run_stack` (`utils/xr-ai-launcher/_stack.py`) treated every interruption like a
clean exit: its `finally` always ran `_shutdown(launched, _no_kill)`, and
model-servers' processes are all `launch_mode="persist"`, so `_no_kill`
contained all of them — abort skipped every one ("keeping alive"). Now it tracks
an `aborted` flag (set in the `except (SystemExit, KeyboardInterrupt)` handler,
and in a generic `except Exception` that re-raises to preserve the traceback)
and calls `_shutdown(launched, no_kill=set() if aborted else _no_kill)`. So a
clean ready-exit (including `exit_after_ready`'s early `return`) keeps persist
alive as before, while an abort kills EVERYTHING — sending SIGTERM to each
wrapper's process group — and then `sys.exit(130)`.

That SIGTERM alone still didn't stop the dockerd-managed container: the docker
wrapper (`utils/xr-ai-vllm/_docker.py` `run`) spawns `docker run` with
`start_new_session=True` (its own session) and installed no signal handler, so
killing the wrapper orphaned the container. The `--stop` path
(`stop_persistent_servers`) was no help — it gates on `/health` 200 and skips
anything not healthy, so it can never reach a mid-download container. That
health-gate is precisely why a wrapper-signal approach was chosen instead: `run`
now installs SIGINT/SIGTERM handlers up front that idempotently (guard flag)
terminate the in-flight `docker` client, `stop_container(name, timeout_s=10)` +
`remove_container(name)` (both by name, health-independent, inside the launcher's
20s SIGKILL window), stop the log streamer, and `sys.exit(130)`. Handlers are
restored once the container reaches ready, so steady-state/`--stop` behavior is
unchanged. The same wrapper backs simple-vlm-example's vlm-server, so its
identical lingering-container bug is fixed for free.

### 2026-06-09 — README: fix invalid CUDA image tag in the GPU smoke-test

The Container Toolkit smoke-test in the README used
`nvidia/cuda:13.0-base`, which is not a published Docker Hub tag —
`nvidia/cuda` tags require a full patch version *and* a distro suffix
(e.g. `13.0.3-base-ubuntu24.04`). The command failed at `docker pull`
(manifest not found) before it could test GPU passthrough at all. Pinned to
`13.0.3-base-ubuntu24.04` (newest 13.0.x, matching the repo's CUDA 13.0
target and the Ubuntu 22.04/24.04 support row above it).

### 2026-06-09 — Launcher: strip a host cuDNN off LD_LIBRARY_PATH before spawning

Each sub-project's venv ships the exact cuDNN its PyTorch was compiled against
(via the `nvidia-cudnn-cu12` wheel). On hosts that export an `LD_LIBRARY_PATH`
pointing at a *different* system cuDNN (common on cloud GPU images), the dynamic
loader found the system copy first and GPU services aborted at torch import with
`RuntimeError: cuDNN version incompatibility: PyTorch was compiled against
(9, 20, 0) but found runtime version (9, 13, 1)`. The launcher's `_spawn` now
sanitizes the child `LD_LIBRARY_PATH`, dropping only the directories that
actually contain a `libcudnn.so*` so the venv-bundled cuDNN wins while every
unrelated entry (CUDA toolkit, driver, app libs) stays put. Done once at the
single point where the child env is built (alongside the existing `VIRTUAL_ENV`
strip), so it covers every launched service; a one-time WARNING records what was
removed. Chosen over editing each service or documenting a manual `unset` so the
stack works out-of-the-box on misconfigured hosts.

### 2026-06-09 — Voice pipeline: idle-timeout auto-cancel off by default, opt-in via YAML

pipecat's `PipelineWorker` defaults to `cancel_on_idle_timeout=True` at
`IDLE_TIMEOUT_SECS`, so an idle voice pipeline is silently cancelled after a
few minutes of no user/bot speech. `make_voice_pipeline` constructed
`PipelineWorker(pipeline)` with no override, so we inherited that — a quiet XR
session (user simply not talking) would drop on its own, which is the wrong
default for this product.

`make_voice_pipeline` now takes `idle_timeout_secs: float | None = None` and is
**disabled by default**: when `None` it passes `cancel_on_idle_timeout=False`
and `cancel_runner_on_idle_timeout=False` so the pipeline is never cancelled
for inactivity. The mechanism is preserved, not deleted — a positive value
opts back in (worker then cancels after that many idle seconds). Surfaced in
each sample's worker YAML as `idle_timeout_secs: 0` (0/unset → disabled) with a
comment, threaded through `simple_vlm_example_worker` and the xr-render-demo
`WorkerConfig`. Documented in `make_voice_pipeline`'s docstring and
`docs/troubleshooting.md`. Tests cover the default-off and opt-in paths at both
the factory and the xr-render config loader.

### 2026-06-09 — Credentials: stop prompting for HF_TOKEN; document it instead

`simple-vlm-example` was the only sample that called
`ensure_credentials("HF_TOKEN")`, which blocks first-run startup on an
interactive `getpass` prompt. The model servers (the heavier, more
download-intensive path) never prompted — they rely on `run_stack`'s automatic
`load_credentials()` (env / `huggingface-cli login` / saved creds). That
asymmetry was confusing, and the prompt isn't warranted: the samples' default
models are **public** (`nvidia/Cosmos-Reason1-7B` is not gated), so `HF_TOKEN`
only raises HuggingFace rate limits / download speed and is strictly required
only for gated models.

Replaced the interactive prompt with a non-blocking path: new
`warn_if_missing(*names)` launcher helper loads any saved/env/CLI token, and if
the token is still absent prints one actionable line (pointing at
`docs/credentials.md`) and continues — it never prompts. `simple-vlm-example`
and `model-servers` now call `warn_if_missing("HF_TOKEN")`. `ensure_credentials`
is unchanged and still used for `NGC_API_KEY`, where the prompt is intentional —
the NIM backend is opt-in and cannot function without the key.
`docs/credentials.md` rewritten to document HF_TOKEN as auto-picked-up +
optional (required-vs-optional spelled out) with the three ways to provide it,
plus pointers from the README quickstart and the `vlm_server.yaml` `hf_token`
field.

### 2026-06-05 — TokenServer: fail startup loudly on bind error; keep shutdown graceful

`TokenServer` ran `self._server.serve()` directly as its task. uvicorn calls
`sys.exit(1)` on a bind failure (e.g. port already in use), so the task ends
with `SystemExit` (a `BaseException`); `stop()`'s `await self._task` then
re-raised it into `LiveKitConnector.stop()`, aborting the remaining
graceful-shutdown steps. Wrapped the task in a `_serve_safe()` coroutine that
catches `SystemExit` and logs a clear "port in use?" error — mirroring the
sibling `WebServer._serve_safe`, which already guards this. To stop a bind
failure from looking healthy, `start()` now awaits the bind (polls
`uvicorn.Server.started` until the serve task either binds or finishes) and
raises `RuntimeError` if the server never bound within `_STARTUP_TIMEOUT_S` —
the token server is the browser-facing auth/signaling entry point, so a dead
endpoint must abort connector startup rather than silently swallow the error.
`_serve_safe` also captures non-`SystemExit` serve failures (so the task's
exception is always retrieved and `start()` chains the real cause), and the
timeout path cancels the orphan serve task before raising. Fixes #192.

### 2026-06-05 — iOS sample: guard switchCamera against concurrent start/switch

Follow-up to #200. `AppModel.switchCamera(to:)` re-invokes the backend's
`startCamera()` (which tears down the active track before publishing the new
one) but, unlike `startCamera()`, took no `isCameraStarting` re-entrancy guard
— so a switch overlapping a concurrent start could interleave backend calls
across `await` suspension points on the main actor. Applied the same
`guard … !isCameraStarting` + `isCameraStarting = true` / `defer` pattern
`startCamera()` uses, so start and switch are serialized. Sample-only, low
severity. Fixes #208.

### 2026-06-05 — Native StreamKit: Android NDK C++20 portability

Native StreamKit no longer depends on C++20 `<format>`. Some Android NDK
libc++ versions used by embedded clients do not ship that header even though
the project otherwise builds as C++20. The affected identity and error-message
strings now use `std::to_string` plus string concatenation, preserving behavior
while keeping the C++ backend portable to those toolchains. The backend also
uses fully qualified enum labels instead of `using enum`, which older NDK r23
compilers reject.

### 2026-06-05 — Native StreamKit: LiveKit room access for receiver-side audio

`LiveKitBackend` now exposes `GetRoom()` for advanced native integrations that
need receiver-side LiveKit APIs, such as rendering remote participant audio
locally or feeding an acoustic echo canceller with the agent's playback audio.
This remains intentionally transport-specific: the generic `StreamingBackend`
surface is unchanged, and callers must opt in by depending on `LiveKitBackend`.
The accessor returns `nullptr` before connect, after disconnect, and in stub
mode.

### 2026-06-05 — Native StreamKit: vector-backed LiveKit AudioFrame construction

`LiveKitBackend::InjectAudioFrame` now constructs `livekit::AudioFrame` through
the vector-data constructor. Some LiveKit C++ SDK builds expose only
`AudioFrame(std::vector<int16_t>, sample_rate, channels, samples_per_channel)`
and not the pointer-data constructor. StreamKit still accepts
`std::span<const int16_t>` at the public `AudioSink` boundary; the conversion is
contained inside the LiveKit backend so callers and custom backends are
unaffected.

### 2026-06-05 — Native StreamKit: publish options for externally captured video

The C++ `LiveKitBackend` publishes camera tracks lazily on the first
`FrameSink::InjectVideoFrame` call because externally captured frames provide
the stream dimensions. That made callers unable to set LiveKit publish-side
encoding options before track creation. `CameraConfig` now includes an optional
`CameraEncodingConfig` with max bitrate, max framerate, and simulcast controls.
The built-in C++ backend stores the config at `StartCamera()` time and applies
it when the first frame creates the `LocalVideoTrack`.

Backends that open and manage their own platform camera may ignore
`CameraConfig::encoding`; it is primarily for C++ hosts that own capture outside
StreamKit and use `FrameSink` for injection.

### 2026-06-05 — Native StreamKit: AudioSink timestamps stay media timestamps

`AudioSink::InjectAudioFrame` carries a capture timestamp in microseconds so
hosts can pass audio and video frames through the same monotonic-clock model.
The LiveKit C++ SDK's `AudioSource::captureFrame` API uses its optional second
argument for a bounded-wait timeout in milliseconds, not for media time. Passing
StreamKit's `timestamp_us` through that parameter can turn an increasing media
timestamp into a long capture wait and add audio latency. The C++ LiveKit backend
now preserves the StreamKit timestamp as API metadata and calls
`AudioSource::captureFrame(frame)` so the SDK uses its realtime default timeout.

### 2026-06-05 — Magpie TTS: honor the launcher's --ready-file contract

The launcher injects `--ready-file <path>` into every spawned process and
blocks in `_wait_ready` (no timeout) until that file appears or the process
exits. Piper and STT touch it after their model loads; Magpie didn't — `run()`
only registered `--config`, so `--ready-file` landed in the ignored unknowns
and `_run` never created the file. Magpie then stays alive serving, so
`proc.poll()` stays `None` too — the launcher deadlocked at startup on the
Magpie TTS service. Mirrored piper/stt: register `--ready-file`, thread it
into `_run`, and `ready_file.touch()` after `_ensure_loaded()` completes
(before `server.serve()`). Fixes #191.

### 2026-06-05 — iOS: reset isCameraActive when a camera switch fails

`AppModel.switchCamera(to:)` only ran on an already-active camera and, on a
failed publish, set `lastError` but left `isCameraActive = true`. Because the
LiveKit backend's `startCamera()` stops the previous track before publishing
the new one, a publish failure mid-switch left nothing streaming while the UI
still showed "Streaming" with a green status and a working Stop button. The
`catch` now sets `isCameraActive = false`, matching the consistency that
`startCamera()`/`stopCamera()` already maintain. Fixes #195.

### 2026-06-05 — pipecat input transport: downmix multi-channel hub audio before resampling

`XRMediaHubInputTransport._on_hub_audio` resampled non-16 kHz hub audio by
passing the int16 PCM straight to `resample_poly` as a 1-D array. For
multi-channel chunks the hub delivers *interleaved* samples (L R L R …), so
the polyphase filter mixed adjacent L/R samples and produced the wrong output
length — corrupting stereo+ audio before STT (the mono common case was fine,
hence latent). Extracted `_hub_pcm_to_mono_16k`, which downmixes to mono
(channel mean) *before* resampling — STT is mono anyway — and the frame is now
emitted with `num_channels=1`. The mono-16 kHz common case is a byte-identical
fast path. Regression test: `tests/test_pipecat_audio_resample.py`. Fixes #193.

### 2026-06-05 — STT: serialize NeMo transcribe() on the shared model

`_AsrBackend._lock` guarded only model *loading*; the hot path `transcribe()`
ran `self._model.transcribe(...)` lock-free. The endpoint dispatches each
`POST /v1/audio/transcriptions` to a thread pool, so concurrent requests
invoked inference on the same NeMo `ASRModel` simultaneously — which is not
re-entrant/thread-safe (shared model buffers, shared CUDA device state),
risking garbled transcriptions or a crash under load. Inference is now wrapped
in the existing lock, mirroring the magpie TTS backend's stated serialization.
`_ensure_loaded()` still runs before the lock (it takes the same non-reentrant
lock for the one-time load). Fixes #199.

### 2026-06-05 — Hub: release held slots before closing a re-registered ring

On `CONNECTOR_REGISTER` for a known `connector_id`, `_handle_registration`
closed and replaced the ring buffer while `_latest_slots` could still hold
`SlotView`s backed by that ring's mmap. Because a live `SlotView` keeps a
sliced memoryview exported, `ShmRingBuffer.close()`'s `self._buf.release()`
raises `BufferError` — leaving the old ring half-closed but still referenced,
so the next `FRAME_SIGNAL` writes through a released/closing buffer
(use-after-close). `_handle_registration` now releases the memoryview and slot
for every `_latest_slots` entry backed by the old ring before closing it,
matching the teardown order already in `close()`. Triggered by a connector
crash/reconnect that re-sends its registration while frames are held.
Regression test: `test_connector_reregistration_releases_held_slots`. Fixes #197.

### 2026-06-05 — Piper TTS: valid empty WAV on empty/whitespace input

`_PiperBackend.synthesize` returned an HTTP 500 for empty/whitespace input.
Piper's `synthesize_wav` sets the WAV format params only on the first
synthesized chunk; empty input produces no chunks, so the params were never
set and `wave.close()` raised `wave.Error: # channels not specified`, which
propagated out of the `/v1/audio/speech` handler unhandled. `synthesize` now
short-circuits empty/whitespace input to a valid, empty (silent) WAV — header
params set, zero audio frames — matching the magpie backend (whose `sf.write`
already emits a valid header for empty audio). The non-empty path is
unchanged. Regression covered by the piper smoke test (`test_piper_tts_smoke`
now also POSTs whitespace input and asserts a 200 + WAV header). Fixes #194.
### 2026-06-05 — piper voice fetch: catch LocalEntryNotFoundError before EntryNotFoundError

Follow-up to #184. That PR added a dedicated `_EXIT_VOICE_UNAVAILABLE = 3`
(retryable → test skips) for the "voice can't be obtained" case, but the
`except LocalEntryNotFoundError` handler was ordered *after*
`except (EntryNotFoundError, RepositoryNotFoundError)`. `LocalEntryNotFoundError`
subclasses `EntryNotFoundError`, so Python's first-match dispatch sent it to the
exit-1 (bad-voice-name) branch and the exit-3 branch was dead code. A transient
HF **429** with no cached copy surfaces as `LocalEntryNotFoundError`, so the
flake #184 was meant to de-flake still hard-failed `test_piper_tts_smoke` on
`main` (run 26995999535). Fix: order the subclass handler first, so the
empty-cache / transient-download case exits 3 and only a genuine bad voice
name (`EntryNotFoundError` from the repo) exits 1 (fail).

With the ordering fixed, `test_piper_tts_smoke` now **skips cleanly** on the
voice-unavailable exit (offline empty cache or transient HF 429) — the smoke
test only asserts the server path when the voice can actually be obtained, so a
HuggingFace hiccup no longer red-fails CI. (An earlier draft of this PR tried to
pre-fetch + cache the voice in CI and fail loudly on a real outage; that was
dropped in favour of the simpler skip — the smoke test isn't worth blocking the
suite on HF availability.)

### 2026-06-05 — Android: synthetic "Virtual Camera" provider over injectVideoFrame

Adds a selectable "Virtual Camera (synthetic)" entry to the Android sample's
camera dropdown that demonstrates the public `StreamSession.injectVideoFrame`
API end-to-end: no physical camera, no CAMERA permission. When selected,
`AppViewModel` runs a coroutine (`SyntheticCameraSource`) generating animated
I420 frames (scrolling colour bars + a bouncing box) into a reused direct
`ByteBuffer` and feeds them at ~30 fps through `session.injectVideoFrame`.

Lifecycle is the subtle part: the synthetic loop is `cancelAndJoin`-ed
*before* `stopCamera()` (otherwise a trailing frame would lazily republish the
injected track after teardown), is gated on a CONNECTED session, and is
cancelled on disconnect so it never calls `injectVideoFrame` on a torn-down
session. The first frame is injected and awaited *before* `isCameraActive`
flips true, so the preview card composes only once the injected track is
published: `CameraPreviewView` reads the non-observable
`session.localCameraTrack` getter a single time, so a track that appeared
*after* composition would never render (this is what left the preview blank).
The injected track is published with `source = CAMERA` so the
in-app preview (`CameraPreviewView`, which reads the CAMERA-source publication)
shows the synthetic frames and the hub treats it as the participant's camera —
this also benefits any other `injectVideoFrame` caller (e.g. external camera
adapters). Always available even on a camera-less device/emulator.

Builds on PR #172 (the `injectVideoFrame` API); not buildable in CI here —
on-device verification is the gate.

### 2026-06-04 — Multi-client isolation: each client talks only to the hub

Two human clients sharing a room saw each other's data-channel messages,
heard each other's microphones, and (on the pipecat voice path) could receive
each other's agent answers. Three independent leaks, fixed at the layer each
belongs to. All client changes are gated on `hubIdentity`
(default `xr-hub-connector`, the connector's join identity); set it
`null`/`nil` to restore the legacy whole-room behaviour.

**1. Outbound data — publish-side (web/web-xr/android/iOS).** `send()` now
addresses data to the hub participant only (`destinationIdentities` /
`identities`), so a client's text/ping/custom messages never reach peers.

**2. Inbound data + audio — subscribe-side (web/web-xr/android/iOS).** The
`DataReceived` handler drops messages whose publisher is not the hub, and the
room connects with auto-subscribe disabled + subscribes only to the hub
participant's tracks (track-published event + a connect-time sweep). A client
no longer receives or plays another participant's microphone — it subscribes
to the hub's per-pid return-audio track only.

**3. Agent return audio — per-participant routing (`xr-ai-pipecat`
foundation).** The pipecat voice pipeline was per-pid on input (VAD/STT/brain
keyed by participant) but collapsed to a single `_target_participant`
(last-join-wins) on output: `XRMediaHubOutputTransport` nulled every frame's
`transport_destination` and routed all TTS through the default sender, so
participant A's spoken-query answer was published on participant B's
return-audio track. Now a per-participant `MediaSender` is registered on
`ParticipantJoinedFrame` (and lazily in `_handle_frame`), frames keep their
`transport_destination = pid`, and `write_audio_frame` addresses the chunk at
the frame's own pid (falling back to `_target_participant` only for
unaddressed audio). Both pipecat samples (simple-vlm, xr-render-demo) inherit
the fix. Covered by `test_output_transport_routes_audio_by_frame_pid_not_single_target`
(fails pre-fix: both answers addressed to the last joiner).

**Known follow-ups (not yet fixed):** within the pipecat foundation,
`StreamingTtsProcessor`'s sentence buffer is still a single shared `_pending`
(two simultaneous speakers' tokens can interleave), interruption is still
global (`InterruptionFrame` → cancel-all rather than pid-scoped, so one
participant's "stop"/supersede cancels another's in-flight response), and
`UserStarted/StoppedSpeaking` frames are pid-less (speculative camera warmup
fans to all joined pids). These degrade only under concurrent speech and are
tracked separately. The native client edits (android/iOS) compile-verify
pending an on-device build; web is syntax-checked and the foundation fix is
unit-tested here.

### 2026-06-04 — Terminate the pid segment on return-traffic topics

The connector subscribed to return traffic on `return_audio.{pid}`,
`return_audio_flush.{pid}`, and `return_data.{pid}` with no delimiter after
the pid (`ipc/_connector.py`). ZMQ `SUBSCRIBE` is a byte-prefix match, so a
connector owning `alice` also matched topics addressed to `alice2` — the same
hazard the processor inbound path already guards against by appending a
trailing `.` (`xr_ai_agent._processor._prefixes`).

**Inert in production, latent elsewhere.** Production runs one connector per
room (`transport/livekit/connector.py`), and that connector subscribes for
every participant in the room anyway, with `RoomClient` routing each return
message to the correct LiveKit participant by payload pid
(`destination_identities` / per-pid `xr-hub-return-{pid}` track). The
over-match delivered nothing the connector wasn't already entitled to. The
leak only manifests in a connector-per-participant topology (what the test
suite constructs), so this is correctness-by-construction hardening, not a
fix for an observed production cross-talk.

**Fix.** Both ends now terminate the pid segment with a trailing `.`. The hub
publishes `return_audio.{pid}.` / `return_audio_flush.{pid}.` (the data topic
was already delimited by its trailing `.{topic}`) and the connector subscribes
with the matching trailing `.`. This assumes participant identities do not
contain the `.` delimiter — the same assumption the processor path already
carries. Added `tests/test_participant_isolation_prefix.py`, which fails on the
pre-fix code (`alice` over-receives `alice2`'s return data/audio/flush) and
passes after. The LiveKit-transport enforcement layer
(`destination_identities`, per-pid return tracks, subscribe permissions)
remains without automated coverage — tracked separately.

### 2026-06-04 — piper TTS smoke test: de-flake + dedicated voice-unavailable exit code

`test_piper_tts_smoke` was failing intermittently in CI with an opaque
"piper_tts_server exited early with code 1" and no further detail. Root
cause: the server downloads the configured voice from HuggingFace on startup,
and `_ensure_voice` only caught the "voice name is wrong" errors —
a transient HF failure (timeout, 429, connection reset) propagated as an
uncaught traceback and exit 1. The test then reported only the exit code
because it never read the subprocess's captured output.

Two fixes:
- **Server**: `_ensure_voice` now catches any other download error and exits
  with a dedicated `_EXIT_VOICE_UNAVAILABLE = 3` (also used for the offline
  empty-cache case), distinct from exit 1 (genuine bad voice name / repo).
  Operators get a clear single line instead of a raw traceback.
- **Test**: `_wait_for_port` reads and surfaces the server's captured
  stdout/stderr on early exit. Exit code 3 → `pytest.skip` (environmental,
  retryable — restores the documented "skip cleanly when the voice can't be
  obtained" contract); any other code → `pytest.fail` with the captured
  output so real regressions are diagnosable in the CI log.

### 2026-06-03 — NVIDIA NIM as a model backend option (LLM + VLM)

Agent samples can now run their LLM and VLM on hosted [NVIDIA
NIM](https://build.nvidia.com) instead of local vLLM, selectable per sample
by config. NIM is OpenAI-compatible, so this rides the existing
`xr-ai-models` client layer — no new `kind`, no worker code changes.

**One code change: `health_check` on every spec (default `true`).** Workers
gate readiness on `service.health()`, which probes `base_url/health`. Hosted
NIM has no such route, so a NIM spec sets `health_check: false` and
`health()` returns `True` without probing. Threaded config → factory →
all four `OpenAICompat*` clients. Chosen over auto-detecting "remote" from
the URL because explicit is safer and self-hosted NIM containers *do* expose
`/v1/health` (operator sets the flag to match their deployment).

**Selection is one config key — `model_backend: local|nim` (no env/CLI
switch, no main.py edits).** Each sample ships a `yaml/models.nim.yaml`
overlay (LLM/VLM → `integrate.api.nvidia.com` with `api_key_env: NGC_API_KEY`;
STT/TTS stay local). Setting `model_backend: nim` in the worker YAML does
everything: the worker loads `models.nim.yaml`, and the orchestrator — which
reads the same key — skips the local model server(s) NIM replaces. The
orchestrator stays stdlib-only by reading the scalar with a regex (the same
technique already used for `lovr_bin`), since orchestrators may not depend on
pyyaml. For xr-render-demo the worker reaches the VLM through `vlm-mcp`, so a
matching `yaml/vlm_mcp_server.nim.yaml` is shipped and the orchestrator points
the `vlm-mcp` process at it automatically in NIM mode. `NGC_API_KEY` is a
managed credential (auto-injected by `run_stack`); the orchestrator prompts
for it once in NIM mode if unset.

**Scope: LLM + VLM only.** Hosted NIM speech (Riva) is not OpenAI
`/v1/audio`-compatible, so STT/TTS remain local. The agentic loop in
xr-render-demo is tuned for the local Nemotron stack; hosted model ids in the
overlay are examples to confirm at build.nvidia.com.

**Security hardening.** Since this feature is the first to ship an
`api_key_env` over a configurable `base_url`, the `OpenAICompat*` clients now
warn at construction when a key would be sent over plain `http://` to a
non-loopback host (cleartext bearer-token transmission, CWE-319); loopback and
`https` are exempt. The shipped overlays use `https://integrate.api.nvidia.com`
so this only trips on a misconfigured self-hosted endpoint. Audit otherwise
clean: keys are read from env and sent only as a header (never logged),
credentials are stored `0600`, `base_url` is operator config (no runtime SSRF
surface), and `httpx` uses `trust_env=False` (no proxy/.netrc token
redirection).

### 2026-06-03 — Removed on-demand camera mode; clients always stream

Dropped the "camera on demand" feature across the stack. Clients now
stream the camera in always-on mode only, and the agent no longer sends
`startCamera`/`stopCamera` control signals on the `clientControl` topic.

**Agent (`agent-samples/simple-vlm-example`).** `SimpleVlmAgent` no
longer requests, holds, or releases the camera. The `frame_max_age_s`,
`camera_on_timeout_s`, and `camera_grace_s` config knobs and the
`clientControl` signalling (plus speculative VAD warmup, freshness
gating, and grace-period stop timers) are gone. `_handle_query` now just
grabs the latest frame for the participant; if none exists yet it replies
"Camera unavailable, please try again."

**Clients (Android, iOS/visionOS, web, web-xr).** Removed the
`cameraOnDemand` setting, its persisted key (iOS), and the "On demand"
toggle UI. The `clientControl` topic is now silently dropped. The manual
`startCamera`/`stopCamera` user controls (camera button) are unchanged.

### 2026-05-28 — xr-render-demo: vec-mcp + redesigned spatial tools + prompt redesign + eval vocab audit

The eval score on the agentic-loop suite climbed from 40/66 to 58/66 by
splitting "vector arithmetic the model is bad at" out of the prompt and
into a deterministic tool surface, then redesigning the prompt around
the new tools.

**New `vec-mcp` server (port 8250, pure FastMCP).** Lives at
`agent-mcp-servers/vec-mcp/` and exposes four pose-independent math
primitives: `between_anchors`, `world_offset`, `along_direction`,
`scale_value`. Greenfield rather than an extension of `oxr-mcp` because
none of these need head pose and forcing them through oxr-mcp would
silently couple their availability to the headless OpenXR session
opening. The split also lets `vec-mcp` stay a `uvicorn + fastmcp +
pyyaml`-only dep so it can be reused by future samples that have no XR
component at all.

**`oxr-mcp` gains named-direction helpers.** Adds `place_user_relative`,
`place_object_relative`, `place_inside_by_id`, `displace_object`,
`displace_objects`. Each takes a `direction` enum (`front`/`back`/
`left`/`right`/`above`/`below`, plus `next_to` on
`place_object_relative`) and an always-positive `distance`. The LLM
never applies signs to user-frame axes, which was the dominant
failure mode on the previous `position_relative`-only surface.
`displace_objects` is the batch variant ("move them all 1 m forward")
that collapses N math calls into one. `place_inside_by_id` uses
deliberately split argument names (`movee_id` paired with
`container_*` rather than the more natural `origin_*`) so "put X in Y"
parses unambiguously. The previous `position_relative(origin=…)`
overload made the model pick the wrong noun's coords ~30 % of the time.

**`place_object_relative` `direction="front"` means *toward the user*,
not "away from the user".** Counter-intuitive but matches user English
("in front of <obj>" = the side of <obj> closer to the user). The
inverted convention is the most common bug for the model to learn;
documented in the docstring and reinforced with a worked example
covering the "Push it away from me" case (which is `direction="back"`).

**Prompt redesign: worked-example heavy, three-check ladder, reserved
vocab.** The new `system.txt` opens with explicit pronoun-resolution
rules then routes placement utterances through three sequential checks
(FIRST: `between`/`middle`/`halfway` → `between_anchors`; SECOND:
anchor is the user → `place_user_relative`; THIRD: proximity to a
named object → `along_direction`) before the LLM picks a tool. Every
non-obvious rule has a paired WORKED EXAMPLE with concrete coords; the
highest-leakage failure modes get WORKED ANTI-EXAMPLEs.

Worked-example fixtures use a *reserved vocabulary* (cones, cylinders,
capsules, magenta/teal/turquoise) that is disjoint from the eval cases'
vocabulary (spheres, boxes, pyramids, red/green/blue/cyan/yellow/brown).
This keeps the eval honest; see the audit below.

**Eval-harness vocab-leakage audit.** `eval.py` gained
`_check_prompt_eval_overlap`, which audits the system prompt's
worked-example blocks against every case fixture at startup. Four
checks: verbatim user utterances (≥12 chars), scene coordinates,
`recent_moves` coords, and reserved-vocab collisions (any colour/shape
word appearing both in a case fixture *and* in a worked-example block
of `system.txt`). Per `AGENTS.md` "Prompt-driven samples", the
warnings surface at every run; `--strict-overlap` turns them into a
CI-grade rc=2 failure. Clearing a warning means changing the prompt's
worked example, not the case fixture.
### 2026-05-21 — `xr-ai-vad` is Silero-only; `xr-render-demo` migrated

Dropped the adaptive-energy fallback path that shipped in the initial
`utils/xr-ai-vad/` introduction and migrated `agent-samples/xr-render-demo`
to use the shared detector.

**Silero-only, no fallback.** The initial cut shipped Silero with an
adaptive-energy gate kicking in if `silero-vad` failed to load. That extra
codepath came with its own knobs (`silence_threshold`, `vad_noise_mult`),
its own state (`_noise_floor`), and silently degraded the worker behaviour
when something went wrong with the model. The fallback wasn't asked for;
the detector now raises at construction if the model can't be loaded so the
failure is loud, and the public surface shrinks to just Silero knobs
(`silero_threshold`, `silence_duration`, `min_speech`).

**Canonical input is int16 PCM.** Previously `feed()` took float32 LE
bytes plus an explicit sample count so it matched
`AudioChunk.data`. Pipecat's `InputAudioRawFrame.audio` is int16 at
16 kHz, and converting to int16 is the natural buffering format (the
detector emits int16 in `on_utterance`). The signature is now
`feed(pcm_int16, sample_rate)`; simple-vlm-example does the
trivial float32→int16 conversion at the call site.

**`xr-render-demo` migrated.** Its previous bespoke
`SttProcessor._feed` duplicated the detector loop. It now constructs a
`VadDetector`, feeds Pipecat's int16 PCM through it, and runs STT +
filler filtering + `TranscriptionFrame` push from the
`on_utterance` callback. Both samples now share one Silero implementation
behind `xr-ai-vad`.

### 2026-05-21 — `utils/xr-ai-vad/` shared Silero VAD utility

Extracted the Silero-VAD utterance detector into a new shared utility
package and migrated `agent-samples/simple-vlm-example` to use it,
replacing the worker's inline RMS energy gate.

**Why a shared `utils/` package (not per-sample copies).** Silero VAD
already exists in two trees on different branches — `xr-render-demo`
(pipecat-coupled, on main) and `glasses-agent-nat` (clean async API,
on a separate branch in active dev). Adding it to `simple-vlm-example`
made a third near-identical copy the obvious next step. The shape is
small and stable enough (one class, two async callbacks) to live behind
one API; copies were the wrong default.

**Why `utils/` not `agent-sdk/`.** `agent-sdk/xr-ai-models/` is the
HTTP-client seam for AI inference services. Local DSP on raw PCM bytes
doesn't fit there. `utils/xr-ai-vad/` mirrors the shape of
`utils/xr-ai-logging/`: focused dependency footprint (numpy +
silero-vad), opt-in via per-sample `[tool.uv.sources]`, no leak into
samples that don't process voice.

**`on_speech_start` hook added vs. the glasses original.** The previous
`simple-vlm-example` VAD path fired a speculative camera-warmup the
moment `speech_s` crossed `min_speech`. The callback-only finalize API
in the glasses original had no equivalent, so the migration would have
been a behavior regression. Added a one-shot per-utterance
`on_speech_start` callback to preserve it.

### 2026-05-20 — Native StreamKit: `AudioSink` mixin + `CameraConfig::Facing` contract (#134)

Two design decisions in response to partner findings on the native C++
StreamKit integration (PR #131 → issue #134):

**`AudioSink` mixin alongside `FrameSink`.** The C++ SDK ships no built-in
mic capture and the previous path required subclassing `LiveKitBackend` to
reach the private `audio_source_` member — fragile and tied to
implementation details. Added a public `AudioSink` interface with a single
`InjectAudioFrame(pcm, rate, channels, samples_per_channel, ts)` entry
point; `LiveKitBackend` now implements it. Shape mirrors `FrameSink` (and
Swift's `AudioInjectable`) rather than introducing a unified `MediaSink`
because audio and video have different validation rules, different real-time
characteristics, and different zero-copy stories — collapsing them produces
a leaky abstraction at no readability gain. No zero-copy `&&` overload on
audio: a 10 ms @ 48 kHz mono PCM frame is ~960 bytes, well below the 1.4 MB
per-frame threshold that justified `FrameSink`'s second overload.

**`CameraConfig::Facing` documented as built-in-camera-open-only.** The
`facing` / `device_id` fields are honoured by backends that open a camera
themselves (iOS / Android / Web) and inert in the built-in C++ backend,
which has no portable camera-open path. Considered splitting `CameraConfig`
per-platform; rejected because it would fork the cross-platform shape that
every other client depends on. Kept the struct identical everywhere and
made the contract explicit in `CameraConfig.h` and `StreamingBackend.h`
docstrings — silently inert on backends that can't act on it.

### 2026-05-20 — Hub releases held ring-buffer slots on participant leave (#143)

`HubEndpoint` holds the latest SHM ring slot per `(participant_id,
track_id)` so processors can fetch pixels on demand without an eager
copy. The slot was only released when the *next* FRAME_SIGNAL for the
same key arrived — so when a track ended (LiveKit `track_unsubscribed`
or `participant_disconnected`), its last slot stayed held forever.
After enough connect/publish/disconnect cycles the ring filled with
abandoned slots and every subsequent frame from any participant was
dropped at the connector with `Ring buffer full — dropped frame`.
A new participant could publish video and the worker would log
`tracks_seen=0` until the hub was restarted.

**Fix.** The hub now releases every slot keyed by a participant when
that participant's `PARTICIPANT_EVENT(joined=False)` arrives. The
`notify_participant_left` path already fires on both `track_unsubscribed`
+ `participant_disconnected` (it is called from `_room_client._handle_left`),
so no new message type is required. Reuses the established
"release_slot without `view.data.release()`" pattern from the FRAME_SIGNAL
branch — the connector's ring is still live at this point, so the
memoryview does not need an explicit release.

Also bumped `_DEFAULT_NUM_SLOTS` from 10 → 16 so a single ill-timed
reconnect within the in-flight window between last frame and disconnect
event can't still hit the ceiling. The slot is 12.4 MiB at the 4K NV12
ceiling, so six extra slots adds ~75 MiB to the worst-case per-connector
shm footprint — cheap insurance.

**Out of scope.** Connector crash / OOM still leaks slots — `joined=False`
is only emitted by the live `notify_participant_left` path, not by an
unclean exit. A heartbeat or TTL scan would close that gap; deferred
until there's evidence it matters in practice. The issue's reproduction
is wholly covered by participant disconnect.

### 2026-05-18 — `nightly XR AI test` workflow on self-hosted `gpu` runner

The `gpu`-marked pytest suite (`tests/test_gpu_*.py`,
`tests/test_integration_livekit.py`, `tests/test_local_render_mcp.py`)
is filtered out of the default `tests` workflow because it needs real
GPU / Docker / NVENC hardware that the `ubuntu-latest` runners don't
provide — see the 2026-05-12 entry for why the marker was introduced.
Until now the suite ran only via `tests/run_local_gpu_tests.sh` on
developer boxes, which meant regressions could slip into `main` between
ad-hoc local runs.

A new `.github/workflows/nightly-xr-ai-test.yml` runs the same suite at
04:00 UTC every day (and on-demand via `workflow_dispatch`) on a
self-hosted runner registered with the `gpu` label. The job mirrors
`tests.yml` — `uv sync` + `pytest -m gpu` from `tests/` — at Python
3.12 only, since these tests are GPU-bound rather than
Python-version-bound and doubling the matrix would just double GPU-hour
cost without new coverage. Concurrency is set to queue (not cancel)
overlapping runs so a long nightly finishes before the next cron fires.

**Runner hygiene.** The `gpu` label points at a persistent host, so
state leaks across runs. Two layers of cleanup keep the suite robust:

1. A workflow-level pre/post step force-removes every container named
   `xr-ai-vllm-*` (the prefix the launcher uses) and asserts ≥ 30 GiB
   of GPU memory is free at start — a clear hard error here beats a
   confusing downstream vLLM OOM.
2. An autouse pytest fixture in `tests/conftest.py` does the same scrub
   around each `@pytest.mark.gpu` test. Per-test `finally` blocks
   already call `stop_persistent_servers`, but those don't run if
   pytest itself is killed or a fixture errors out — and any leak
   between LLM tests on a single 46 GiB GPU OOMs the next one.

**CUDA toolkit discovery.** A discovery step picks the toolkit from a
priority list (`/usr/local/cuda-13.0`, `…-13`, `…`, `$CUDA_HOME`, then
`which nvcc`) and exports `CUDA_HOME` plus `CUDACXX` via `$GITHUB_ENV`
so downstream JIT compilers (FlashInfer, torch.cpp_extension, cmake)
stop guessing. A follow-up step asserts the chosen `nvcc` supports
`compute_89`, failing the run early on a misconfigured host.

**vLLM tests use the docker backend.** Hosting vLLM in
`nvcr.io/nvidia/vllm:26.04-py3` instead of pip means the host's CUDA /
FlashInfer JIT toolchain (or its absence) no longer affects the tests
— the container ships nvcc and a working FlashInfer build. The image's
JIT cache lives at the runner's `/ephemeral/cache/flashinfer/...`,
which is invisible to per-test setup; switching to the container side-
steps it entirely.

**`extra_pip` seam in `xr_ai_vllm.serve`.** The launcher already pip-
installed `hf_transfer` into the NGC container before `vllm serve` ran
(the image hard-errors with `HF_HUB_ENABLE_HF_TRANSFER=1` otherwise).
That seam is generalised: `serve(..., extra_pip=[...])` is threaded
through `_docker.run` → `build_run_argv` and appended to the same
`pip install -q ... && vllm serve ...` shell line. `nemotron_omni`
defaults `extra_pip=["mamba-ssm", "causal-conv1d"]` so its hybrid SSM
backbone — which the NGC image doesn't bundle — loads cleanly. The
knob is `cfg["extra_pip"]`-overridable for version pinning. pip-mode
silently ignores it (deps belong in `pyproject.toml` there).

### 2026-05-14 — `xr-ai-models` seam adopted by migrated workers; one migration pending

`vlm-mcp` (#139), `xr-render-demo` (#140), and `xr-ai-pipecat` (#137) now
depend on `agent-sdk/xr-ai-models` and construct their LLM / VLM / STT / TTS
clients from a per-sample `yaml/models.yaml` via `make_llm` / `make_vlm` /
`make_stt` / `make_tts`.  Per-model quirks (`chat_template_kwargs.enable_thinking`,
`thinking_budget`, `reasoning` vs `reasoning_content` field naming,
served-model-name strings) live in built-in presets — no caller branches on
backend, and swapping a model is a `kind:` + `base_url:` YAML edit.

The seam is consumed by `vlm-mcp` (#139), `xr-render-demo` (#140), and
`xr-ai-pipecat` (#137); `simple-vlm-example`'s worker still uses inline
`httpx` callers and migrates in #138.  The AGENTS.md hard rule
"All HTTP calls to AI services go through `agent-sdk/xr-ai-models`"
becomes universally enforceable on review once #138 lands.

Top-level docs (`README.md`, `docs/ai-services.md`,
`docs/adding-a-sample.md`, `AGENTS.md`) surface the `models.yaml` convention
in the new-sample checklist and the per-service call examples.

See PR #135 (Unit 1, SDK) and Units 2–5 (consumer migrations) for the
individual diffs.

### 2026-05-14 — Introduce `agent-sdk/xr-ai-models` SDK; collapse hand-rolled httpx clients behind four protocols

Before this change, every consumer of an AI service rolled its own `httpx`
wrapper around `/v1/chat/completions` / `/v1/audio/transcriptions` /
`/v1/audio/speech` — `VlmClient` existed in three places (vlm-mcp,
simple-vlm-example/worker/services.py, xr-render-demo/worker/processors.py
where four inline `httpx.post(.../v1/chat/completions)` sites duplicated the
OpenAI request shape); `SttClient` / `TtsClient` lived in both
xr-ai-pipecat and simple-vlm-example.  Per-model quirks
(`chat_template_kwargs.enable_thinking`, `thinking_budget=1024` in the
agentic loop, the `reasoning` vs `reasoning_content` field-name difference
between vLLM's `nano_v3` and `nemotron_v3` reasoning parsers) leaked into
every caller.  Swapping a model meant editing N files.

The new `agent-sdk/xr-ai-models/` package introduces four service
protocols — `LLMService`, `VLMService`, `STTService`, `TTSService` — and
one `OpenAICompat*` implementation per protocol that covers every in-tree
backend (vLLM-served VLM/LLMs, NeMo Parakeet STT, Piper/Magpie TTS) and
any future OpenAI-compatible endpoint.  Worker code constructs services
from a per-sample `yaml/models.yaml` via `make_llm` / `make_vlm` /
`make_stt` / `make_tts`; built-in presets (`cosmos_vlm`,
`llama_nemotron`, `nemotron3_nano`, `nemotron_omni`, `parakeet_stt`,
`piper_tts`, `magpie_tts`) pre-fill the model-specific quirks so a sample
entry only needs `kind: preset:<name>` + `base_url:`.

`ChatResponse.reasoning` is the canonical reasoning surface; the
`reasoning_field` knob normalizes `reasoning_content` (nemotron_v3) into
that one name so callers do not branch.  `enable_thinking` and
`thinking_budget` are typed kwargs on `chat()` that flatten into
`chat_template_kwargs` on the wire — callers never construct that dict.

**Wire-format note (vlm-mcp migration, #139).** Pre-migration, an
explicit `enable_thinking=True` from a caller produced a request with no
`chat_template_kwargs` key at all (the legacy `VlmClient` only emitted
the key when *false*). Post-migration the SDK always emits
`chat_template_kwargs: {"enable_thinking": <bool>}`. Functionally
equivalent — the model still generates `<think>` tokens — but worth
recording for anyone bisecting wire traces across the migration boundary.

**Why not LiteLLM or any-llm-sdk.** Both are excellent for cross-vendor
fan-out but solve a problem we do not have yet — every in-tree backend
already speaks OpenAI-compatible HTTP, and both libraries pass our most
painful quirk (the reasoning-field name) straight through; we would still
write the normalization layer on top.  They would also pull `openai`,
`pydantic`, `tiktoken`, and friends into every worker venv.  The
`factory.py::make_*` `kind` dispatch is the seam where a `LiteLLMBackend`
slots in as a new `kind` later if/when Phase B brings true cross-vendor
needs — protocols and callers do not change.

This is Unit 1 of a multi-PR refactor.  Subsequent units migrate
vlm-mcp, simple-vlm-example, xr-render-demo, and xr-ai-pipecat to depend
on `xr-ai-models` instead of rolling their own clients.

`VLMService` also exposes `ask_video(video, question)`, mirroring
`ask_image`.  The wire format is a `{"type": "video_url", "video_url":
{...}}` content part — what vLLM's Qwen2.5-VL serving expects when
`--limit-mm-per-prompt {"video": >=1}` is set.  `cosmos_vlm` declares
`capabilities: { vision, video }` because Cosmos-Reason1-7B is a
Qwen2.5-VL fine-tune primarily designed for video reasoning; video is
opt-in at the server because vLLM reserves tens of GiB of activation
memory for it at startup.  Callers that haven't enabled video on their
spec get a `ValueError` from `ask_video` rather than a silent server
500.

### 2026-05-14 — CodeQL Advanced Setup (committed workflow) instead of Default Setup

`Analyze (python)` and `Analyze (javascript-typescript)` are required status
checks on `main`. With GitHub's Default Setup (the implicit
`dynamic/github-code-scanning/codeql` workflow), CodeQL only runs when a PR's
diff touches a configured language — so a PR whose diff is entirely C++/CMake
or docs silently skipped analysis and the required contexts never reported,
leaving the PR permanently `mergeStateStatus=BLOCKED` even with an approval
and all visible checks green (first hit by PR #131). We've committed
`.github/workflows/codeql.yml` so the matrix runs on every PR and posts both
required contexts unconditionally. Default Setup must stay disabled in
`Settings → Code security → Code scanning` — Advanced Setup and Default Setup
cannot coexist.

### 2026-05-12 — `FrameSink::InjectVideoFrame` gains a zero-copy overload

The span-only `InjectVideoFrame(std::span<const std::byte>, …)` forced backends to allocate + memcpy the entire pixel buffer on every frame before they could construct an owning `livekit::VideoFrame`. On an embedded 32-bit ARMv7-A target (720p I420 @ 30 fps) this dominated per-frame cost at ~63 ms — the camera HAL collapsed to ~6 fps. Hardware-validated A/B against a new `InjectVideoFrame(std::vector<std::uint8_t>&&, …)` overload that moves the buffer directly into the SDK: total per-frame cost drops from ~70 ms to ~5 ms (14× speedup), matching the in-house baseline publisher.

**Design**: the overload is strictly additive. Default impl forwards to the span overload, so any backend that only overrides one path keeps working. Callers with read-only / shared buffers continue to use the span overload; owning callers get the fast path.

The fix benefits every native C++ consumer (CloudXR, native game-engine plugins, embedded camera SDKs), not just embedded — the embedded case is just where the cost crosses from "wasteful" to "unusable". See finding #12 in [issue #134](https://github.com/NVIDIA/xr-ai/issues/134) for the diagnostic methodology, on-device numbers, and the cross-platform impact estimate.

### 2026-05-12 — `gpu` pytest marker + local-only dev script for hardware-bound tests

Some components (Docker-backed vLLM lifecycle, NVENC paths, anything that
binds a real GPU) cannot be exercised on the GitHub `ubuntu-latest`
runners. Rather than skip them at import time or hide them behind ad-hoc
environment flags, we registered a single `gpu` pytest marker in
`tests/pyproject.toml` (and defensively in `tests/conftest.py::pytest_configure`
so branches that haven't picked up the pyproject change yet don't emit
`PytestUnknownMarkWarning`). CI's pytest invocation in
`.github/workflows/tests.yml` now passes `-m "not gpu"`, and developers
run the hardware-bound suite via `tests/run_local_gpu_tests.sh`, which
just calls `uv sync` then `pytest -m gpu` on the local box. Subsequent
batches that add GPU / Docker / NVENC tests should decorate them with
`@pytest.mark.gpu`; no further wiring is required.

### 2026-05-11 — Native StreamKit `LiveKitBackend` implementation

The stub at `client-samples/native/StreamKit/src/Backends/LiveKit/LiveKitBackend.cpp` is replaced with a working implementation against the upstream LiveKit C++ SDK (`livekit::Room`, `LocalParticipant`, `AudioSource`, `VideoSource`). The backend now covers `Connect` / `Disconnect` / `Send` / data-channel `_agent.status` interception / `FrameSink::InjectVideoFrame` lazy publish. CMake gains `LIVEKIT_SDK_ROOT` + `LIVEKIT_LIB_DIR` cache vars; without `LIVEKIT_SDK_ROOT` the backend compiles header-only in stub mode so CI stays green.

**Design**: shared_ptr (not unique_ptr) holds the opaque LiveKit handles so the destructor is well-defined in translation units that don't include the SDK headers (stub mode). The `livekit::VideoSource(width, height)` ctor requires explicit dimensions, so the LiveKit-backed FrameSink can't honour the "track is published on `StartCamera`" interpretation — instead `StartCamera` arms state and the track is created + published on the first injected frame (consistent with FrameSink's documented contract).

**Left out (called out in the README's "Constraints" table)**: platform mic capture (no portable C++ API), platform camera open (same), `FetchToken` HTTP (host overrides), `AudioConfig::MicrophoneMode` AEC/AGC/NS mapping (would need `AudioProcessingModule`). See [issue #134](https://github.com/NVIDIA/xr-ai/issues/134) for the cross-SDK API friction surfaced during the integration.

### 2026-05-10 — TLS by default; canonicalize the same-origin wss:// proxy; drop the client-side `secure` toggle

`web_server_tls` now defaults to **true**. The hub web server terminates
HTTPS on `web_server_port` (8080) and exposes a same-origin
`wss://<host>:8080/rtc[/<version>]` route that proxies LiveKit signaling
to the internal plaintext 7880 (`_lk_proxy.py`); that proxy is now the
**only** client-facing signaling path. LiveKit's native 7880 stays on
`127.0.0.1` — no external client connects to it directly.

The proxy gained protocol-version-aware paths in the same change: the
LiveKit JS SDK v2.x appends `/rtc/v1` to the base URL it is given (see
`client-sdk-js/src/api/utils.ts::createRtcUrl`), so the proxy now matches
`/rtc/{tail:path}` and forwards the version segment verbatim. The
previous `/rtc`-only routes would have closed v2.x sessions through the
catch-all WebSocket handler.

The Android, iOS, and visionOS samples lost their "HTTPS token" / `secure`
toggle. The `secure` field is gone from `BackendConfiguration`
(`.swift`/`.kt`); the URL is unconditionally `wss://<host>:<port>` and the
default token endpoint is `https://<host>:<port>/token`. The web non-XR
client auto-detects from `window.location.protocol`. The default `port`
field on mobile changed from `7880` → `8080` (the hub web-server port,
*not* LiveKit's native port). iOS persists this via `UserDefaults`, so a
device that had connected with the old default will keep `7880` saved
and needs to be edited in the app — Android's Compose `mutableStateOf`
isn't persisted, so it picks up the new default on next launch.

**Why this over native LiveKit TLS.** `docs/architecture.md` previously
listed three workarounds and called native LiveKit TLS "the correct
long-term fix but has not been implemented yet". On audit, the proxy
(workaround #1) was already in place and exercised by `web-xr`; the
remaining bug was that mobile clients built their own `ws://host:7880`
URL instead of using the URL `/token` returns. Native LiveKit TLS would
have required Rust-side cert verification in `livekit-rtc` for the
internal Python connector talking to its own LiveKit instance — an
uncertain surface to depend on. Canonicalizing the proxy is a smaller
change with the same user-visible outcome.

**What stayed.** `web_server_tls: false` is still a valid escape hatch
for `localhost`-only dev; the same-origin proxy then serves plain `ws://`.
The internal Python connector still talks to LiveKit over plain `ws://`
on `127.0.0.1:7880`. WebRTC media on 7881/TCP fallback and 7882/UDP is
DTLS/SRTP regardless — those ports are unchanged.

**iOS / visionOS cert install is mandatory.** Investigating a real client
rejection: the LiveKit Swift SDK's `URLSession` (`WebSocket.swift`
`Delegate`) does not implement
`urlSession(_:didReceive:completionHandler:)` for server-trust
challenges, and ATS does not bypass cert-chain validation regardless of
`NSAllowsArbitraryLoads` — the previous code comment claiming otherwise
was wrong. The `TrustingSessionDelegate` inside `LiveKitBackend.swift`
only covers the `/token` HTTP fetch, and was additionally gated by
`#if DEBUG` (so Release builds rejected the cert there too). The
`#if DEBUG` gate is gone, and the hub now exposes `/cert` (MIME
`application/x-x509-ca-cert`) so iOS Safari can install the hub's
self-signed cert in one tap. The iOS README + `docs/networking.md` +
`docs/troubleshooting.md` document the install + Full Trust toggle.
Android also requires cert install; see the tech-debt cleanup sub-entry below.

**The Full Trust toggle did not appear in initial testing** because
`_tls.py` generated the cert with `BasicConstraints CA:FALSE`; iOS only
exposes the toggle for CA-marked certs. The generator now writes a
self-signed CA (the mkcert pattern: `CA:TRUE`, `KeyUsage` with both
`key_cert_sign` and `digital_signature` + `key_encipherment`, EKU
`SERVER_AUTH`) and `ensure_self_signed_cert()` detects a cached non-CA
cert from older builds and regenerates with a one-line log banner.
Devices that installed the old profile must remove it from VPN &
Device Management before reinstalling the new one.

**The cert's SAN missed the LAN IP** in a second round of iOS testing
(`errSSLBadCert` / NSURLErrorDomain `-1202`, "pretending to be
10.29.90.196"). The previous SAN-population path used
`socket.gethostbyname(socket.gethostname())`, which on Ubuntu returns
the `/etc/hosts` loopback alias `127.0.1.1` rather than the LAN IP.
`_tls.py` now enumerates routable IPv4 addresses via the UDP-connect
trick (open a non-blocking UDP socket to `8.8.8.8` / `1.1.1.1` /
`169.254.169.254` and ask the kernel which interface it would route
from) and `gethostbyname_ex` for /etc/hosts aliases, then includes every
non-loopback hit in the SAN. `ensure_self_signed_cert()` also detects a
SAN that's missing a currently-detected local IP and regenerates with a
log banner directing users to reinstall the profile.

**Third iOS issue: 401 from LiveKit after a successful TLS handshake.**
The LiveKit Swift SDK 2.13.0 puts the JWT in
`Authorization: Bearer <token>` (the JS SDK uses `?access_token=…` in
the query string instead). The same-origin proxy in `_lk_proxy.py`
forwarded the query string but dropped every request header, so
LiveKit-server saw an unauthenticated WSS upgrade from the Swift SDK
and 401'd it. The proxy now forwards end-to-end headers (everything
except a hop-by-hop allowlist) to both the `/rtc/validate` HTTP shim and
the `/rtc[/<version>]` WebSocket, via httpx and `websockets.connect`'s
`additional_headers`. The web client is unaffected — it puts the token
in the query string and worked before.

**Tech-debt cleanup: `TrustAllCerts.kt` removed; Android now requires cert
install like iOS.** `TrustAllCerts.kt` was not "trust our specific self-signed
cert" — it accepted any cert from any issuer, including legitimately-misissued
or attacker-controlled ones, defeating TLS entirely for the Android client.
The file is deleted. `LiveKitOverrides` (which existed solely to inject the
trust-all OkHttpClient) is dropped from `LiveKitBackend.kt` as well. Android
now validates against the system + user CA store identically to iOS. To
replace the manual URL-typing step, both apps gained an in-app **"Install hub
certificate"** button in the Connection section (enabled when Host is non-empty,
visible in the disconnected state). Android fetches the cert from
`https://<host>:<port>/cert` using a single-connection trust-all
`HttpsURLConnection` scoped to that one call only (the cert cannot be
pre-validated because it is what we are installing), then opens the system
`KeyChain` install dialog. iOS opens Safari at the same URL, which triggers
the existing profile-install flow. The `network_security_config.xml`
`<certificates src="user"/>` anchor — already present — is the mechanism that
makes Android accept the cert once it is OS-installed; no domain-config
exception was needed or added.

### 2026-05-05 — Docker backend option for vLLM-backed servers

All four vLLM-backed services (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`, `nemotron_omni_llm_server`) gained an opt-in
`vllm_backend: docker` mode that hosts vLLM via the NVIDIA NGC container
(`nvcr.io/nvidia/vllm:26.04-py3` by default) instead of the pip-installed
`vllm` CLI. Goal: let users try NVIDIA's optimized vLLM build without
disturbing the pip path that everything ships on.

(Cleanup side-effect: `llama_nemotron` was already a vLLM launcher despite
being described as "transformers in-process (+ LMFE)" in the docs — the
description was stale from a prior implementation. `DEPENDENCIES.md`,
`docs/ai-services.md`, and the service's `README.md` were corrected in the
same change.)

**Design.** Backend selection is per-server via a new YAML key:

```yaml
vllm_backend: pip       # default — today's behavior
# or
vllm_backend: docker
vllm_image:   nvcr.io/nvidia/vllm:26.04-py3
```

The `Process(...)` declaration in each orchestrator stays identical; flipping
backends is one YAML edit. Both modes honor the same `model:`, `port:`, and
vLLM-flag config — only the runtime that interprets them differs.

A new stdlib-only utility package `utils/xr-ai-vllm/` exposes `serve(...)`
(dispatches pip vs docker) and `stop_persistent_servers(...)` (docker-aware
cleanup that tries `docker stop <container_name>` first, falls back to the
existing port → PID → SIGTERM/SIGKILL path for pip mode). Each of the four
wrapper `__main__.py` files now reads YAML, builds its model-specific
`extra_serve_args`, and calls `serve()` — the per-service logic
(`_gpu_compute_major()` quant selection, nano_v3 parser fetch,
`media_io_kwargs` for omni, `llama3_json` tool-call parser for
llama_nemotron) stays in the wrapper because it is not runtime-agnostic.

**Why a shared helper instead of triplicated logic.** Four concrete consumers
justify the abstraction per the AGENTS.md rule. Keeping it stdlib-only matters
because docker mode's whole point is to avoid pulling vllm/torch into the
wrapper's venv when it is not actually used; the helper sits beside
`xr-ai-launcher` (also stdlib-only) at `utils/xr-ai-vllm/`.

**Why per-YAML toggle, not separate sibling packages.** The user-visible
change is "the same vLLM, hosted differently". A YAML key matches that
exactly; sibling packages would have meant duplicating every service's
pyproject, README, and per-profile YAML for a runtime-only swap. The plain
boolean `Process(...)` line stays unchanged.

**Why detached docker containers (not foreground).** `vlm_server`,
`llama_nemotron_llm_server`, and `nemotron3_nano_llm_server` already implement
vLLM persistence in pip mode via `start_new_session=True` so the container
survives wrapper restarts — loading 16–30 GB of weights every dev iteration
would be a regression. Docker mode mirrors this with
`docker run -d --rm --name xr-ai-vllm-<service>`; cleanup is via
`xr_ai_vllm.stop_persistent_servers()` invoked from `xr_render_demo --stop`.
`nemotron_omni_llm_server` was non-persistent in pip mode (used `os.execvp`)
and is non-persistent in docker mode (foreground `docker run --rm`) — same
shutdown semantics for both runtimes.

**Why `--network host` + `--ipc host`.** Mirrors the LiveKit container in
`server-runtime/xr_media_hub/transport/livekit/_docker.py` (the only other
docker-managed subprocess in the repo) and gives vLLM workers the shared
memory region they need for KV cache shards. Linux-only, which the repo
already requires.

**NGC auth.** The wrapper auto-runs `docker login nvcr.io --password-stdin`
when `NGC_API_KEY` is in the environment (loaded by `load_credentials()` per
`docs/credentials.md`) and no existing auth is found in `~/.docker/config.json`.
Existing logins are not overwritten.

**Side-effect bug fix.** `nemotron_omni_llm_server` previously used
`os.execvp` which never touched the launcher's `--ready-file`. Routing it
through `xr_ai_vllm.serve(persistent=False, ...)` adds the standard
`/health` poll + ready-file touch path, so the launcher's `_wait_ready` no
longer hangs when this service is selected.

**As-was.** Defaults, ports, model IDs, `Process(...)` lines, persistence
semantics, and the `--stop` UX are all unchanged for users who do not flip
`vllm_backend:`. Existing pip-mode `model_cache` weights are reused by docker
mode (mounted at the same path inside the container) and vice versa.

### 2026-05-05 — Unified loguru stack; `launcher/` and `xr-ai-logging/` consolidated under `utils/`

Two related infrastructure changes shipped together.

**Loguru migration.** New `xr-ai-logging` package wraps loguru with a single
`setup_logging(name, namespace=...)` entry point that every process calls
once at startup. Installs a stderr sink (INFO by default, DEBUG when
`XR_AI_VERBOSE` is truthy), an always-DEBUG file sink at
`/tmp/log_<namespace>_<YYYY-MM-DD_HH-MM-SS>/<process>.log`, and a stdlib
`logging` -> loguru bridge so `utils/xr-ai-launcher/` (stdlib-only by
contract) and `agent-sdk/xr_ai_agent/` (pyzmq+msgpack-only by contract)
participate without importing loguru. Subprocess coordination uses three
stamped env vars (`XR_AI_LOG_NAMESPACE` / `XR_AI_LOG_TIMESTAMP` /
`XR_AI_LOG_ROOT`). Stderr-vs-file split lets the user keep a quiet console
while retaining full DEBUG history per run.

The launcher's child-stdout/stderr forwarder also moved from raw `print()`
to a level-aware `log.<level>(...)` (parses the loguru level from each
captured line and re-emits at that level), so library banners (NeMo,
OpenXR loader, LOVR Vulkan) stay out of the default console but are
preserved in the file sink. ~16 INFO calls were demoted to DEBUG (per-data-
message, per-NVENC-chunk, per-tool-call duplicates, per-VAD-false-positive,
etc.) so INFO is now strictly lifecycle / once-per-utterance / periodic
stats.

**`utils/` consolidation.** Both `launcher/` and `agent-sdk/xr-ai-logging/`
are pure infrastructure used by every process, not specific to agents.
Moved to `utils/xr-ai-launcher/` and `utils/xr-ai-logging/` so the layout
reflects actual scope. The `xr-ai-launcher` "stdlib-only" rule still
applies — `utils/xr-ai-launcher/pyproject.toml` keeps `dependencies = []`.
`utils/xr-ai-logging/` has its own pyproject (`loguru>=0.7`). Python import
paths (`xr_ai_launcher`, `xr_ai_logging`) are unchanged; only filesystem
paths in `[tool.uv.sources]` and doc references shifted.

### 2026-05-05 — vLLM model persistence across stack restarts

vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`) now survive stack shutdowns so model weights stay
loaded across worker crashes and debug restarts.

**Mechanism:** each wrapper checks its own `/health` endpoint before spawning
vLLM.  If already healthy the wrapper signals ready immediately and idles (exits
cleanly on SIGTERM without touching vLLM).  If not healthy it spawns vLLM
normally.  vLLM itself is started with `start_new_session=True` so the
launcher's `killpg()` does not reach it.

**Cleanup:** `uv run xr_render_demo --stop` from the sample directory hits each
server's `/health`, finds the PID via `ss`/`lsof`, and sends SIGTERM (escalates
to SIGKILL after 20 s).

**Why this approach over launcher-level `persistent=` flag:** keeps `main.py`
and `_stack.py` unchanged; persistence is a detail of each service's own
startup script, not the orchestrator.

### 2026-05-01 — visionOS Enterprise license bundling

Apple Vision Pro main-camera passthrough
(`com.apple.developer.arkit.main-camera-access.allow`) requires the entitlement
signed into the binary **and** a per-team `Enterprise.license` file bundled
into the `.app`. Without the license the API is a silent no-op
(`CameraVideoFormat.supportedVideoFormats(...)` returns `[]`, LiveKit AR camera
publish fails with `LiveKitError.invalidState`). visionOS auto-loads the file
from the app bundle.

The license file is per-team and Apple's terms restrict redistribution, so it
is gitignored (`**/Enterprise.license`) rather than committed. A placeholder
`App/Enterprise.license.sample` documents the path for new contributors. An
Xcode "Copy Enterprise.license" build phase copies the file into the `.app`
at build time; if missing, the build succeeds with a warning and the camera
path no-ops at runtime (audio + data + simulator GIF feed are unaffected).
Symlinks at the expected path are supported and still gitignored.

The sample's display name (`StreamKitSample`) is intentionally decoupled from
its Bundle ID (`com.nvidia.xr-ai-example`) so a fork that renames the Bundle
ID still ships under the same on-device app name.

### 2026-04-30 — Unified MCP IDs: identity sidecars, live vs recorded splits, transcript source_id

The MCP servers had two consistency gaps: (1) `list_*` tools returned
sanitized filesystem names rather than the original LiveKit identities,
so a caller round-tripping a value through `list_recording_participants`
→ `get_latest_frame` could miss; (2) the transcript store named its
key `participant_id` even though transcripts can come from non-
participant sources (e.g. an agent's own TTS).

Changes:

- **Recorder + stores write a `.identity` sidecar per source.** The
  hub's `_recorder.py` writes `<recordings_dir>/<safe>/.identity`;
  transcript-mcp writes `<transcripts_dir>/<safe>.identity` next to
  the JSONL. Sidecar contents are the raw caller-supplied ID verbatim.
  Collisions between distinct raw IDs that happen to share a
  `_safe_name` get a counter suffix (`alice_home`, `alice_home_2`, …).
- **List tools return raw IDs.** `list_recorded_participants` (renamed
  from `list_recording_participants`) and `list_sources` (renamed from
  `list_participants` on transcript-mcp) read sidecars and return
  exactly what the writer passed in.
- **New tool `list_live_participants`** on video-mcp — surfaces
  `ep.connected_participants` from the ProcessorEndpoint so callers
  can ask "who's actually live right now?". This is the only set
  `get_latest_frame` will succeed for.
- **Transcript-mcp renames `participant_id` → `source_id`** in tool
  signatures, response keys, and stored identity sidecars. The store
  treats `source_id` as opaque, allowing agents to write under
  internal names (`"agent-vlm"`, `"tts"`) alongside live participant
  records. video-mcp keeps `participant_id` since video really does
  come from real participants.
- mcp-agent worker updated to use `source_id` when calling transcript
  tools.

**Why:** the underlying storage was always string-keyed and didn't
care, but the API leaked sanitized filenames and overloaded
"participant" semantics onto things that aren't participants. The
sidecar lifts the raw name back out cleanly; the rename names the
field for what it actually is.

### 2026-04-29 — Video recording on tmpfs; video-mcp gains live-frame + frame-at-time

`server-runtime/xr_media_hub/video/_recorder.py`:
- Default `out_dir` flipped from `/tmp/xr_recordings` (disk) to
  `/dev/shm/xr-ai/recordings` (tmpfs — RAM-backed). Writes don't touch
  disk by default.
- Eviction policy is now **size-based, global**: `max_total_bytes`
  (default 500 MB) caps total chunk size across all participants.
  When the cap is exceeded, oldest chunks are evicted FIFO. Replaces
  the prior per-participant `max_chunks` count.

`agent-mcp-servers/video-mcp/`:
- Now connects to the hub as a `ProcessorEndpoint` with
  `filter=Subscribe.VIDEO`. A small `FrameProvider` tracks the most
  recent `FrameSignal` per pid; pixel bytes are pulled on demand via
  `request_frame()`. No on-disk side-channel — the live path is
  entirely IPC-based.
- New MCP tool `get_latest_frame(participant_id)` — calls into the
  provider, converts the returned `FrameData` to RGB, writes a PNG to
  `out_dir`, returns `{path, width, height, timestamp_us, track_id}`.
- New MCP tool `get_frame_at_time(participant_id, timestamp_us)` —
  finds the chunk covering the timestamp, decodes it with NVDEC via
  PyNvVideoCodec, picks the frame closest to the timestamp by linear
  interpolation across the chunk, encodes PNG, returns `{path, width,
  height, timestamp_us, chunk_path}`.
- video-mcp gains `xr-ai-agent`, `PyNvVideoCodec`, `Pillow`, and
  `numpy` runtime deps. mcp-agent's composed `mcp_server` adopts the
  same model — owns its own `ProcessorEndpoint` and lifecycle.

**Why:** disk IO was wasted overhead for chunks that almost always get
evicted within minutes; `/dev/shm` cuts the IO cost to RAM bandwidth
without changing the file-based interface that the video-mcp uses for
historical queries. Live frames bypass the chunk store entirely — the
hub already has the most recent SHM slot held open per (pid, track),
so a `request_frame()` is a single zero-copy memcpy at the hub plus a
pixel-format conversion at the consumer.

### 2026-04-29 — MCP servers go pure FastMCP

`transcript-mcp-server`, `video-mcp-server`, and the composed `mcp-server`
in `agent-samples/mcp-agent/` no longer wrap a FastAPI app. Each runs the
`FastMCP.http_app(path="/mcp")` Starlette app directly under uvicorn.

- All worker ingress is now an MCP tool call. Transcript ingest is the new
  `transcript_add_transcript` tool (replaces `POST /ingest`); stats fetches
  use the existing `transcript_get_transcript_stats` / `video_get_video_stats`
  tools (replace the `/transcript/stats/{pid}` and `/video/stats/{pid}` REST
  routes).
- The composed `mcp-server` no longer reads a `skills:` config block — both
  sub-servers are always mounted; their per-server config lives at the top
  level of `mcp_server.yaml` under `transcript:` and `video:`.
- `/health` is gone. The mcp-agent worker's readiness probe now uses
  `fastmcp.Client.list_tools()` against `/mcp` to confirm the server is
  serving (a stronger guarantee than a 200 from `/health` ever was).
- Drops the `fastapi` and `pydantic` runtime dependencies on transcript-mcp,
  video-mcp, and the composed mcp-server. Worker gains a `fastmcp>=0.4`
  dependency.

**Why:** the dual REST + MCP surface had no value once workers got an MCP
client. Two interfaces meant two contracts to keep in sync, two error-
handling paths, and two readiness checks. Pure FastMCP is one contract,
one error model, and a stronger readiness check via `list_tools()`.

### 2026-04-29 — Participant-keyed agent subscriptions

`ProcessorEndpoint` now models subscriptions as **participants**, not topic
prefixes. The unit of opt-in is "I want everything for participant X";
categories (data / audio / video) are an opt-out filter inside that.

- New `Subscribe` flag enum (`DATA`, `AUDIO`, `VIDEO`, `ALL`) replaces the
  prior raw-bytes `topics=` parameter.
- `ProcessorEndpoint(auto_subscribe=True, filter=Subscribe.ALL)` is the
  default. The endpoint installs an internal participant handler that
  calls `subscribe(pid)` on join and `unsubscribe(pid)` on leave —
  agents see every client's full inbound stream out of the box.
- `ep.subscribe(pid, filter=...)` and `ep.unsubscribe(pid)` are the
  primitives. Idempotent. Calling subscribe with a different filter
  diffs the active subscriptions. Subscribing before the pid joins is
  fine — ZMQ holds the SUBSCRIBE.
- `auto_subscribe=False` is the escape hatch for single-client agents:
  the agent only sees `participant` + `control` until it explicitly
  subscribes. Use `ep.subscribed_participants` to introspect live state.
- New `ROSTER_REQUEST` IPC type (`MsgType.ROSTER_REQUEST = 12`).
  `request_roster()` (called automatically once at the start of
  `run()` when auto-subscribe is on) makes the hub re-publish
  `PARTICIPANT_EVENT(joined=True)` for every current pid so endpoints
  started mid-session catch up. Replays go on the regular `participant`
  topic, so other endpoints' `on_participant` callbacks may fire again
  for known pids — keep them idempotent.
- Topic prefixes always include the trailing `.` so `data.alice.` does
  not bleed into `data.alice2.chat`. The helper centralises this so
  individual agents never type the prefix themselves.

**Why:** the previous bytes-tuple `topics=` parameter forced agents to
know hub-internal topic conventions, made per-pid scoping awkward
(write your own join handler, remember the trailing dot), and didn't
solve the mid-session catch-up problem. The new model treats the
participant as the unit of subscription, which is what every real
agent actually wants — most just take the default broadcast; the few
that need scoping flip `auto_subscribe=False` and call `subscribe`.

### 2026-04-29 — Multi-client / multi-agent isolation; topic surface; tests

The hub formally supports many clients and many agents at once. The IPC and
LiveKit transport layers were extended so:

- **Per-participant return audio** — `RoomClient` now publishes one
  `xr-hub-return-{pid}` audio track per active participant, with subscribe
  permissions restricted via `set_track_subscription_permissions` so a
  participant can only hear their own return audio. Tracks are unpublished
  on participant leave.
- **Targeted return data** — `RoomClient.send_return_data` passes
  `destination_identities=[participant_id]` so return text/binary is no
  longer broadcast to other participants in the room.
- **`ReturnAudioFlush` control message** (`MsgType.RETURN_AUDIO_FLUSH = 11`)
  added to `xr-ai-agent`. `ProcessorEndpoint.flush_return_audio(pid)`
  routes through the hub on `return_audio_flush.<pid>` to the connector,
  which calls `AudioSource.clear_queue()` for that pid only. Used to
  cleanly interrupt agent TTS playback when a new query arrives.
- **StreamKit `onDataReceived(topic, data)`** — the previously dropped
  data-channel `topic` is now surfaced to the application across web and
  iOS/visionOS. The reserved `_agent.status` topic is still intercepted
  internally and never reaches `onDataReceived`.
- **`tests/` top-level suite** — multi-client / multi-agent coverage over
  the real IPC layer (no Docker / LiveKit needed). CI workflow at
  `.github/workflows/tests.yml` runs the suite on every push and PR
  across Python 3.11 and 3.12.

### 2026-04-30 — LLM servers reorganized into per-model packages

`ai-services/llm-server/` (single package, single model) is split into two
sibling packages under `ai-services/llm/`, each with its own entry-point
command, YAML, default port, and dependency set. This lets a sample pick the
LLM that matches its tool-calling / reasoning / hardware requirements without
dragging in the dependencies of the others (notably vLLM and
`lm-format-enforcer`).

| New package | Command | Port | Model | Backend |
|---|---|---|---|---|
| `llm/llama_nemotron/` | `llama_nemotron_llm_server` | 8106 | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | HF transformers + `lm-format-enforcer` — native tool calls, reasoning toggle |
| `llm/nemotron3_nano/` | `nemotron3_nano_llm_server` | 8107 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | vLLM (execvp shim) — Blackwell FP4 MoE |

- **HTTP contract is identical** across both (OpenAI-compatible
  `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`). Workers point
  at a different port to swap backends — no worker-side code changes.
- **Ports** chosen to be non-overlapping so both LLM backends can coexist in
  the same stack if a sample actually wants that (unusual; typically pick one).
- **`llama_nemotron`** adds grammar-constrained tool-call decoding via
  `lm-format-enforcer`. When `tools=[...]` is present in the request, a
  `UnionParser([tool_call_grammar, free_text])` is fed as
  `prefix_allowed_tokens_fn` so the model's vocabulary is masked every step to
  either valid `<TOOLCALL>[{...}]</TOOLCALL>` JSON or plain assistant text.
  Rationale: the native Llama-3.1 chat template instructs the model to emit
  tool calls as JSON, but sampling noise / schema drift can still produce
  syntactically broken output. LMFE eliminates that entirely.
- **`nemotron3_nano`** is intentionally thin (~200 lines). vLLM already
  exposes the OpenAI API, parses Nemotron-3-Nano's XML tool-call format via
  `--tool-call-parser qwen3_coder`, and splits the `<think>…</think>` preamble
  via `--reasoning-parser nano_v3` (custom plugin auto-fetched from the model
  card into `model_cache`). The shim reads the YAML, sets
  `VLLM_USE_FLASHINFER_MOE_FP4=1`, and `os.execvp`s into `vllm serve` so the
  launcher's signals go straight to vLLM with no intermediate wiring.
- **`enforce_eager: true`** is the default for `nemotron3_nano` — CUDA graph
  capture plus FlashInfer FP4 MoE autotune are silent and take 3–8 min on
  first run, which is a bad UX for a voice agent waiting to become healthy.
  Eager mode starts in ~5 s after weight load and is 10–20% slower per token
  (imperceptible at <250 tokens/turn where STT+VAD+TTS already dominate).

Dependency fan-out stays contained: only `llama_nemotron` pulls
`lm-format-enforcer`, only `nemotron3_nano` pulls `vllm>=0.12.0`.

### 2026-04-29 — render-mcp + oxr-mcp added; xr-render-demo as integration

Two new MCP servers under `agent-mcp-servers/`, port-per-server, no LiveKit
dep. oxr-mcp is pure FastMCP; render-mcp mixes one streaming HTTP route
with FastMCP tools.

**render-mcp** (`agent-mcp-servers/render-mcp/`, port 8220) — owns the LOVR
child (the OpenXR rendering app) and is the only process that pushes ops
onto LOVR's `scene_socket` (msgpack over ZMQ PUSH).

- **`POST /sphere/radius` is a plain FastAPI route**, not an MCP tool.
  The worker hits it ~50 Hz from the audio path; routing a streaming
  control signal through FastMCP's per-request dispatch + JSON-RPC
  envelope is the wrong shape and makes the server log unreadably chatty.
  The discrete operations (`start_xr`, `set_sphere_color`, …) stay on
  `/mcp` where an LLM agent can discover and drive them.

- **`xr.session.started` gates LOVR spawn.** CloudXR returns
  `XR_ERROR_FORM_FACTOR_UNAVAILABLE` from `xrGetSystem` until a streaming
  client has actually connected. Spawning LOVR at process start lands it
  in the desktop simulator forever. The caller is expected to call
  `start_xr` only after seeing the streaming client come up.
- **`start_xr` returns immediately; caller polls `get_health.lovr_started`.**
  The cloudxr readiness wait can take a minute; matching a single tool
  call's timeout to it would couple two unrelated knobs. render-mcp spawns
  LOVR + waits for cloudxr in a background task, caches terminal failures
  so retries fail fast, and exposes progress through `get_health`.

**oxr-mcp** (`agent-mcp-servers/oxr-mcp/`, port 8230) — exposes head pose
through a `get_head_pose()` MCP tool.

- **Two OpenXR sessions, one CloudXR.** LOVR holds the rendering session;
  oxr-mcp opens a SECOND headless session (`XR_MND_HEADLESS`) for pose
  only. Verified empirically: pos/quat update from the headset while LOVR
  keeps streaming pixels, no contention. Session opens lazily on first
  `/pose` request, so it doesn't fight CloudXR's startup either.

**Shared infra** — `launcher/_cloudxr_env.py`. Both MCPs need to wait for
`cloudxr.env`, source it, and wait for `runtime_started` before opening
their OpenXR sessions; the launcher (which already manages the cloudxr
child) is the natural home.

**xr-render-demo** (`agent-samples/xr-render-demo/`) — integration sample.
Web client streams mic audio; the worker computes RMS → sphere radius
continuously and runs VAD → STT → LLM whose JSON action list it translates
into render-mcp HTTP calls.

- **User-frame coordinates with worker-side transform.** The LLM emits
  user-frame coordinates (`+x` user's right, `-z` in front of the user).
  The worker fetches head pose from oxr-mcp once per utterance, rotates by
  yaw + translates by head position before forwarding to render-mcp.
  Putting the transform in the worker keeps render-mcp transport-agnostic
  and means the LLM never has to learn vector math.

### 2026-04-27 — MCP example: transcript + video MCP servers; NVENC recording in hub

`agent-samples/mcp-agent/` added as a demonstration of MCP integration with XR data.

**Transcript MCP server** (`agent-mcp-servers/transcript-mcp/`, port 8200):
- Single FastAPI process hosts both the non-MCP HTTP ingest endpoint (`POST /ingest`)
  and the FastMCP tools (`/mcp`) so agents can query historical transcripts.
- Agent workers POST transcripts over plain HTTP; MCP is for LLM tool-use only.
- JSONL storage persists across server restarts; one file per participant.

**Video MCP server** (`agent-mcp-servers/video-mcp/`, port 8210):
- Thin FastMCP wrapper around the hub video HTTP API (`GET /video`).
  Fetches the concatenated H.264 byte stream, writes it to a temp file, returns path.
- Kept separate from the transcript server so either can be used independently.

**Hub NVENC video recording** (`server-runtime/xr_media_hub/video/_recorder.py`):
- Opt-in via `video_recording.enabled: true` in `xr_media_hub.yaml`.
- Uses `PyNvVideoCodec` (on PyPI) for NVENC encoding; included in the standard `uv sync`.
  The config guard (`enabled: true`) prevents instantiation when recording is not needed.
- VBR mode, no B-frames (`bf=0`), `repeat_sps_pps=1`.  Each chunk uses a fresh encoder
  session so it always begins with SPS+PPS+IDR and is independently decodable.
  Chunks are binary-concatenable with `cat`.
- Hub exposes a video query HTTP API on port 8090 (`GET /video?pid=&start_us=&end_us=`).

**PyNvVideoCodec pitfalls (hard-won)**:
- `Encode()` must receive a **2D numpy array** of shape `(H*3//2, W)` — do **not** call
  `.flatten()`.  NVENC reads the array using numpy strides to determine the row pitch.
  A 1D array causes NVENC to assume an internally aligned pitch (e.g. 512 for W=320),
  producing a circular horizontal shift in every decoded frame.
- `GetSequenceParams()` does not exist in PyNvVideoCodec 2.x.  Use `repeat_sps_pps=1`
  in `CreateEncoder` kwargs instead; it prepends SPS+PPS automatically before each IDR.
- WebRTC adaptive bitrate changes the frame resolution mid-stream.  The encoder must be
  recreated (and the current chunk flushed) whenever `width` or `height` changes.  Feeding
  wrong-sized frames to the encoder silently corrupts all subsequent output.
- There is no reliable option that forces repeated IDR frames in PyNvVideoCodec 2.x.
  `gopLength`, `gop`, `idrPeriod` were all tested — NVENC only emits one IDR at the start
  of a session regardless.  Use per-chunk fresh encoders (`EndEncode` → `CreateEncoder`)
  to guarantee IDR boundaries; each new encoder session always begins its output with IDR.

**mcp-agent worker** (`agent-samples/mcp-agent/worker/`):
- Runs continuous STT (same VAD logic as echo-agent).
- POSTs each final utterance to the transcript-mcp-server over HTTP.
- Does not speak TTS — pure observation/logging pipeline.

### 2026-04-24 — AI inference servers added; NVIDIA models; shared model cache

`ai-services/` added as a sibling of `server-runtime/`, containing three reusable
OpenAI-compatible HTTP inference servers.

Model choices — all NVIDIA:
- **vlm-server**: `nvidia/Cosmos-Reason1-7B` in-process via HuggingFace
  transformers (Qwen2.5-VL architecture).  Accepts base64 image_url in messages.
- **stt-server**: `nvidia/parakeet-tdt-0.6b-v3` in-process via NeMo ASR.
  English-only TDT model, CC-BY-4.0.  ~1.5 GB VRAM.
- **tts/magpie**: `nvidia/magpie_tts_multilingual_357m` in-process via NeMo TTS.
  Multilingual, NVIDIA Open Model License.  ~1 GB VRAM.
- **tts/piper**: any rhasspy/piper-voices ONNX voice; ~100 ms/sentence on CPU.

Shared model cache: all weights land in `models/` at the repo root (gitignored).
Each YAML configures `model_cache` (resolved relative to the YAML file) so the
same physical directory is used regardless of which sample root the YAML is in.

Sample YAMLs for all four services ship with `mcp-agent` as a template.

OpenAI-compatible APIs chosen so workers never need to know backend details —
swap models by changing the YAML only.

### 2026-04-22 — CloudXR runtime extracted to top-level shared component

`cloudxr-runtime/` added as a peer of `server-runtime/`, wrapping
`isaacteleop[cloudxr]` (NVIDIA IsaacTeleop SDK).  Samples opt-in by adding
`Process("cloudxr", "../../cloudxr-runtime", "cloudxr_runtime")` to their
`PROCESSES` list and providing a `cloudxr_runtime.yaml` in the sample root.

The native CloudXR service runs entirely as a local process (no Docker).
`isaacteleop`'s Python `wss_run()` provides a TLS WebSocket proxy on port 48322
required for `auto-webrtc` profile; `auto-native` does not need it.
CloudXR and the hub are fully independent: CloudXR streams rendered/sim content
to XR devices over WebRTC while the hub handles agent media via LiveKit.

### 2026-04-22 — Launchable convention + StackLauncher

Each runnable sub-project (hub, worker, future CloudXR runtime, MCP servers) is a
**launchable**: an entry-point command + an optional `<command>.yaml` config.
The launcher discovers YAML files automatically by convention — no separate
launcher config file (the previous `stack.toml` idea was dropped).

The orchestrator code declares the process sequence using `Process` + `run_stack`.
All processes start concurrently; startup order does not matter because every
launchable must be resilient to peers not being ready (ZMQ reconnects, etc.).
`run_stack` is fail-fast: any process exit terminates the whole stack.

`launcher/` gained `Process`, `StackLauncher`, and `run_stack` (all stdlib-only).
`HubLauncher` / `ProjectLauncher` remain as lower-level building blocks.

### 2026-04-21 — Agent-SDK extracted; samples use orchestrator + worker subprocess model

`agent-sdk/` (`xr-ai-agent`) was extracted as a standalone package with only
`pyzmq` + `msgpack` as runtime dependencies. The four IPC client modules
(`_types`, `_codec`, `_shm`, `_processor`) moved there from `server-runtime`.
`server-runtime/xr_media_hub/ipc/__init__` re-exports everything for backwards compat.

Each sample now has two entry points:
- **Orchestrator** (`<name>`): stdlib + `xr-ai-launcher` only. Uses `HubLauncher`
  (which runs the hub via `uv run --project server-runtime`) and `ProjectLauncher`
  (which runs the worker via `uv run --project .`). Waits for the worker to exit.
- **Worker** (`<name>_worker`): imports only from `xr_ai_agent`. Contains all
  agent logic. Launched as a subprocess by the orchestrator.

`launcher/` gained `ProjectLauncher` — a generic context manager that runs any
uv project command as a managed subprocess in its own isolated venv, yielding
the `asyncio.subprocess.Process` for lifecycle control.

**Why:** complete venv isolation between hub (server-runtime), agent (sample), and
orchestrator (launcher-only). No cross-contamination of server deps into agent
venvs and vice versa. `uv run --project` is the mechanism — uv resolves and caches
each project's venv independently.

### 2026-04-21 — VLM agent sample added

`agent-samples/vlm-agent/` — answers natural-language queries about live XR
video using a locally-hosted vision-language model.
**Model:** `nvidia/Cosmos-Reason1-7B` (NVIDIA Open Model License + Apache 2.0,
commercial use permitted; ~16 GB VRAM at BF16). Architecture:
`Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` + `qwen-vl-utils`.
**Protocol:** client sends `vlm.query` data message (raw text or
`{"query":"…","track_id":"…"}`); agent replies on `vlm.response`.
**Frame flow:** `on_frame()` tracks latest `FrameSignal` per (participant,
track); on query, `request_frame(signal)` pulls a pixel copy, converts to PIL
via numpy (I420/NV12/RGB24/RGBA/BGRA), then calls `_VlmBackend.infer()` in a
thread pool so the asyncio loop is not blocked. Model is loaded lazily on the
first query. Override model via `VLM_MODEL` env var.

### 2026-04-21 — Process management moved to `launcher/`

`HubLauncher` lives in `launcher/xr_ai_launcher/`, not in `server-runtime`.
**Why:** process management should not be part of the processes it manages.
The launcher will eventually start MCP servers, CloudXR runtime, and other
components — keeping it separate keeps dependency chains lean and the boundary
clean. `launcher/` has zero runtime dependencies (stdlib only).

### 2026-04-21 — NVDEC/NVENC required; OpenH264 must not be used

`LiveKitConnector.start()` calls `require_nvidia_video_codecs()` before doing
anything else. It checks for `libnvcuvid.so` (NVDEC) and `libnvidia-encode.so`
(NVENC) via ctypes and raises `RuntimeError` if either is absent (Linux only).
**Why:** `livekit-rtc` bundles `libwebrtc` which includes OpenH264 as a software
fallback. OpenH264 is royalty-bearing for end users and must not ship in this
product. The guard prevents silent fallback at the cost of a hard startup failure.
In Docker: `--gpus all` or `--device /dev/nvidia*` must be passed.

### 2026-04-21 — Video frame delivery: metadata push, pixel pull

Processors receive `FrameSignal` metadata at full frame rate via `on_frame()`.
Pixel data is only copied when the processor calls `await ep.request_frame(signal)`.
The hub holds one SHM slot per (participant, track) — always the latest frame.
The slot stays `_STATE_READY` (not released to the connector) until the next frame
arrives for the same track, so `bytes(view.data)` in FRAME_REQUEST is safe.
**Why:** avoids copying every frame over IPC; agents sample at their own rate.
Concurrent `request_frame()` calls for the same track are coalesced into one
FRAME_REQUEST; all waiters receive the same FRAME_DATA response.

### 2026-04-21 — `AgentEndpoint` + `ConsumerEndpoint` → `ProcessorEndpoint`

`ipc/_agent.py` and `ipc/_consumer.py` are deleted. Both are replaced by a
single `ProcessorEndpoint` in `ipc/_processor.py`.
**Why:** `ConsumerEndpoint` was unused scaffolding; `AgentEndpoint` was too
narrow a name (the endpoint suits analytics, recording, etc. — not just agents).
`ProcessorEndpoint` auto-maintains `connected_participants: frozenset[str]` so
processors always know who is present without manual event tracking.

### 2026-04-21 — Agent return path through hub

Agents push `RETURN_DATA`/`RETURN_AUDIO` on the hub's PULL socket.
The hub's `_dispatch` routes them to `send_return_data`/`send_return_audio`,
which PUBs them on `return_data.<pid>` / `return_audio.<pid>` topics.
The `ConnectorEndpoint` SUBs these topics and calls registered callbacks
→ `RoomClient` → LiveKit → client.
**Why:** closes the loop so agents can send audio and data back to participants.

### 2026-04-21 — Echo-agent sample added

`agent-samples/echo-agent/` — echoes audio back to the originating participant
and sends a JSON stats ping (`topic="agent.stats"`) every 5 s to each
connected participant. Demonstrates `ProcessorEndpoint` usage end-to-end.

### 2026-04-20 — Track task management keyed by track SID

`RoomClient._track_tasks` changed from `list[Task]` to `dict[str, Task]`
keyed by track SID. A `track_unsubscribed` handler cancels the exact task.
**Why:** without this, stop/start camera caused a new streaming task to start
while the old one kept running, doubling (then tripling) fps counts.

### 2026-04-20 — Audio format: float32 on the wire, int16 in LiveKit

LiveKit delivers audio as int16 PCM. The hub's IPC layer (`AudioChunk`) uses
float32 LE interleaved. Conversion happens in `_room_client.py`:
- Inbound: `int16 / 32768.0 → float32`
- Outbound (return audio): `clip(float32, -1, 1) * 32767 → int16`

### 2026-04-20 — `xr_media_hub.yaml` config file

Flat YAML at repo root. Fields map 1:1 to `LiveKitConnectorConfig` dataclass.
Relative paths (e.g. `web_client_dir`) resolve relative to the YAML file's
own directory, not CWD. `HubLauncher` searches upward from CWD to find it
automatically.
