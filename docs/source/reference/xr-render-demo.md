<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo — architecture

This architecture reference describes the xr-render-demo sample. Start with
{doc}`/getting_started/quickstart` to run it. For inference-server mechanics
shared with other samples, refer to {doc}`/components/ai-services`.

## Process stack

The orchestrator (`xr_render_demo`, stdlib-only via `xr-ai-launcher`) reuses all
model processes and starts its application processes serially; each owned
process touches its ready file before the next starts. `run_stack` is fail-fast:
any owned process exit terminates the application stack.

| Role | Ownership | Directory | Command | Port |
|---|---|---|---|---|
| hub | sample | `services/device-io-hub/` | `device_io_hub` | 8080 (HTTPS and `/rtc` WSS proxy); LiveKit 7880 stays on 127.0.0.1 |
| cloudxr | sample | `services/cloudxr-runtime/` | `cloudxr_runtime` | 48322 (WSS proxy) |
| stt | reused | `services/stt-server/` | `stt_server` | 8103 |
| tts | reused | `services/piper-tts/` | `piper_tts_server` | 8105 |
| omni | reused | `services/nemotron-omni-llm/` | `nemotron_omni_llm_server` | 8108 (LLM) |
| vlm | reused | `services/vlm-server/` | `vlm_server` | 8100 (Cosmos VLM) |
| video-memory | sample | `services/video-memory-service/` | `video_memory_service` | 8310 (recorded-video typed RPC) |
| scene | sample | `agent-samples/xr-render-demo/scene/` | `xr_render_scene` | 8320 (typed RPC) |
| openxr-service | sample | `services/openxr-service/` | `openxr_service` | 8330 (typed RPC) |
| worker | sample | `agent-samples/xr-render-demo/worker/` | `xr_render_demo_worker` | — |

Before starting the stack, the orchestrator runs two setup steps:

- **Web vendor bundle** — builds the CloudXR + LiveKit ESM bundle via
  `client-samples/web-xr-build/build.sh` (skipped if already present;
  requires `npm`). Built only for WebRTC device profiles; native profiles
  never serve the web page, so the build (and its npm dependency) is skipped.
- **LOVR binary** — auto-downloads LOVR v0.18.0 AppImage to `deps/lovr/` if
  not present and sets `$LOVR_BIN`. Resolution order: `$LOVR_BIN` env var →
  `lovr_bin:` in `scene/scene_service.yaml` → cached AppImage → fresh download.

## Selecting the client type (WebRTC vs native)

`NV_DEVICE_PROFILE` selects which XR clients can connect. For the native iOS
and visionOS apps, set it to `auto-native`:

```bash
NV_DEVICE_PROFILE=auto-native uv run xr_render_demo
```

The environment value takes precedence over YAML. The `cloudxr_env` value in
`yaml/cloudxr_runtime.yaml` supplies the default only when the variable is
unset; `auto-webrtc` serves WebRTC and web XR clients. Native profiles omit the
static web page and its npm build but keep `/token`, `/cert`, and `/rtc` on the
hub, so the Apple Vision Pro app's default
`https://<host>:8080/token` request continues to work.

## GPU pinning for the XR side

`gpu_index` (int) in `yaml/cloudxr_runtime.yaml` selects the physical GPU
that the CloudXR compositor pins to. The cloudxr-runtime wrapper translates
the index to a PCI bus address via `nvidia-smi` and sets three selectors
(`CUDA_VISIBLE_DEVICES`, `VK_LOADER_DEVICE_SELECT`, `DRI_PRIME`) on its own
environment before spawning the native service. All three are required: the
compositor runs on Vulkan and needs the matching CUDA device for interop,
so on a multi-GPU host Vulkan and CUDA can otherwise land on different
physical GPUs.

The same three selectors are appended to `cloudxr.env` (under
`~/.cloudxr/run/`). The scene process sources that file when it spawns LOVR, so
LOVR inherits the pin; `openxr-service` picks it up the same way.

If `nvidia-smi` is missing, fails, reports no GPUs, or does not list the
requested index, the wrapper logs a warning and skips pinning rather than
failing startup.

The model-side fields live under `agent-samples/model-servers/yaml/`. Set them
to different GPUs so
the XR compositor and the agentic LLM do not share a card.

## Worker configuration

The worker reads two config files:

- `yaml/xr_render_demo_worker.yaml` — native capability endpoints, text-memory directory, and VAD tunables.
- `yaml/models.json` — fixed reuse-only model endpoint declarations consumed by
  `xr-ai-models`. Each entry maps a logical name
  (`llm`, `agent_llm`, `stt`, `tts`, `vlm`) to an adapter (preset or explicit
  spec), an endpoint, and a deployment. Edit this file to change which model
  runs where without touching the worker code.

## The LLM server

### Nemotron-3-Nano-Omni-30B-A3B-Reasoning — port 8108

A vLLM `execvp` shim: a small Python wrapper that reads YAML configuration,
sets `HF_HOME` and token environment variables, then `os.execvp`s into `vllm serve`. The
Python process is replaced by vLLM; vLLM owns the HTTP API, weight loading,
and tool calling from that point on.

`vllm serve` uses `--tool-call-parser qwen3_coder` and
`--reasoning-parser nemotron_v3`. The launcher selects NVFP4 on Blackwell and
FP8 on Ada, Hopper, or Ampere, with BF16 available as an explicit fallback.

One server backs both LLM roles in `yaml/models.json`: `agent_llm` runs the
supervisor and subagent tool-calling loops, and `llm` remains available for
untooled chat calls. Thinking stays off unless a call explicitly enables it.

## VLM — Cosmos3 Nano Reasoner

Port 8100 (`vlm_server`).

The render agent first selects a current or recorded frame and passes its
`ImageReference` to `query_image`. Its native capability set also
includes `query_images` for ordered collections and `query_video` for
timestamped frame sequences; all use the same `xr-ai-models` multi-image VLM
path. These raw
selectors and query tools are internal composition primitives; the reasoning
model sees only participant-safe perception facades. Recorded selection is
grouped into latest tools, whose windows end at the newest recording, and
historical tools, whose frame or video window begins at one absolute `start_us`.
Recorded-frame timestamps are estimates interpolated from chunk metadata.

There is a deliberate startup ordering constraint: `VoiceAgent` readiness
blocks on the VLM's `/health` endpoint, which
returns 200 only after weights are fully loaded. This ensures model memory has
settled before LOVR starts its Vulkan device, preventing a transient OOM race.

## STT — parakeet-tdt-0.6b-v3

Port 8103. NeMo ASR in-process. English-only, ~1.5 GB VRAM.

```
LiveKit mic (int16 PCM) → hub IPC (float32) → VoiceAgent
  → VAD/STT
      pre-roll buffer    last 10 chunks (~320 ms) kept at all times;
                         prepended to the utterance buffer on speech onset
                         so the first word's attack isn't clipped
      VAD                Silero (ONNX, 512-sample / 32 ms windows,
                         probability threshold) via shared xr-ai-vad util
      accumulates        audio while speaking
      finalizes when     silence ≥ 0.8s AND speech ≥ 0.15s
                         OR max utterance length (30s) hit
      filler filter      drops single- and multi-word filler utterances
                         ("um", "uh", "yeah", "okay", "mm-hmm", etc.)
      STT call           POST multipart/form-data WAV → stt-server :8103
  → accepted participant query
```

STT calls are serialized — an `stt_busy` flag prevents a new finalize while
one is in-flight.

## TTS — Piper

Served on port 8105. The voice runtime streams the supervisor's final
reply to Piper and returns the audio to the participant.

```
voice.output topic (final response)
  → VoiceAgent → private media-session TTS
      sentence-batched synthesis
      POST text → tts-server :8105 → WAV bytes
      RETURN_AUDIO IPC → hub → LiveKit → participant's headphones
```

`VoiceAgent` owns interruption handling. A new utterance while TTS is playing
triggers `ReturnAudioFlush`, so the hub clears the LiveKit audio queue for that
participant. Its interruption callback also cancels the participant's active
render-agent task without waiting on its cleanup in the media processor. Each
turn publishes one complete `voice.output` message; a superseded turn is
cancelled before its reply is published, so no partial stream is left open.

## Agent runtime and voice topology

```
VoiceAgent → private media session → VAD/STT ─→ voice.transcript topic
                                  └→ VoiceGate ─┐
           → typed hub text ingress ────────────┴→ xr-render.user-query topic
  → RenderAgent → SceneSupervisor → five focused subagents → voice.output topic
  → VoiceAgent → private media-session TTS → hub return audio
```

Pipecat is an internal implementation detail of `xr-ai-voice`; application
input, participant-scoped agent execution, and voice output use public SDK contracts.
An XR start failure sends `render.failed` (with the reason) on the data
channel and speaks a short failure notice through `voice.output`.

## Agentic loop

`SceneSupervisor` coordinates five focused subagent tools (placement,
appearance, object, vision, memory) over `xr_ai_tools.tool_calling.run_tool_loop`.
Each subagent runs its own inner tool loop against the scene and tracking
services. On each accepted `xr-render.user-query` event:

1. **Recent conversation** is recalled from `TextMemoryTools` and injected
   as context so the model understands references like "fix that" or "undo".
2. **Supervisor loop** (`run_tool_loop`, up to 12 iterations) — Nemotron-Omni
   :8108 routes the request to one or more subagent tools. Each subagent
   runs its own inner `run_tool_loop` (up to 4 iterations) against the
   scene, XR-tracking, and vision services.
3. **Verification pass** — only for turns with mutation intent (a
   mutating subagent was delegated, or the utterance contains a
   change-requesting verb): if no scene change is observed within 150 ms
   of the loop completing, a second `run_tool_loop` call is made so the
   supervisor can delegate remaining work or confirm a no-op turn.
4. **Conversation history persisted** — the user utterance and agent reply
   are written to `TextMemoryTools` under `{participant_id}:user` and
   `{participant_id}:agent` source keys.
5. **Final response** published as one complete `voice.output` message for
   the voice subscriber and TTS.

Mutation intent is read from the supervisor loop's own tool-call record
(which subagents were delegated) and the utterance's action verbs; the
scene diff decides whether the verification pass runs.

At worker startup, `app.py` composes the five subagent tools from the scene,
XR-tracking, spatial-math, vision, video-memory, and text-memory `Tool`
instances provided by `xr_ai_tools`. Lifecycle tools (`start_xr`,
`get_health`) remain worker-managed and are not exposed to the supervisor.

Final messages are also persisted through native
   text memory without model scratch output or tool traces.

## Native capability composition

The sample-local scene process owns scene state and LOVR. `openxr-service`
owns the headless tracking session, and `video-memory-service` owns recorded
video decoding. `LiveFrameSource` supplies current-frame requests. Relay-managed
native tools provide the typed surface over those services; the demo does
not launch or call MCP adapters.

### Spatial tool surface

The worker composes XR tracking with shared spatial-math tools. This
offloads vector arithmetic the LLM is bad at while keeping pose-dependent math
in one place:

- **Placement tools** move existing objects: `nudge` (signed user-frame
  offsets), `move_user_relative` (named direction from the user),
  `move_object_relative` (named relation to an anchor object),
  `move_inside`, `move_between`, `move_toward`, `move_toward_user`,
  `swap_positions`, and `move_to` (explicit coordinates). Every tool takes
  the instruction's exact words for each object, resolves them against the
  scene, performs the move itself, and returns the final position; the LLM
  never applies signs to user-frame axes or copies coordinates.
- **Appearance tool**: `recolor` resolves color words, RGB triples, and
  copy-the-color-of-an-object references deterministically.
- **Object tools** create and retire objects: `create_user_relative`,
  `create_object_relative` (one anchor, or the midpoint of two),
  `create_at`, `change_shape` (the scene replaces the object and returns
  its new id), `resize_object`, and `remove_object`.

## Prompt structure

Each agent has its own prompt file under
`worker/xr_render_demo_worker/` (supervisor plus five subagents, six files total).
The supervisor prompt routes requests to subagents; each subagent prompt
is worked-example heavy and opens with pronoun and reference resolution.

The placement agent's prompt maps utterance shapes to tools with contrast
pairs: a stated distance is a shift (`nudge`); a user-anchored destination
uses `move_user_relative`; a destination anchored on another object uses
`move_object_relative` (stacking is relation `above`); "into/inside" is
containment (`move_inside`). Every rule has a paired worked example, and
the highest-leakage failure modes carry explicit contrast examples.

## XR session lifecycle

CloudXR returns `XR_ERROR_FORM_FACTOR_UNAVAILABLE` from `xrGetSystem` until
a streaming client connects. LOVR cannot start before then.

```
1. User opens https://<host>:8080, grants mic + XR permissions
2. User clicks "Launch XR"
3. Client sends `xr.session.started` data message → hub IPC → worker
4. Worker invokes native `start_xr`
   → scene process spawns LOVR + waits for CloudXR in a background task
5. Worker polls `get_health` every 500 ms (up to 120s)
   lovr_started: true  → send `render.ready` to client → XR session unlocked
   spawn_error: "..."  → log + abort
6. On reconnect or refresh: `xr.session.started` arrives again
   → `_xr_started` is already True → skip spawn, send `render.ready`
   immediately
```

## Eval harness

Offline regression suite for the agentic loop, run against the live agent LLM.
It derives schemas from the worker's native tools and evaluates tool
effects against deterministic fixtures, so the live LOVR scene is not mutated.
Refer to
[`agent-samples/xr-render-demo/eval/README.md`](https://github.com/NVIDIA/xr-ai/blob/main/agent-samples/xr-render-demo/eval/README.md)
for the case format and the watch-mode loop. Run with:

```bash
uv run --project agent-samples/xr-render-demo/eval xr_render_demo_eval
```

## Tracing and debugging

Every turn carries a `trace_id` derived from the runtime's `message_id`
(set by `xr-ai-runtime` when it dispatches the `UserQuery`). The id is
logged at supervisor entry:

```
[worker] DEBUG supervisor turn participant=alice trace=<uuid> transcript="make a red sphere"
```

To trace one complete turn, filter worker logs by `trace=<uuid>` or by
`participant=<id>` and `timestamp_us=<value>` when the trace id is
unavailable (e.g. in offline eval). Key log landmarks:

| Event | Logger | Level |
|---|---|---|
| Turn received | `xr_render_demo_worker.supervisor` | DEBUG |
| Subagent delegated | `xr_render_demo_worker.agents.*` | DEBUG |
| Tool loop error | `xr_render_demo_worker.supervisor` | WARNING (ToolLoopError) |
| Subagent tool loop error | `xr_render_demo_worker.agents.*` | WARNING (ToolLoopError) |
| Turn failed | `xr_render_demo_worker.agent` | ERROR + traceback |

**Error policy.** Expected degradation paths (camera unavailable, scene
not started) surface as subagent result strings and reach the user as a
clarifying reply. Unexpected exceptions (`ValueError` on bad tool input
logs at DEBUG; all other exceptions log at ERROR with full traceback via
`logger.exception`) propagate so the runtime remains fail-fast and the
turn is recorded as failed in the runtime event log.

**Concurrent participants.** Each participant's turns are serialized by
the supervisor's per-participant lock, and a global scene lock serializes
the snapshot, mutation, and verification window across participants, so scene
turns from different participants queue rather than interleave.

### Prompt/eval overlap audit

Per `AGENTS.md` "Prompt-driven samples", the harness audits every worker
prompt against every tier's case inputs at startup and warns on overlap:
verbatim case utterances, case fixture ids, and any quoted prompt example
pairing an eval-vocabulary color with an eval-vocabulary shape. Clearing a
warning means changing the prompt, not the case.
