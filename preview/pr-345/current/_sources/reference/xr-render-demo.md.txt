<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo — architecture

This page describes the architecture of the xr-render-demo sample. Start with
{doc}`/getting_started/quickstart` to run it. For inference-server mechanics
shared with other samples, see {doc}`/components/ai-services`.

## Process stack

The orchestrator (`xr_render_demo`, stdlib-only via `xr-ai-launcher`) starts its processes serially in declaration order; each touches its ready file before the next process starts. `run_stack` is fail-fast: any
exit terminates the whole stack.

| Role | Directory | Command | Port |
|---|---|---|---|
| hub | `services/device-io-hub/` | `device_io_hub` | 8080 (https + wss /rtc proxy); LiveKit 7880 stays on 127.0.0.1 |
| cloudxr | `services/cloudxr-runtime/` | `cloudxr_runtime` | 48322 (WSS proxy) |
| stt | `services/stt-server/` | `stt_server` | 8103 |
| tts | `services/piper-tts/` | `piper_tts_server` | 8105 |
| omni | `services/nemotron-omni-llm/` | `nemotron_omni_llm_server` | 8108 (LLM) |
| vlm | `services/vlm-server/` | `vlm_server` | 8100 (Cosmos VLM) |
| video-memory | `services/video-memory-service/` | `video_memory_service` | 8310 (recorded-video typed RPC) |
| scene | `agent-samples/xr-render-demo/scene/` | `xr_render_scene` | 8320 (typed RPC) |
| openxr-service | `services/openxr-service/` | `openxr_service` | 8330 (typed RPC) |
| worker | `agent-samples/xr-render-demo/worker/` | `xr_render_demo_worker` | — |

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

The corresponding model-side fields live under
`agent-samples/model-servers/yaml/<profile>/`. Set them to different GPUs so
the XR compositor and the agentic LLM do not share a card.

## Worker configuration

The worker reads two config files:

- `yaml/xr_render_demo_worker.yaml` — native capability endpoints, text-memory directory, and VAD tunables.
- `yaml/models.local.json` (deployment profile set by `models_config:` in the
  worker YAML): model endpoint and deployment declarations consumed by
  `xr-ai-models` and the orchestrator.  Each entry maps a logical name
  (`llm`, `agent_llm`, `stt`, `tts`, `vlm`) to an adapter (preset or explicit
  spec), an endpoint, and a deployment.  Edit this file to change which model
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

One server backs both LLM roles in `yaml/models.yaml`: `agent_llm` runs the
multi-step tool-calling loop and `llm` serves two cheap, latency-sensitive
calls. Thinking stays off unless a call explicitly enables it.

- **Quick-ack** — awaited before the agentic loop starts, the moment an
  utterance lands. Returns `{"ack": "On it!", "think": false}` — a 3–6 word
  spoken acknowledgment. Also classifies whether the request needs
  open-ended reasoning (`think: true/false`): positional operations always
  run without thinking because the math tools compute exact answers, so
  thinking is reserved for vague corrections and free-form compositions
  no tool pattern settles. Max 40 tokens, 8s timeout. The ack
  is sent on the data channel (`agent.progress` topic) and spoken on every
  turn so the user immediately knows they were heard.
- **Still-working messages** — if the agentic loop exceeds 5s, this model
  generates a short contextual phrase like *"Still finding the right
  position"* on a 10s repeat. Sent to the data channel only — never spoken,
  to avoid stacking up in the TTS queue behind the real response.

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
voice.output topic (agentic-loop quick-ack or final response)
  → VoiceAgent → private media-session TTS
      sentence-batched synthesis
      POST text → tts-server :8105 → WAV bytes
      RETURN_AUDIO IPC → hub → LiveKit → participant's headphones
```

`VoiceAgent` owns interruption handling. A new utterance while TTS is playing
triggers `ReturnAudioFlush`, so the hub clears the LiveKit audio queue for that
participant. Its interruption callback also cancels the participant's active
render-agent task without waiting on its cleanup in the media processor. A
consumer-aborted render stream closes its scene generator without publishing a
terminator to the already-closed voice stream; producer supersession completes
the old stream before the replacement starts.

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
Lifecycle failures publish notices to a sample-local runtime topic instead of
manufacturing voice-pipeline frames.

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
3. **Verification pass** — if no scene change was observed within 150 ms of
   the loop completing, a second `run_tool_loop` call is made so the
   supervisor can delegate remaining work or confirm a no-op turn.
4. **Conversation history persisted** — the user utterance and agent reply
   are written to `TextMemoryTools` under `{participant_id}:user` and
   `{participant_id}:agent` source keys.
5. **Final response** published as `voice.output` chunks for the voice
   subscriber and TTS.

Subagents signal scene mutation via `SceneContext.mark_delegated`; the
supervisor uses this to distinguish scene-mutating turns from conversational
turns when deciding whether to run the verification pass.

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

- **Pose-aware named-direction helpers** take a `direction` enum (`front`,
  `back`, `left`, `right`, `above`, `below`, plus `next_to` on
  `place_object_relative`) and always-positive `distance`. The LLM never
  applies signs to user-frame axes.
  - `place_user_relative(direction, distance)`: user-anchored teleport
    ("above my head", "to my left 1 m").
  - `place_object_relative(origin_x, origin_y, origin_z, direction, distance)`:
    object-anchored teleport. `direction="front"` means *toward the user*;
    `"back"` means *away*. Left/right/above/below map literally.
  - `displace_object(current_x, current_y, current_z, right, up, forward)`:
    user-frame signed-delta on an existing object. Multi-axis ("up and
    to the left") in one call.
  - `displace_objects(object_ids, current_xs, current_ys, current_zs,
    right, up, forward)`: batch user-frame delta over N objects. Returns
    `{"items": [{obj_id, x, y, z}, …]}` so the model fans out to N
    `update_primitive` calls with one math call total.
  - `place_inside_by_id(movee_id, container_x, container_y, container_z)`:
    containment for "put X in Y". Argument names (`movee_id` paired
    with `container_*`) force the model to pick the right noun's coords;
    the return shape feeds straight into `update_primitive`.
- **Pure-math primitives** are pose-independent:
  - `between_anchors(a_x, a_y, a_z, b_x, b_y, b_z)`: component-wise midpoint.
  - `world_offset(origin_x, origin_y, origin_z, dx, dy, dz)`:
    axis-aligned world-Y-up shift.
  - `along_direction(origin_x, origin_y, origin_z, target_x, target_y,
    target_z, distance)`: origin moved `distance` toward target. Used
    for "closer to or further from <named-obj>", which the user-frame
    helpers can't model.
  - `scale_value(current, factor)`: scalar multiplication for sizes.

## Prompt structure

Each agent has its own prompt file under
`worker/xr_render_demo_worker/` (supervisor plus five subagents, six files total).
The supervisor prompt routes requests to subagents; each subagent prompt
is worked-example heavy and opens with pronoun and reference resolution.

The placement agent routes placement utterances through sequential checks:

1. **FIRST CHECK**: `"between"`/`"middle"`/`"halfway"` → route to
   `between_anchors`; stop considering other placement tools.
2. **SECOND CHECK**: anchor is the user (`"me"`/`"my"`) → route to
   `place_user_relative`; `place_object_relative` with `origin=user_pos`
   returns the wrong side of the user.
3. **THIRD CHECK**: proximity to a named object (`"closer to <obj>"`,
   `"toward <obj>"`) → route to `along_direction`. The user's facing
   direction is unrelated to where the target object sits, so
   `displace_object` is wrong here.

Every rule has a paired WORKED EXAMPLE and, for the highest-leakage
failure modes, a WORKED ANTI-EXAMPLE.

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
6. On reconnect / refresh: `xr.session.started` arrives again
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
the supervisor's per-participant lock. Different participants can run
concurrently; the underlying scene service serializes conflicting writes.

### Prompt/eval overlap audit

Per `AGENTS.md` "Prompt-driven samples", the harness audits every worker
prompt against every tier's case inputs at startup and warns on overlap:
verbatim case utterances, case fixture ids, and any quoted prompt example
pairing an eval-vocabulary color with an eval-vocabulary shape. Clearing a
warning means changing the prompt, not the case.
