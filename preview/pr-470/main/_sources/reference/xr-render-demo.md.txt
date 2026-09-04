<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo — architecture

This architecture reference describes the xr-render-demo sample. Refer to
{doc}`/getting_started/quickstart` to run the sample. For inference-server
mechanics shared with other samples, refer to
{doc}`/components/ai-services`.

## Process stack

The orchestrator (`xr_render_demo`, via `xr-ai-launcher`) reuses all
model processes and starts its application processes serially; each owned
process touches its ready file before the next starts. `run_stack` is fail-fast:
any owned process exit terminates the application stack.

| Role | Ownership | Directory | Command | Port |
|---|---|---|---|---|
| hub | sample | `services/device-io-hub/` | `device_io_hub` | 8080 (HTTPS and `/rtc` WSS proxy); 7880 (plaintext LiveKit direct-debug path; firewall-restricted) |
| cloudxr | sample | `services/cloudxr-runtime/` | `cloudxr_runtime` | 48322 (WSS proxy for WebRTC profiles; unused by `auto-native`) |
| stt | reused | `services/stt-server/` | `stt_server` | 8103 |
| tts | reused | `services/piper-tts/` | `piper_tts_server` | 8105 |
| omni | reused | `services/nemotron-omni-llm/` | `nemotron_omni_llm_server` | 8108 (LLM) |
| vlm | reused | `services/vlm-server/` | `vlm_server` | 8100 (Cosmos VLM) |
| video-memory | sample | `services/video-memory-service/` | `video_memory_service` | 8310 (recorded-video typed RPC) |
| scene | sample | `agent-samples/xr-render-demo/scene/` | `xr_render_scene` | 8320 (typed RPC) |
| openxr-service | sample | `services/openxr-service/` | `openxr_service` | 8330 (typed RPC) |
| worker | sample | `agent-samples/xr-render-demo/worker/` | `xr_render_demo_worker` | — |

Before starting the stack, the orchestrator runs two setup steps:

- **Web vendor bundle** — builds the CloudXR and LiveKit ESM bundle via
  `client-samples/web-xr-build/build.sh` (skipped if already present;
  requires `npm`). Built only for WebRTC device profiles; native profiles
  never serve the web page, so the build (and its npm dependency) is skipped.
- **LOVR binary** — auto-downloads LOVR v0.18.0 AppImage to `deps/lovr/` if
  not present and sets `$LOVR_BIN`. Resolution order: `$LOVR_BIN` env var →
  `lovr_bin:` in `scene/scene_service.yaml` → cached AppImage → fresh download.

## Source map and extension points

`main.py` is the orchestrator. `worker/xr_render_demo_worker/app.py` composes
the SDK tools and `RenderAgent`; `agent.py` receives participant turns;
`supervisor.py` owns the top-level tool loop; and `scene.py` owns snapshots,
diffs, and move history. The `agents/` packages contain placement, appearance,
object, vision, and memory subagents. The separately packaged `scene/` process
owns LOVR and scene state, while `eval/` owns offline and live regression tiers.

To add a subagent, create a package under `agents/`, return its model-visible
`Tool` from a `make_<name>_agent()` factory, export it, compose it into the
supervisor, and add component eval cases. Add shared capability groups to
`xr-ai-tools` once two applications need them; otherwise keep a sample-specific
group with the worker and inject it into the relevant subagent.

Prompt-only changes need no rebuild. Edit the supervisor or subagent prompt in
place and add a corresponding eval case without copying the worked example's
specific vocabulary into the fixture.

## Selecting the client type (WebRTC vs native)

`NV_DEVICE_PROFILE` selects which XR clients can connect. For the native iOS
and visionOS apps, set it to `auto-native`. Run from
`agent-samples/xr-render-demo/`:

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

(worker-configuration)=
## Configuration

Run and edit the sample from `agent-samples/xr-render-demo/`. Each process
receives its checked-in configuration directly. Edit the owning file and
restart `xr_render_demo` to apply a change.

| File | Owns |
|---|---|
| `yaml/cloudxr_runtime.yaml` | CloudXR install state, EULA acceptance, client profile, compositor GPU, and environment overrides |
| `yaml/xr_render_demo_worker.yaml` | Native capability endpoints, text-memory directory, VAD, idle timeout, and voice-gate selection |
| `yaml/voice_gate.yaml` | Always-on speech or wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | Reused model adapters, endpoints, and readiness checks |
| `yaml/device_io_hub.yaml` | LiveKit, web and token servers, networking, and video recording |
| `yaml/video_memory_service.yaml` | Recorded-query endpoint, output directory, and GPU |
| `yaml/openxr_service.yaml` | OpenXR endpoint, CloudXR environment, and eval-only simulated pose |
| `scene/scene_service.yaml` | LOVR binary and app, scene endpoint, and CloudXR environment |

`NV_DEVICE_PROFILE` in the environment overrides
`cloudxr_env.NV_DEVICE_PROFILE` in `cloudxr_runtime.yaml`. `LOVR_BIN` similarly
overrides `lovr_bin` in `scene/scene_service.yaml`. Use a GPU index reported by
`nvidia-smi` for `gpu_index`; the CloudXR, video-memory, and model-server GPU
settings are independent and must be planned together. Keep
`allow_sim_pose: false` outside the live eval harness.

Each `models.json` entry maps a logical role (`llm`, `agent_llm`, `stt`, `tts`,
or `vlm`) to an adapter, endpoint, and deployment. Editing it changes which
operator-owned endpoint the demo consumes; it does not reconfigure or restart
the shared model. Refer to {doc}`/guides/customizing-model-servers` for
server-side model, port, GPU, or memory changes, then restart the persistent
model stack.

Refer to the generated {doc}`configuration <configuration>` reference for exact
fields, checked-in values, and adjacent YAML comments.

## The LLM server

### Nemotron-3-Nano-Omni-30B-A3B-Reasoning — port 8108

A small Python wrapper reads YAML configuration, sets the model environment,
and starts vLLM through the selected `pip` or Docker backend. The vLLM process
or container runs in a separate session so the shared model remains available
across stack restarts; the wrapper monitors its health and reports readiness.
The shared model-server stack reuses a healthy instance and replaces one whose
launch configuration changed.

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
  → VAD and STT
      pre-roll buffer    last 10 chunks (~320 ms) kept at all times;
                         prepended to the utterance buffer on speech onset
                         so the first word's attack isn't clipped
      VAD                Silero ONNX (512 samples per 32 ms window,
                         probability threshold) via shared xr-ai-vad util
      accumulates        audio while speaking
      finalizes when     silence ≥ 0.8s AND speech ≥ 0.15s
                         OR max utterance length (30s) hit
      STT call           POST multipart/form-data WAV → stt-server :8103
  → accepted participant query
```

The STT server serializes inference with a process-local lock because its shared
NeMo model is not reentrant.

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
VoiceAgent → private media session → VAD and STT ─→ voice.transcript topic
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
   The block is session-scoped: only the last eight turns since the
   participant last departed, and within ten minutes of the request, are
   injected; older history is memory_agent's to recall.
2. **Supervisor loop** (`run_tool_loop`, up to 12 iterations) — Nemotron-Omni
   :8108 routes the request to one or more subagent tools. Each subagent
   runs its own inner `run_tool_loop` (up to 4 iterations) against the
   scene, XR-tracking, and vision services.
3. **Verification pass** — only for turns with mutation intent (a
   mutating subagent was delegated, or a non-question utterance contains a
   change-requesting verb): if no scene change is observed within 150 ms
   of the loop completing, a second `run_tool_loop` call is made so the
   supervisor can delegate remaining work or confirm a no-op turn. Success
   is then evidence-backed: unless a scene write was applied somewhere in
   the turn (or a recolor found the requested color already in place), a
   completion claim is replaced with a fixed honest no-change sentence and
   any other non-question reply gets the no-change fact appended. Evidence
   is counted per turn, not per delegated task.
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
- **Appearance tool**: `recolor` takes a discriminated color source the
  model selects through the tool schema — `literal` (stated color words or
  numbers, garbles repaired), `scene_object` (copy an XR object), or
  `physical` (observe a real-world thing). The code dispatches on the
  variant without reinterpreting the phrase; the `physical` variant runs
  one camera query through `resolve_physical_color`, whose VLM answers in
  a closed grammar (`VISIBLE r g b` or `UNKNOWN`) and fails closed on
  anything else. The same source type drives the object tools' creations.
- **Object tools** create and retire objects: `create_user_relative`,
  `create_object_relative` (one anchor, or the midpoint of two),
  `create_at`, `change_shape` (the scene replaces the object and returns
  its new id), `resize_object`, and `remove_object`.

## Prompt structure

Each agent has its own prompt file under
`worker/xr_render_demo_worker/` (supervisor plus five subagents, six files total).
The supervisor prompt carries only cross-cutting turn discipline; the
routing rules and ownership boundaries live in each subagent's tool
`DESCRIPTION` constant (`agents/<name>/agent.py`), the surface the
supervisor's model reads when selecting a tool. The vision agent selects
between two descriptions at construction time: the full one when video
memory is wired, and a live-only variant that disclaims past-moment
questions when it is not. A few rules are stated in
both places on purpose: descriptions alone are under-weighted mid-loop, so
the supervisor prompt keeps a short backstop copy. Each subagent prompt is
worked-example heavy and opens with pronoun and reference resolution.

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
1. User opens https://<host>:8080, grants microphone and XR permissions
2. User clicks "Launch XR"
3. Client sends `xr.session.started` data message → hub IPC → worker
4. Worker invokes native `start_xr`
   → scene process spawns LOVR and waits for CloudXR in a background task
5. Worker polls `get_health` every 500 ms (up to 120s)
   lovr_started: true  → send `render.ready` to client → XR session unlocked
   spawn_error: "..."  → log and abort
6. On reconnect or refresh: `xr.session.started` arrives again
   → `_xr_started` is already True → skip spawn, send `render.ready`
   immediately
```

## Eval harness

The evaluation project derives schemas from the worker's native tools. Offline
tiers call the live agent LLM but apply tool effects to deterministic fixtures,
so they do not mutate the LOVR scene.

| Tier | Command | Runs against | Approximate cost |
|---|---|---|---|
| Supervisor routing | `xr_render_demo_eval_supervisor` | Fake subagents that record delegations | 15 seconds per case |
| Subagent components | `xr_render_demo_eval_subagents` | One real subagent over fake leaf functions | 30 seconds per case |
| End-to-end offline | `xr_render_demo_eval` | Supervisor and subagents over deterministic services | 30 minutes for the full corpus |
| Live smoke | `xr_render_demo_live_smoke` | Running demo stack | Minutes |
| Live pose matrix | `xr_render_demo_live_pose_matrix` | Running demo stack and simulated poses | Minutes |
| Live manipulation | `xr_render_demo_live_manip` | Running demo stack and real scene state | Minutes |
| Live speech noise | `xr_render_demo_live_garble` | Running demo stack with noisy utterances | Minutes |
| Live exploration | `xr_render_demo_live_explore` | Running demo stack with novel prompts | Minutes |
| Live perception routing | `xr_render_demo_live_perception` | Running demo stack and the worker's transcript store | Minutes |

Run all commands from `agent-samples/xr-render-demo/eval/`:

```bash
uv sync

# Full offline corpus, precision cases, and utterance battery
uv run xr_render_demo_eval

# Fast prompt-change gate
uv run xr_render_demo_eval utterances

# Filter by case name; unknown names fail
uv run xr_render_demo_eval move_left_one_meter between_two_spheres

# Routing and component tiers; the component command accepts an agent or case
uv run xr_render_demo_eval_supervisor
uv run xr_render_demo_eval_subagents placement
```

The offline tiers require only the agent LLM, which defaults to
`http://localhost:8108`. They do not require the demo stack, capability
services, or LOVR.

### Live drivers

Start the demo stack and set `allow_sim_pose: true` in
`../yaml/openxr_service.yaml` before running live evaluations:

```bash
uv run xr_render_demo_live_smoke
uv run xr_render_demo_live_pose_matrix
uv run xr_render_demo_live_manip
uv run xr_render_demo_live_garble
uv run xr_render_demo_live_explore
uv run xr_render_demo_live_perception
```

Live drivers join as synthetic participants, inject typed text, set simulated
head pose, and score real scene state. Preserve these isolation rules:

- Use a fresh participant ID for every case so transcript history cannot affect
  later supervisor behavior.
- Clear the scene through typed RPC between cases so stale objects cannot make
  referents ambiguous.
- Vary prompt phrasing across cases rather than building an artificial repeated
  self-history.
- Write the complete run log to a file, then filter that file. Do not filter the
  live command's output pipeline.
- Repeat each run three times before treating a near-tie result as a regression;
  model choices can vary even at temperature zero.

`xr_render_demo_live_garble` covers homophones, truncations, corrections, and
stutters. Its restraint scoring fails incorrect mutations and accepts a
clarifying response. `xr_render_demo_live_explore` sends novel conversational
phrasing and scores intent invariants. Promote any violation into a permanent
tier case before fixing it.

`xr_render_demo_live_perception` builds real multi-turn history, then sends
perception utterances and judges the reply text read back from the worker's
transcript store. The live stack publishes no camera track, so a correctly
routed perception turn must report an honest inability, recite no scene
object, and leave the scene unchanged; one case also injects cross-session
transcript turns to pin the supervisor's recall recency window, which keeps
them out of [Recent conversation]. It catches routing misses the offline
tiers cannot reproduce, whose recalled history is far shorter than a live
session's.

### Add or change a case

End-to-end cases are dictionaries in `cases.py`; use the existing cases for
pose overrides, multi-turn `history`, and undo `recent_moves` examples.
Precision and utterance cases are `Case` values in `harness.py`, routing cases
are in `supervisor.py`, component cases are in `subagents.py`, and live cases
are in the corresponding `live_*.py` module. A live manipulation case has this
shape:

```python
{
    "name": "my_case",
    "fixtures": [("sphere", x, y, z, r, g, b, size)],
    "prompt": "...",
    "check": lambda ids, objects: ...,  # True means pass
}
```

### Prompt-tuning loop

The current model follows worked examples and contrast pairs more reliably than
bare prohibitions. Pair every refusal example with a proceed example so the
restriction does not contaminate neighboring behavior. When examples cannot
produce reliable schema arguments, move deterministic resolution into code and
name the parameter for the text the model can copy. The `anchor_words`
parameter is the precedent: the model copies descriptors verbatim, and
`spatial_ops` resolves shapes, damaged nouns, and colors against scene state.

Run `uv run xr_render_demo_eval utterances` after every prompt or operations
change. Its cases take about three minutes. Run the longer scenario and
precision tiers before completing a tuning round.

(prompt-eval-overlap-audit)=

### Prompt and evaluation overlap audit

The harness audits every worker prompt against every tier's inputs at startup.
It warns when:

1. a case utterance appears verbatim in a prompt;
2. a case fixture ID appears in a prompt;
3. a quoted prompt example pairs an evaluation color with an evaluation shape.

Fix overlap by changing the prompt, not the case. Use colors such as teal,
lavender, magenta, or turquoise and shapes such as cone, cylinder, capsule, or
torus in worked examples. A case that passes only while its vocabulary appears
in the prompt measures recall rather than capability.

Offline tiers do not cover the live worker's voice pipeline, real LOVR effects,
or real camera perception. Use a live tier for those boundaries.

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
| Turn failed | `xr_render_demo_worker.agent` | ERROR with traceback |

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
