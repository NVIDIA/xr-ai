<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo — architecture

This page describes the architecture of the xr-render-demo sample. For the
user-facing
quickstart, refer to the [main README](https://github.com/NVIDIA/xr-ai/blob/main/README.md#xr-render-demo-voice-driven-sphere-in-cloudxr).
For inference-server mechanics shared with other samples, refer to
[`docs/ai-services.md`](https://github.com/NVIDIA/xr-ai/blob/main/docs/ai-services.md).

## Process stack

The orchestrator (`xr_render_demo`, stdlib-only via `xr-ai-launcher`) starts
its processes concurrently. There is no startup ordering — every process
must tolerate peers that are not yet ready. `run_stack` is fail-fast: any
exit terminates the whole stack.

| Role | Directory | Command | Port |
|---|---|---|---|
| hub | `server-runtime/` | `xr_media_hub` | 8080 (https + wss /rtc proxy); LiveKit 7880 stays on 127.0.0.1 |
| cloudxr | `cloudxr-runtime/` | `cloudxr_runtime` | 48322 (WSS proxy) |
| stt | `ai-services/stt-server/` | `stt_server` | 8103 |
| tts | `ai-services/tts/piper/` | `piper_tts_server` | 8105 |
| vlm | `ai-services/vlm-server/` | `vlm_server` | 8100 |
| llm | `ai-services/llm/llama_nemotron/` | `llama_nemotron_llm_server` | 8106 |
| agent-llm | `ai-services/llm/nemotron3_nano/` | `nemotron3_nano_llm_server` | 8107 |
| vlm-mcp | `agent-mcp-servers/vlm-mcp/` | `vlm_mcp_server` | 8240 |
| video-mcp | `agent-mcp-servers/video-mcp/` | `video_mcp_server` | 8210 |
| render-mcp | `agent-mcp-servers/render-mcp/` | `render_mcp` | 8220 |
| oxr-mcp | `agent-mcp-servers/oxr-mcp/` | `oxr_mcp_server` | 8230 |
| vec-mcp | `agent-mcp-servers/vec-mcp/` | `vec_mcp_server` | 8250 |
| worker | `agent-samples/xr-render-demo/worker/` | `xr_render_demo_worker` | — |

Before starting the stack, the orchestrator runs two setup steps:

- **Web vendor bundle** — builds the CloudXR + LiveKit ESM bundle via
  `client-samples/web-xr-build/build.sh` (skipped if already present;
  requires `npm`). Built only for WebRTC device profiles; native profiles
  never serve the web page, so the build (and its npm dependency) is skipped.
- **LOVR binary** — auto-downloads LOVR v0.18.0 AppImage to `deps/lovr/` if
  not present and sets `$LOVR_BIN`. Resolution order: `$LOVR_BIN` env var →
  `lovr_bin:` in `render_mcp.yaml` → cached AppImage → fresh download.

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
`~/.cloudxr/run/`). `render-mcp` sources that file when it spawns LOVR, so
LOVR inherits the pin; `oxr-mcp` picks it up the same way.

If `nvidia-smi` is missing, fails, reports no GPUs, or does not list the
requested index, the wrapper logs a warning and skips pinning rather than
failing startup.

The corresponding model-side fields live under
`agent-samples/model-servers/yaml/<profile>/`. Set them to different GPUs so
the XR compositor and the agentic LLM do not share a card.

## Worker configuration

The worker reads two YAML files:

- `yaml/xr_render_demo_worker.yaml` — MCP base URLs and VAD tunables.
- `yaml/models.yaml` (path set by `models_yaml:` in the worker YAML) — model
  endpoint declarations consumed by `xr-ai-models`.  Each entry maps a logical
  name (`llm`, `agent_llm`, `stt`, `tts`, `vlm`) to a `kind: preset:<name>`
  and a `base_url`.  Edit this file to change which model runs where without
  touching the worker code.

## The LLM servers

Both are vLLM `execvp` shims — a small Python wrapper that reads YAML configuration,
sets `HF_HOME` and token environment variables, then `os.execvp`s into `vllm serve`. The
Python process is replaced by vLLM; vLLM owns the HTTP API, weight loading,
and tool calling from that point on.

### Llama-3.1-Nemotron-Nano-8B-v1 — port 8106 — fast reactive brain

`vllm serve` with `--tool-call-parser llama3_json --enable-auto-tool-choice`.
`enforce_eager` defaults to `false`. Used for three cheap, latency-sensitive
calls — none of which actually use tool calling:

- **Quick-ack** — fires in parallel with the agentic loop the moment an
  utterance lands. Returns `{"ack": "On it!", "think": false}` — a 3–6 word
  spoken acknowledgment. Also classifies whether the request needs spatial
  reasoning (`think: true/false`), so the 30B model knows before it starts
  whether to engage its thinking budget. Max 40 tokens, 8s timeout. The ack
  is always sent on the data channel (`agent.progress` topic); it is only
  also spoken via TTS when `think=true`, since that is when the user will
  actually be waiting 5–10s and needs to know they were heard.
- **Still-working messages** — if the agentic loop exceeds 5s, this model
  generates a short contextual phrase like *"Still finding the right
  position"* on a 7s repeat. Sent to the data channel only — never spoken,
  to avoid stacking up in the TTS queue behind the real response.

### NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 — port 8107 — agentic loop

`vllm serve` with `--tool-call-parser qwen3_coder` and
`--reasoning-parser nano_v3` (plugin auto-fetched from the model card into
`model_cache`). `enforce_eager` defaults to `true` — CUDA graph capture +
FlashInfer FP4 MoE autotune silently takes 3–8 minutes on cold start without
it. Requires a Blackwell GPU (B200, RTX PRO 6000, or Jetson Thor) for native
FP4; swap to the BF16 variant for Hopper or Ampere.

This is the model that runs the multi-step tool-calling loop.

## VLM — Cosmos-Reason1-7B

Port 8100 (`vlm-server`) and port 8240 (`vlm-mcp`).

Loaded in-process by `vlm-server` via HuggingFace transformers
(Qwen2.5-VL architecture). `<think>…</think>` blocks are stripped before
returning. The `vlm-mcp` is a thin FastMCP wrapper exposing a single
`ask_image(question, image_path)` tool: it reads the PNG at that path,
base64-encodes it, and POSTs it to `vlm-server` as an `image_url` message.
Visual queries from the user are handled by the brain-local
`look_at_current_frame(question)` tool (see tool routing below), which turns
the camera on automatically, grabs the live frame, and calls `vlm-server`
directly — bypassing `vlm-mcp` entirely for the default perception path.

There is a deliberate startup ordering constraint: the worker's
`wait_for_services` probe blocks on the VLM's `/health` endpoint, which
returns 200 only after weights are fully loaded. This ensures GPU 0 memory
has settled before LOVR starts its Vulkan device, preventing a transient
OOM race.

## STT — parakeet-tdt-0.6b-v3

Port 8103. NeMo ASR in-process. English-only, ~1.5 GB VRAM.

```
LiveKit mic (int16 PCM) → hub IPC (float32) → XRMediaHubTransport.input()
  → SttProcessor
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
  → TranscriptionFrame pushed downstream
```

STT calls are serialized — an `stt_busy` flag prevents a new finalize while
one is in-flight.

## TTS — Piper

Port 8105. `rhasspy/piper-voices` ONNX. Runs on CPU, ~100 ms per sentence. All
synthesis runs in a thread pool so the asyncio loop is never blocked.

```
TextFrame (from agentic loop final response, or quick-ack when think=true)
  → TtsProcessor
      sentence-batched synthesis
      POST text → tts-server :8105 → WAV bytes
      RETURN_AUDIO IPC → hub → LiveKit → participant's headphones
```

`allow_interruptions=True` in the Pipecat pipeline. A new utterance while TTS
is playing triggers `ReturnAudioFlush` → hub clears the LiveKit audio queue
for that participant.

## Pipecat pipeline

```
XRMediaHubTransport.input()
  → SttProcessor          (Silero VAD → utterance → parakeet STT
                           → TranscriptionFrame)
  → RenderSceneProcessor  (quick-ack + agentic loop → TextFrame)
  → TtsProcessor          (TextFrame → Piper TTS → return audio)
  → XRMediaHubTransport.output()
```

## Agentic loop

At worker startup, `list_tools()` is called on all MCP clients
(`render-mcp`, `oxr-mcp`, `vlm-mcp`, `video-mcp`, `vec-mcp`). Results are
converted to OpenAI tool format and held in memory. `start_xr` and
`get_health` are excluded from the tool list — the worker calls those
directly, not the LLM.

On each `TranscriptionFrame`:

1. **Quick-ack** fires immediately (Llama-8B :8106, parallel task).
2. **Still-working timer** starts (fires at 5s, repeats every 7s, data
   channel only).
3. **Pre-fetch** (concurrent): `get_scene_state` + `get_head_pose` +
   `position_ahead(1.5)` — results injected into the user message so the
   model skips those tool calls and goes straight to the operation.
4. **Nemotron-30B :8107** runs with `tools=[…]`, up to 10 iterations:
   - Model emits `tool_calls` → worker routes and executes → result appended
     to conversation → next iteration.
   - Tool routing: `look_at_current_frame` → **brain-local** (intercepts before
     MCP routing: turns camera on, grabs live frame, calls `vlm-server` directly);
     oxr-mcp tools (`get_head_pose`, `position_ahead`, `position_relative`,
     `place_user_relative`, `place_object_relative`, `place_inside_by_id`,
     `displace_object`, `displace_objects`) → `oxr-mcp`; vec-mcp tools
     (`between_anchors`, `world_offset`, `along_direction`, `scale_value`) →
     `vec-mcp`; `ask_image` → `vlm-mcp` (with path existence guard); video
     tools → `video-mcp`; everything else → `render-mcp`.
   - Progress message sent on `agent.progress` topic before each tool
     executes (data channel).
   - If `think=true`: reasoning preamble injected into system prompt
     (RESOLVE object → LOCATE coordinates → COMPUTE new position →
     EXECUTE). The `<think>` block stays private; only one short sentence
     goes to the user. Token budget: 2048 total, 1024 thinking budget.
   - If thinking fills the token budget without a tool call
     (`finish_reason=length`): retry the same iteration with
     `needs_thinking=False`.
   - If the model outputs a bare tool name as text instead of a proper tool
     call: worker synthesizes a no-arg tool call and continues.
5. **Final response** sent on `agent.response` topic and as a `TextFrame`
   downstream to TTS.
6. **Turn appended** to a rolling 4-turn history buffer — injected as
   context in future turns so the model understands "fix that", "undo",
   "the one I just added".

## MCP servers

| Server | Port | Tools |
|---|---|---|
| `render-mcp` | 8220 | `start_xr`, `get_health`, `add_primitive`, `update_primitive`, `remove_primitive`, `get_scene_state` |
| `oxr-mcp` | 8230 | `get_head_pose`, `position_ahead`, `position_relative`, `place_user_relative`, `place_object_relative`, `place_inside_by_id`, `displace_object`, `displace_objects`, `get_health` |
| `vec-mcp` | 8250 | `between_anchors`, `world_offset`, `along_direction`, `scale_value` |
| `vlm-mcp` | 8240 | `ask_image` |
| `video-mcp` | 8210 | `list_live_participants`, `get_frame_from_time` (always); `list_recorded_participants`, `get_video_stats`, `query_video` (recording enabled only); `get_latest_frame` (deprecated) |

`render-mcp` owns the LOVR child process and is the only thing that pushes
ops onto LOVR's scene socket (msgpack over ZMQ PUSH). `oxr-mcp` opens a
second headless OpenXR session (`XR_MND_HEADLESS`) separate from LOVR's
rendering session — both coexist without contention; the session opens
lazily on first tool call.

### Spatial tool surface

The tool surface is split across `oxr-mcp` (pose-aware named-direction
helpers) and `vec-mcp` (pure-math primitives). The split offloads vector
arithmetic the LLM is bad at while keeping pose-dependent math in one place:

- **oxr-mcp named-direction helpers** take a `direction` enum (`front`,
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
- **vec-mcp pure-math primitives** are pose-independent:
  - `between_anchors(a_x, a_y, a_z, b_x, b_y, b_z)`: component-wise midpoint.
  - `world_offset(origin_x, origin_y, origin_z, dx, dy, dz)`:
    axis-aligned world-Y-up shift.
  - `along_direction(origin_x, origin_y, origin_z, target_x, target_y,
    target_z, distance)`: origin moved `distance` toward target. Used
    for "closer to or further from <named-obj>", which the user-frame
    helpers can't model.
  - `scale_value(current, factor)`: scalar multiplication for sizes.

## Prompt structure

The system prompt at `worker/prompts/system.txt` is worked-example heavy.
It opens with pronoun and reference resolution, then routes placement
utterances through sequential checks before the LLM picks a tool:

1. **FIRST CHECK**: `"between"`/`"middle"`/`"halfway"` → route to
   `between_anchors`; stop considering other placement tools.
2. **SECOND CHECK**: anchor is the user (`"me"`/`"my"`) → route to
   `place_user_relative`; `place_object_relative` with `origin=user_pos`
   returns the wrong side of the user.
3. **THIRD CHECK**: proximity to a named object (`"closer to <obj>"`,
   `"toward <obj>"`) → route to `along_direction`. The user's facing
   direction is unrelated to where the target object sits, so
   `displace_object` is wrong here.

Every rule that's not obviously self-explanatory has a paired WORKED
EXAMPLE (concrete coords + tool call) and, for the highest-leakage
failure modes, a WORKED ANTI-EXAMPLE. The two-step contract is
hammered: every move emits one math-tool call followed by exactly one
`add_primitive`/`update_primitive` call carrying all three of `x`,
`y`, `z` from the math result.

## XR session lifecycle

CloudXR returns `XR_ERROR_FORM_FACTOR_UNAVAILABLE` from `xrGetSystem` until
a streaming client connects. LOVR cannot start before then.

```
1. User opens https://<host>:8080, grants mic + XR permissions
2. User clicks "Launch XR"
3. Client sends `xr.session.started` data message → hub IPC → worker
4. Worker calls render-mcp `start_xr`
   → render-mcp spawns LOVR + waits for CloudXR in a background task
5. Worker polls `get_health` every 500 ms (up to 120s)
   lovr_started: true  → send `render.ready` to client → XR session unlocked
   spawn_error: "..."  → log + abort
6. On reconnect / refresh: `xr.session.started` arrives again
   → `_xr_started` is already True → skip spawn, send `render.ready`
   immediately
```

## Eval harness

Offline regression suite for the agentic loop, run against the live model
stack (no LLM/MCP mocks; render-mcp tools are fake-succeeded so the live
LOVR scene is not mutated). Refer to
[`agent-samples/xr-render-demo/eval/README.md`](https://github.com/NVIDIA/xr-ai/blob/main/agent-samples/xr-render-demo/eval/README.md)
for the case format and the watch-mode loop. Run with:

```bash
agent-samples/xr-render-demo/eval/eval.py
```

### Prompt/eval overlap audit

The harness audits the system prompt's worked-example blocks against every
case fixture at startup and warns if they share specifics: verbatim user
utterances (≥12 chars), scene coordinates rendered as `(x.xx, y.yy, z.zz)`,
`recent_moves` coords, or any reserved colour or shape word that appears in
both a case fixture and a worked-example block. This guards against the eval
cases overfitting to the prompt's worked examples.
