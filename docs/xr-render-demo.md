<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo — architecture

Read this when working inside `agent-samples/xr-render-demo/` or wiring a new
sample that follows the same multi-LLM + multi-MCP pattern. For the user-facing
quickstart see the [main README](../README.md#xr-render-demo-voice-driven-sphere-in-cloudxr).
For inference-server mechanics shared with other samples see
[`docs/ai-services.md`](ai-services.md).

## Process stack

The orchestrator (`xr_render_demo`, stdlib-only via `xr-ai-launcher`) starts
twelve processes concurrently. There is no startup ordering — every process
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
| worker | `agent-samples/xr-render-demo/worker/` | `xr_render_demo_worker` | — |

Before starting the stack, the orchestrator runs two setup steps:

- **Web vendor bundle** — builds the CloudXR + LiveKit ESM bundle via
  `client-samples/web-xr-build/build.sh` (skipped if already present;
  requires `npm`).
- **LOVR binary** — auto-downloads LOVR v0.18.0 AppImage to `deps/lovr/` if
  not present and sets `$LOVR_BIN`. Resolution order: `$LOVR_BIN` env var →
  `lovr_bin:` in `render_mcp.yaml` → cached AppImage → fresh download.

## Worker configuration

The worker reads two YAML files:

- `yaml/xr_render_demo_worker.yaml` — MCP base URLs and VAD tunables.
- `yaml/models.yaml` (path set by `models_yaml:` in the worker YAML) — model
  endpoint declarations consumed by `xr-ai-models`.  Each entry maps a logical
  name (`llm`, `agent_llm`, `stt`, `tts`, `vlm`) to a `kind: preset:<name>`
  and a `base_url`.  Edit this file to change which model runs where without
  touching the worker code.

## The two LLM servers

Both are vLLM `execvp` shims — a small Python wrapper that reads YAML config,
sets `HF_HOME` / token env vars, then `os.execvp`s into `vllm serve`. The
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
it. Requires a Blackwell GPU (B200 / RTX PRO 6000 / Jetson Thor) for native
FP4; swap to the BF16 variant for Hopper / Ampere.

This is the model that runs the multi-step tool-calling loop.

## VLM — Cosmos-Reason1-7B

Port 8100 (`vlm-server`) / port 8240 (`vlm-mcp`).

Loaded in-process by `vlm-server` via HuggingFace transformers
(Qwen2.5-VL architecture). `<think>…</think>` blocks are stripped before
returning. The `vlm-mcp` is a thin FastMCP wrapper exposing a single
`ask_image(question, image_path)` tool: it reads the PNG at that path,
base64-encodes it, and POSTs it to `vlm-server` as an `image_url` message.
Only invoked when the user asks a visual question.

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

Port 8105. `rhasspy/piper-voices` ONNX. Runs on CPU, ~100 ms / sentence. All
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

At worker startup, `list_tools()` is called on all four MCP clients
(`render-mcp`, `oxr-mcp`, `vlm-mcp`, `video-mcp`). Results are converted to
OpenAI tool format and held in memory. `start_xr` and `get_health` are
excluded from the tool list — the worker calls those directly, not the LLM.

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
   - Tool routing: `get_head_pose` / `position_ahead` / `position_relative`
     → `oxr-mcp`; `ask_image` → `vlm-mcp` (with path existence guard);
     video tools → `video-mcp`; everything else → `render-mcp`.
   - Progress message sent on `agent.progress` topic before each tool
     executes (data channel).
   - If `think=true`: reasoning preamble injected into system prompt
     (RESOLVE object → LOCATE coordinates → COMPUTE new position →
     EXECUTE). The `<think>` block stays private; only one short sentence
     goes to the user. Token budget: 2048 total / 1024 thinking budget.
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
| `oxr-mcp` | 8230 | `get_head_pose`, `position_ahead`, `position_relative`, `get_health` |
| `vlm-mcp` | 8240 | `ask_image` |
| `video-mcp` | 8210 | `list_live_participants`, `list_recorded_participants`, `get_video_stats`, `query_video`, `get_latest_frame`, `get_frame_at_time` |

`render-mcp` owns the LOVR child process and is the only thing that pushes
ops onto LOVR's scene socket (msgpack over ZMQ PUSH). `oxr-mcp` opens a
second headless OpenXR session (`XR_MND_HEADLESS`) separate from LOVR's
rendering session — both coexist without contention; the session opens
lazily on first tool call.

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
LOVR scene is not mutated). See
[`agent-samples/xr-render-demo/eval/README.md`](../agent-samples/xr-render-demo/eval/README.md)
for the case format and the watch-mode loop. Run with:

```bash
agent-samples/xr-render-demo/eval/eval.py
```
