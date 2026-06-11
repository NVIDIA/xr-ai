<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Dependency Map

> **AGENTS: This file is mandatory to maintain.**
> Any change to a `pyproject.toml`, a YAML config/example, a documented
> interface, or an architectural decision **must** be reflected here in the
> same commit. A change is not complete until this file is up to date.

---

## Python version

Every `pyproject.toml` in this repo pins `requires-python = ">=3.11,<3.13"` by
convention. The upper bound exists because `PyNvVideoCodec` (used by
`xr-media-hub` and `video-mcp-server` for NVENC encode / NVDEC decode) does not
yet publish wheels for Python 3.13. With the cap in place, `uv sync` will pick
3.12 even on a host where 3.13 is also installed. Loosen the upper bound only
after `PyNvVideoCodec` ships 3.13 wheels.

A project may state a different range when its dependencies require it; the
constraints stay honest because `.github/workflows/lock-check.yml` runs
`uv lock` on every `pyproject.toml` in the repo on every push and PR. `uv lock`
resolves universally across the full `requires-python` range stated in each
file, so a single invocation per project proves the declared range is
satisfiable end-to-end. Drift between `requires-python` and what the dep graph
actually supports fails CI.

CI matrices:
- `.github/workflows/tests.yml` — pytest across Python 3.11 and 3.12.
- `.github/workflows/lock-check.yml` — `uv lock` per project (no Python matrix
  needed; uv covers the range internally).

---

## Internal packages

```
xr-ai-agent  (agent-sdk/)
    └── pyzmq >=27.0
    └── msgpack >=1.0

xr-ai-pipecat  (agent-sdk/xr-ai-pipecat/)
    └── xr-ai-agent     [editable: ..]
    └── xr-ai-logging   [editable: ../../utils/xr-ai-logging]
    └── xr-ai-models    [editable: ../xr-ai-models]
    └── xr-ai-vad       [editable: ../../utils/xr-ai-vad]
    └── xr-ai-voicegate [editable: ../../utils/xr-ai-voicegate]
    └── pipecat-ai >=0.0.46
    └── numpy >=1.24
    └── scipy >=1.11
    └── httpx >=0.27
    └── fastmcp >=0.4
    Unified Pipecat voice pipeline. Owns the transport bridge to
    ProcessorEndpoint (ZMQ IPC) plus the four library FrameProcessors —
    VadSttProcessor, VoiceGateProcessor, BrainProcessor, StreamingTtsProcessor —
    composed by ``make_voice_pipeline``. Resamples hub float32 audio →
    16 kHz int16 for STT, converts TTS int16 PCM back to float32 AudioChunks
    for return. SttClient / TtsClient are thin wrappers around xr-ai-models'
    OpenAICompatSTT / OpenAICompatTTS — PCM→WAV conversion is handled by
    the SDK. httpx is retained for http_probe() readiness checks.
    Not a dep of xr-ai-agent itself — import only in workers that use Pipecat.

xr-ai-voicegate  (utils/xr-ai-voicegate/)
    └── numpy >=1.24
    └── pyyaml >=6.0
    Pipecat-free speech-input opt-in gate. Owns the magic-phrase + follow-up
    + STOP ladder, the lazy listening chime synthesized at the TTS sample
    rate, and the participant-joined greeting hook. Workers feed STT
    transcripts via ``feed`` and register handlers — either one-at-a-time via
    ``on_*`` setters or together via ``bind(...)``. Consumed inside
    xr-ai-pipecat by ``VoiceGateProcessor`` so sample workers don't import it
    directly when they use the unified pipeline.

xr-ai-models  (agent-sdk/xr-ai-models/)
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── httpx >=0.27
    └── pyyaml >=6.0
    Unified service protocols (LLMService, VLMService, STTService, TTSService)
    and OpenAI-compatible HTTP clients that cover every in-tree model backend
    (vLLM-served VLM/LLMs, NeMo Parakeet STT, Piper/Magpie TTS).  Per-model
    quirks live behind one seam: reasoning-field aliasing (nano_v3 →
    `reasoning`, nemotron_v3 → `reasoning_content`), `chat_template_kwargs`
    plumbing for `enable_thinking` / `thinking_budget`, and built-in presets
    for the seven in-tree services.  Future backends (LiteLLM, vendor SDKs)
    plug in as new `kind`s in `factory.py::make_*` without touching the
    protocols or callers.  Workers depend on this instead of rolling their
    own httpx wrappers.

xr-ai-launcher  (utils/xr-ai-launcher/)
    └── (stdlib only — zero runtime deps)

xr-ai-logging  (utils/xr-ai-logging/)
    └── loguru >=0.7

xr-ai-vllm  (utils/xr-ai-vllm/)
    └── (stdlib only — zero runtime deps)
    Pluggable vLLM hosting for the four vLLM-backed services.  Dispatches to
    either pip-installed `vllm serve` or `docker run nvcr.io/nvidia/vllm:<tag>`
    based on each YAML's `vllm_backend:` key.  Stays stdlib-only so docker mode
    does not pull vllm/torch/etc. into the wrapper's venv just to manage a
    container.  Imported by the four vllm wrappers and by the orchestrator
    `--stop` flow.  Besides `serve` / `stop_persistent_servers`, exposes the
    shared wrapper helpers `resolve_model_cache`, `load_config`, `setup_hf_env`,
    and `gpu_compute_major` (all stdlib-only; pyyaml is imported function-locally
    inside `load_config` so the `--stop` path stays dependency-free).

xr-ai-vad  (utils/xr-ai-vad/)
    └── numpy >=1.24
    └── silero-vad >=5.1  (pulls onnxruntime transitively)
    └── torch >=2.0       (detector.py imports torch directly)
    └── onnxruntime >=1.17
    Shared per-participant Silero VAD utterance detector for agent workers
    that ingest microphone audio.  Uses the ONNX backend (no GPU required
    at runtime).  Consumes raw int16 PCM bytes and emits int16 PCM utterance
    bytes via an async ``on_utterance`` callback; an optional
    ``on_speech_start`` hook fires when speech first crosses ``min_speech``
    for speculative downstream warmup (e.g. start the camera before STT
    completes).

xr-media-hub  (server-runtime/)
    └── xr-ai-agent  [editable: ../agent-sdk]
    └── pyzmq >=27.0
    └── livekit >=0.17
    └── livekit-api >=0.7
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── httpx >=0.27
    └── websockets >=12.0
    └── numpy >=1.24
    └── pyyaml >=6.0
    └── cryptography >=42.0
    PyNvVideoCodec >=1.0 (NVENC H.264 encoder; used when video_recording.enabled: true)

transcript-mcp-server  (agent-mcp-servers/transcript-mcp/)
    └── uvicorn[standard] >=0.29
    └── fastmcp >=0.4
    └── pyyaml >=6.0
    Pure FastMCP — every operation is an MCP tool at /mcp (no REST).
    Storage: JSONL files per participant in configurable transcripts_dir.

vlm-mcp-server  (agent-mcp-servers/vlm-mcp/)
    └── uvicorn[standard] >=0.29
    └── fastmcp >=0.4
    └── pyyaml >=6.0
    └── Pillow >=10.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-models   [editable: ../../agent-sdk/xr-ai-models]
    Pure FastMCP — one tool at /mcp (no REST). Reads a local image file,
    encodes it as a JPEG data URL, and calls vlm-server via xr-ai-models
    ``OpenAICompatVLM``. Back-compat: legacy ``vlm_server:`` URL key is
    still accepted with a deprecation warning.

video-mcp-server  (agent-mcp-servers/video-mcp/)
    └── uvicorn[standard] >=0.29
    └── fastmcp >=0.4
    └── pyyaml >=6.0
    └── xr-ai-agent  [editable: ../../agent-sdk]
    └── PyNvVideoCodec >=1.0
    └── Pillow >=10.0
    └── numpy >=1.24
    Pure FastMCP — every operation is an MCP tool at /mcp (no REST).
    Reads NVENC H.264 chunks written by the hub from disk for historical
    queries; connects to the hub as a ProcessorEndpoint to fetch live
    frames for `get_latest_frame`. Decodes chunks via NVDEC and
    re-encodes selected frames as PNG via Pillow.

cloudxr-runtime  (cloudxr-runtime/)
    └── isaacteleop[cloudxr]
    └── pyyaml

render-mcp-server  (agent-mcp-servers/render-mcp/)
    └── xr-ai-launcher  [editable: ../../utils/xr-ai-launcher] (ManagedProcess + load_cloudxr_env)
    └── pyzmq >=27.0       (PUSH socket → LOVR; libzmq.so reused by LOVR FFI)
    └── msgpack >=1.0      (wire format for LOVR ops)
    └── pyyaml >=6.0
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── fastmcp >=0.4
    Spawns LOVR (the OpenXR rendering app) on the first start_xr call.
    cloudxr-runtime must start before render-mcp (serial launch order);
    cloudxr.env is read synchronously via load_cloudxr_env at start_xr time.

oxr-mcp-server  (agent-mcp-servers/oxr-mcp/)
    └── xr-ai-launcher  [editable: ../../utils/xr-ai-launcher] (load_cloudxr_env)
    └── isaacteleop                                (headless OpenXR + HeadTracker)
    └── pyyaml >=6.0
    └── uvicorn[standard] >=0.29
    └── fastmcp >=0.4
    Pure FastMCP at /mcp. Reads pose from CloudXR via a second (headless)
    OpenXR session; runs alongside LOVR's rendering session.
    cloudxr-runtime must start before oxr-mcp (serial launch order).

vec-mcp-server  (agent-mcp-servers/vec-mcp/)
    └── uvicorn[standard] >=0.29
    └── fastmcp >=0.4
    └── pyyaml >=6.0
    Pure FastMCP at /mcp. Deterministic spatial-math primitives
    (between_anchors, world_offset, along_direction, scale_value).
    Offloads vector arithmetic from the LLM.

xr-ai-tests  (tests/)
    └── xr-ai-agent             [editable: ../agent-sdk]
    └── xr-ai-models            [editable: ../agent-sdk/xr-ai-models]
    └── xr-ai-pipecat           [editable: ../agent-sdk/xr-ai-pipecat]
    └── xr-media-hub            [editable: ../server-runtime]    (pulls in livekit, livekit-api for the wss /rtc proxy + room-client tests)
    └── xr-ai-launcher          [editable: ../utils/xr-ai-launcher]
    └── xr-ai-logging           [editable: ../utils/xr-ai-logging]
    └── xr-ai-vad               [editable: ../utils/xr-ai-vad]
    └── xr-ai-voicegate         [editable: ../utils/xr-ai-voicegate]
    └── xr-ai-vllm              [editable: ../utils/xr-ai-vllm]
    └── transcript-mcp-server   [editable: ../agent-mcp-servers/transcript-mcp]
    └── vlm-mcp-server          [editable: ../agent-mcp-servers/vlm-mcp]
    └── render-mcp              [editable: ../agent-mcp-servers/render-mcp]
    └── video-mcp-server        [editable: ../agent-mcp-servers/video-mcp]
    └── vec-mcp-server          [editable: ../agent-mcp-servers/vec-mcp]
    └── pytest >=8.0
    └── pytest-asyncio >=0.23
    └── numpy >=1.24
    └── fastmcp >=0.4   (only used by tests marked `gpu`)
    └── Pillow >=10.0   (only used by tests marked `gpu`)
    └── pyyaml >=6.0    (only used by tests marked `gpu`)
    The unmarked suite is multi-client / multi-agent integration tests over
    the IPC layer, driven via ZMQ `ipc://` only — no Docker / LiveKit /
    NVENC required. Also covers unit tests for the leaf util packages
    (launcher, logging, vllm), a CI-viable subprocess test for
    CPU-viable subprocess smoke tests for transcript-mcp-server and
    vec-mcp-server (fastmcp pulled in transitively), and the vlm-mcp /
    render-mcp adapter surfaces (mocked upstreams). oxr-mcp is not
    included: it needs native isaacteleop + a CloudXR runtime, so its
    smoke test self-skips on CPU (see tests/README.md).

    Tests marked `@pytest.mark.gpu` are the local-only set (skipped by
    `-m "not gpu"` in CI). They spawn real ai-services via `uv run` (e.g.
    `test_gpu_stt_server.py`, `test_gpu_video_mcp.py`), import
    `livekit.rtc` directly to drive `_room_client.py`, exercise NVENC /
    NVDEC via PyNvVideoCodec, and shell out to `docker` to manage a
    LiveKit container — `livekit`, `livekit-api`, `PyNvVideoCodec`, and
    `docker` all come in transitively via `xr-media-hub` /
    `video-mcp-server` rather than redeclared here.

vlm-server  (ai-services/vlm-server/)
    └── vllm >=0.12.0
    └── pyyaml >=6.0
    └── hf-transfer >=0.1.4
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-vllm     [editable: ../../utils/xr-ai-vllm]
    Model: nvidia/Cosmos-Reason1-7B (Qwen2.5-VL architecture, vLLM).
    Wrapper Popens `vllm serve` so the launcher's killpg() does not reach
    vLLM — model survives stack restarts (see docs/changelog.md 2026-05-05).
    vllm_backend: pip|docker — pip path uses the wrapper's vllm; docker path
    runs `nvcr.io/nvidia/vllm:<tag> vllm serve …` instead.

stt-server  (ai-services/stt-server/)
    └── nemo_toolkit[asr] >=2.5
    └── lightning >2.2.1,<=2.4.0    # routed to github.com/Lightning-AI/pytorch-lightning
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── python-multipart >=0.0.9
    └── pyyaml >=6.0
    Model: nvidia/parakeet-tdt-0.6b-v3 (NeMo ASR, in-process)

magpie-tts-server  (ai-services/tts/magpie/)
    └── nemo_toolkit[tts] >=2.5
    └── lightning >2.2.1,<=2.4.0    # routed to github.com/Lightning-AI/pytorch-lightning
    └── soundfile >=0.12
    └── numpy >=1.24
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    Model: nvidia/magpie_tts_multilingual_357m (NeMo TTS, in-process)

llama-nemotron-llm-server  (ai-services/llm/llama_nemotron/)
    └── vllm >=0.12.0
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../../utils/xr-ai-logging]
    └── xr-ai-vllm     [editable: ../../../utils/xr-ai-vllm]
    Model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1 (vLLM).
    Native Llama-3.1 tool calling via vLLM's llama3_json parser
    (--enable-auto-tool-choice --tool-call-parser llama3_json) + per-turn
    reasoning toggle ("detailed thinking on/off") via system prompt.
    vllm_backend: pip|docker — same dispatch as the other vllm-backed services.

nemotron3-nano-llm-server  (ai-services/llm/nemotron3_nano/)
    └── vllm >=0.12.0
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../../utils/xr-ai-logging]
    └── xr-ai-vllm     [editable: ../../../utils/xr-ai-vllm]
    Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8} (vLLM)
           (auto-selected by GPU compute capability — Blackwell SM>=10
           gets NVFP4 + FP8 KV cache, otherwise FP8 weights).
    Persistent wrapper around `vllm serve`; reuses an already-running
    instance if /health answers; survives stack restarts. Qwen3-Coder
    tool-call parser + nano_v3 reasoning parser handled server-side by
    vLLM (the parser plugin is auto-fetched into model_cache on first run).
    vllm_backend: pip|docker — same dispatch as vlm-server.

nemotron-omni-llm-server  (ai-services/llm/nemotron_omni/)
    └── vllm >=0.12.0
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../../utils/xr-ai-logging]
    └── xr-ai-vllm     [editable: ../../../utils/xr-ai-vllm]
    Model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-{NVFP4,FP8,BF16} (vLLM)
    Multimodal (text + video). Non-persistent foreground wrapper; auto-selects
    quant by GPU compute capability (NVFP4 on Blackwell, FP8 on Ada/Hopper,
    BF16 via `use_bf16: true`). nemotron_v3 reasoning parser + qwen3_coder
    tool-call parser handled server-side by vLLM.
    vllm_backend: pip|docker — same dispatch as vlm-server.

piper-tts-server  (ai-services/tts/piper/)
    └── piper-tts >=1.4.0
    └── huggingface-hub >=0.22
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── pyyaml >=6.0
    Voices: rhasspy/piper-voices on HuggingFace (ONNX, auto-downloaded)
    Trade-off vs magpie: ~100 ms/sentence on CPU vs. 2-5 s; no GPU needed.
```

---

## AI inference servers

| Server | Package | Command | Default port | Model | Backend |
|---|---|---|---|---|---|
| `ai-services/vlm-server/` | `vlm-server` | `vlm_server` | 8100 | Cosmos-Reason1-7B | vLLM (pip or docker) |
| `ai-services/stt-server/` | `stt-server` | `stt_server` | 8103 | parakeet-tdt-0.6b-v3 | NeMo ASR in-process |
| `ai-services/tts/magpie/` | `magpie-tts-server` | `magpie_tts_server` | 8104 | magpie_tts_multilingual_357m | NeMo TTS in-process |
| `ai-services/tts/piper/` | `piper-tts-server` | `piper_tts_server` | 8105 | rhasspy/piper-voices (ONNX) | piper-tts in-process |
| `ai-services/llm/llama_nemotron/` | `llama-nemotron-llm-server` | `llama_nemotron_llm_server` | 8106 | Llama-3.1-Nemotron-Nano-8B-v1 | vLLM (pip or docker) |
| `ai-services/llm/nemotron3_nano/` | `nemotron3-nano-llm-server` | `nemotron3_nano_llm_server` | 8107 | NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8} (GPU-selected) | vLLM (pip or docker) |
| `ai-services/llm/nemotron_omni/` | `nemotron-omni-llm-server` | `nemotron_omni_llm_server` | 8108 | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-{NVFP4,FP8,BF16} | vLLM (pip or docker) — multimodal text+video |
| `agent-mcp-servers/transcript-mcp/` | `transcript-mcp-server` | `transcript_mcp_server` | 8200 | — | Pure FastMCP (JSONL storage) |
| `agent-mcp-servers/video-mcp/` | `video-mcp-server` | `video_mcp_server` | 8210 | — | Pure FastMCP (reads NVENC chunks from disk) |
| `agent-mcp-servers/render-mcp/` | `render-mcp-server` | `render_mcp_server` | 8220 | — | FastAPI streaming + FastMCP tools → LOVR (msgpack/ZMQ) |
| `agent-mcp-servers/oxr-mcp/` | `oxr-mcp-server` | `oxr_mcp_server` | 8230 | — | Pure FastMCP → headless OpenXR / CloudXR |
| `agent-mcp-servers/vlm-mcp/` | `vlm-mcp-server` | `vlm_mcp_server` | 8240 | — | Pure FastMCP; forwards images to vlm-server via xr-ai-models |
| `agent-mcp-servers/vec-mcp/` | `vec-mcp-server` | `vec_mcp_server` | 8250 | — | Pure FastMCP; deterministic spatial-math primitives (no model) |

All model weights are cached under `models/` at the repo root (gitignored except
`.gitkeep`).  Cache path is configured via `model_cache` in each YAML, resolved
relative to the YAML file's directory.

---

## Client samples

### Android  (client-samples/android/)

Jetpack Compose sample app mirroring the web and iOS/visionOS clients feature-for-feature.

| Layer | Language | External deps |
|---|---|---|
| StreamKit library | Kotlin | `io.livekit:livekit-android` 2.7.0 (provides `TextureViewRenderer` used by the in-SDK `CameraPreviewView` composable; no extra `livekit-android-compose-components` dep) |
| App UI | Kotlin + Jetpack Compose | Compose BOM 2024.11.00, `lifecycle-viewmodel-compose` 2.8.7, `activity-compose` 1.9.3 |

The `gradle-wrapper.jar` is not checked in (binary artifact); Android Studio generates it on first sync.

### iOS / visionOS  (client-samples/ios-visionos/)

Swift / SwiftUI + Swift Package Manager.  See `client-samples/ios-visionos/README.md`.

### Web  (client-samples/web/)

Vanilla JS. The page's import map loads `livekit-client` and
`@nvidia/cloudxr` from `client-samples/web/vendor/`, served same-origin
by the hub so headsets / offline LANs work. Both bundles are gitignored
build output of `client-samples/web-xr-build/build.sh` — every host
serving any web sample runs that script once:

  - `cloudxr-sdk.esm.mjs`   — webpack-bundled from the @nvidia/cloudxr NGC tarball
  - `livekit-client.esm.mjs` — copied from npm's prebuilt ESM

---

## Agent samples

### simple-vlm-example  (agent-samples/simple-vlm-example/)

Vision Q&A driven by voice, text, or "ping": audio → STT → query;
text → query; "ping" → default-prompt query.  Each query runs against
the latest video frame via streaming VLM and replies with both
`vlm.response` text and sentence-batched Piper TTS audio.

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `simple-vlm-example` | `xr-ai-launcher` | — |
| Worker | `simple-vlm-example-worker` | `xr-ai-agent`, `xr-ai-logging [editable]`, `xr-ai-models [editable]`, `xr-ai-pipecat [editable]` | numpy >=1.24, Pillow >=10.0, pyyaml >=6.0 (xr-ai-vad + xr-ai-voicegate + pipecat-ai + scipy + httpx + fastmcp pulled in via xr-ai-pipecat) |

Worker runs on the unified pipecat voice pipeline assembled by
`xr_ai_pipecat.make_voice_pipeline`. `SimpleVlmBrain` (a
`BrainProcessor`) owns the camera-on-demand state machine, frame
tracking, the VLM streaming call, and the data-channel side path
("ping" + ad-hoc text); voice gate (magic phrases, follow-up grace,
listening chime, stop ack) lives in `xr_ai_voicegate` inside the
`VoiceGateProcessor`. VAD/STT and sentence-batched TTS are also
provided by the pipeline so the worker only configures the knobs.

Worker calls stt-server (8103), vlm-server (8100), and piper-tts-server
(8105) over HTTP via `xr-ai-models` SDK — no model weights loaded
in-process.  Model endpoints are configured via `yaml/models.yaml`
(default: Cosmos profile) or `yaml/models.omni.yaml` (Nemotron-Omni
on port 8108). Voice-gate knobs are configured via `yaml/voice_gate.yaml`.

### model-servers  (agent-samples/model-servers/)

Standalone launcher that starts the four AI inference servers and keeps
them alive across stack restarts.  No hub, worker, or agent involved —
run this first to warm up model weights before starting any demo sample.

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `model-servers` | `xr-ai-launcher`, `xr-ai-logging`, `xr-ai-vllm` (for `--stop`) | — |

Starts stt-server (8103), nemotron3-nano-llm-server (8107, `persistent=True`),
vlm-server (8100, `persistent=True`), llama-nemotron-llm-server (8106, `persistent=True`).
The three vLLM servers survive launcher restarts; use `--stop` to shut them down.
GPU profiles: `dual_48G_ada`, `spark`, `96G_blackwell` (auto-detected).

### xr-render-demo  (agent-samples/xr-render-demo/)

Voice-driven sphere rendered into a CloudXR session: web mic → STT → LLM
action list (user-frame coords) → render-mcp → LOVR. Pose from oxr-mcp lets
the worker convert user-frame requests ("to my left") to world-frame before
forwarding.

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `xr-render-demo` | `xr-ai-launcher`, `xr-ai-logging` | loguru >=0.7 |
| Worker | `xr-render-demo-worker` | `xr-ai-agent`, `xr-ai-models` [editable], `xr-ai-pipecat` [editable], `xr-ai-voicegate` [editable], `xr-ai-logging` [editable] | fastmcp >=0.4, pyyaml >=6.0, pipecat-ai >=0.0.46 (silero-vad pulled in via xr-ai-pipecat → xr-ai-vad) |

Model endpoints (llm, agent_llm, stt, tts, vlm) are declared in
`yaml/models.yaml` and loaded via `xr-ai-models` `load_models_config` /
`make_llm` / `make_stt` / `make_tts` / `make_vlm`.  `httpx` is retained as
a transitive dep of `xr-ai-pipecat` and `fastmcp`.

Requires `model-servers` to be running first — model servers are declared as
`launch_mode="reuse"` so the launcher skips spawning them but the dependency
is explicit in the process list.
Starts: hub, cloudxr-runtime, piper-tts (8105), vlm-mcp (8240),
video-mcp (8210), render-mcp (8220), oxr-mcp (8230), vec-mcp (8250), worker.
Web client must be a build that includes the bundled CloudXR JS SDK
(see `client-samples/web-xr-build/`).

---

## Change impact map

When you change something in the left column, **all items on the right must be
updated in the same commit**.

| Component changed | Must also update |
|---|---|
| `agent-sdk/` API or types | `AGENTS.md` worker boilerplate, any sample worker that uses the changed API |
| `server-runtime/` config fields (`LiveKitConnectorConfig`) | `server-runtime/xr_media_hub.yaml` (reference copy), each sample's `xr_media_hub.yaml`, `AGENTS.md` Config section |
| `utils/xr-ai-launcher/` `Process` / `run_stack` API | `AGENTS.md` orchestrator boilerplate and process model section |
| `utils/xr-ai-vllm/` API (`serve`, `stop_persistent_servers`, `resolve_model_cache`, `load_config`, `setup_hf_env`, `gpu_compute_major`) | All four vllm wrappers (`ai-services/vlm-server/`, `ai-services/llm/llama_nemotron/`, `ai-services/llm/nemotron3_nano/`, `ai-services/llm/nemotron_omni/`), `agent-samples/xr-render-demo/main.py` (`_PERSISTENT_SERVERS`) |
| `vllm_backend` / `vllm_image` YAML keys | `ai-services/{vlm-server,llm/llama_nemotron,llm/nemotron3_nano,llm/nemotron_omni}/<server>.yaml`, every per-profile copy in `agent-samples/`, `docs/ai-services.md` |
| Container name used by a vllm wrapper | `_CONTAINER_NAME` in the wrapper's `__main__.py`, `_PERSISTENT_SERVERS` in `agent-samples/xr-render-demo/main.py` |
| vlm-server model class or supported architectures | `ai-services/vlm-server/vlm_server.yaml` comments |
| vlm-server YAML config keys (`model`, `model_cache`, …) | `ai-services/vlm-server/vlm_server.yaml`, `agent-samples/simple-vlm-example/vlm_server.yaml` |
| cloudxr-runtime YAML config keys | `agent-samples/xr-render-demo/cloudxr_runtime.yaml`, `AGENTS.md` CloudXR section |
| `utils/xr-ai-launcher/xr_ai_launcher/_cloudxr_env.py` API | render-mcp + oxr-mcp `__main__.py` imports, `AGENTS.md` cloudxr-env section |
| render-mcp YAML config keys | `agent-mcp-servers/render-mcp/render_mcp.yaml`, sample copies, worker URL constants |
| oxr-mcp YAML config keys | `agent-mcp-servers/oxr-mcp/oxr_mcp_server.yaml`, sample copies, worker URL constants |
| Any `pyproject.toml` dependency | `DEPENDENCIES.md` (this file) |
| Any new sample added | `DEPENDENCIES.md`, `AGENTS.md`, `README.md` |
| Any new shared component added (peer of `server-runtime/`) | `AGENTS.md` Architecture section, `DEPENDENCIES.md` |
| `xr-ai-models` protocols (`LLMService`, `VLMService`, …) or `models.yaml` schema | `AGENTS.md` "HTTP calls go through `xr-ai-models`" rule, `agent-sdk/xr-ai-models/README.md`, every sample's `yaml/models.yaml` |
| `xr-ai-models` preset added (new in-tree service or backend variant) | `agent-sdk/xr-ai-models/xr_ai_models/presets/__init__.py` registry, `agent-sdk/xr-ai-models/README.md` preset table |

---

## Dependency rules (enforced)

- `utils/xr-ai-launcher/` — zero runtime dependencies. Stdlib only.
- `utils/xr-ai-logging/` — only `loguru`. Used by every process via `setup_logging()`.
- `utils/xr-ai-vllm/` — zero runtime dependencies. Stdlib only. Adding deps
  here would defeat docker mode (whose point is to keep heavy vllm-side deps
  out of the wrapper's venv).
- `agent-sdk/` (`xr-ai-agent`) — only `pyzmq` + `msgpack`. No server-side packages.
- `agent-sdk/xr-ai-models/` — `xr-ai-logging` + `httpx` + `pyyaml` only. No
  vendor SDKs (no `openai`, no `anthropic`, no `litellm`). All in-tree
  backends speak OpenAI-compatible HTTP; vendor adapters arrive as new
  `kind`s in Phase B if/when needed.
- Agent workers — `xr-ai-agent` + `xr-ai-models` + task-specific libs (numpy,
  torch, etc.). Must never import from `xr-media-hub` or `xr-ai-launcher`.
- New external deps require a note here explaining why they were added.
