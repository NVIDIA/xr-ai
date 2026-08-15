<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Dependency Map

> **AGENTS: This file is mandatory to maintain.**
> Update this map in the same commit as any `pyproject.toml` change or
> internal package or service addition, removal, or rename.

---

## Python version

Every `pyproject.toml` in this repo pins `requires-python = ">=3.11,<3.13"` by
convention. PyNvVideoCodec 2.2 publishes Python 3.13 wheels, but the rest of the
repository dependency graph and test matrix have not been qualified on 3.13.
Loosen the upper bound only as a coordinated repository-wide change.

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
xr-ai-agent-runtime  (agent-sdk/xr-ai-runtime/)
    └── nemo-relay >=0.7.2,<0.8
    └── pydantic >=2.10
    └── xr-ai-tools [editable: ../xr-ai-tools]
    In-process typed ``publish`` fan-out for agents that expose ordinary
    ``Tool`` and ``AsyncTool`` instances from ``xr-ai-tools``. Agents own their
    resources, background tasks, lifecycle, and synchronization. Tool
    execution, model clients, tool loops, planning, memory, and raw media
    transport are not runtime responsibilities. Relay scopes record runtime
    publications and receiving-agent subscription callbacks.

xr-ai-hub-client  (agent-sdk/xr-ai-hub/)
    └── pyzmq >=27.0
    └── msgpack >=1.0
    Ships the canonical ``xr_ai_hub`` package.

xr-ai-voice  (agent-sdk/xr-ai-voice/)
    └── nemo-relay >=0.7.2,<0.8
    └── pydantic >=2.10
    └── xr-ai-agent-runtime [editable: ../xr-ai-runtime]
    └── xr-ai-hub-client [editable: ../xr-ai-hub]
    └── xr-ai-logging   [editable: ../../utils/xr-ai-logging]
    └── xr-ai-models    [editable: ../xr-ai-models]
    └── xr-ai-vad       [editable: ../../utils/xr-ai-vad]
    └── xr-ai-voicegate [editable: ../../utils/xr-ai-voicegate]
    └── pipecat-ai >=1.3
    └── nltk !=3.10.1       (3.10.1 rejects deps in in-project venvs)
    └── numpy >=1.24
    └── scipy >=1.11
    Native voice runtime used by simple-vlm-example. Exposes ``VoiceAgent``,
    its ``UserQuery`` / ``VoiceOutput`` / participant-lifecycle schemas,
    ``VoiceSession``, ``HubVoiceTransport``, and
    ``VadConfig``. Voice lifecycle events enter application-named topics so
    application agents own their cleanup. Pipecat, audio framing, and pipeline
    processors are implementation details. Service health gates transport
    construction, while the session touches its ready file only after the input
    transport starts its hub IPC receive loop. The
    readiness contract is split across the ``_readiness`` / ``_session``
    modules. Not a dep of xr-ai-hub-client itself — import only in workers that
    opt into the voice runtime.

xr-ai-voicegate  (utils/xr-ai-voicegate/)
    └── numpy >=1.24
    └── pyyaml >=6.0
    Pipecat-free speech-input opt-in gate. Owns the magic-phrase + follow-up
    + STOP ladder, the lazy listening chime synthesized at the TTS sample
    rate, and the participant-joined greeting hook. Workers feed STT
    transcripts via ``feed`` and register handlers — either one-at-a-time via
    ``on_*`` setters or together via ``bind(...)``. Voice runtimes consume it
    through ``VoiceGateProcessor``; applications may also load the shared YAML
    config directly.

xr-ai-models  (agent-sdk/xr-ai-models/)
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── httpx >=0.27
    └── pyyaml >=6.0
    Unified service protocols (LLMService, VLMService, STTService, TTSService,
    EmbeddingService)
    and OpenAI-compatible HTTP clients that cover every in-tree model backend
    (vLLM-served VLM/LLMs, NeMo Parakeet STT, Piper/Magpie TTS).  Per-model
    profiles separate adapter behavior, endpoint connectivity/readiness, and
    launcher-facing deployment ownership. Relay may pass controlled per-call
    context headers; configured model credentials remain non-overridable.
    Per-model quirks remain behind one
    seam: reasoning-field aliasing (nano_v3 →
    `reasoning`, nemotron_v3 → `reasoning_content`), `chat_template_kwargs`
    plumbing for `enable_thinking` / `thinking_budget`, and built-in presets
    for the in-tree services, including distinct Cosmos3 Nano Reasoner and
    Cosmos-Reason1 VLM profiles. Workers depend on this instead of rolling their
    own httpx wrappers. Profiles may separate adapter, endpoint, and deployment
    metadata while the existing flat YAML schema remains valid.

xr-ai-tools  (agent-sdk/xr-ai-tools/)
    └── nemo-relay >=0.7.2,<0.8
    └── pydantic >=2.10
    └── [relay] xr-ai-models [editable: ../xr-ai-models]
    ├── [frames] numpy >=1.24, Pillow >=10.0, xr-ai-hub-client [editable: ../xr-ai-hub]
    ├── [image-editing] Pillow >=10.0
    ├── [vision] xr-ai-models [editable: ../xr-ai-models]
    ├── [qr-code] numpy >=1.24, zxing-cpp >=2.3,<4, Pillow >=10.0, xr-ai-hub-client [editable: ../xr-ai-hub]
    └── [services] msgpack >=1.0, pyzmq >=27.0
    Toolkit-independent native tools: Pydantic request and response models,
    Relay-managed finite and async execution, model tool-call workflow helpers,
    frame selection, lossless polygon image editing, single/multi-image
    inference, participant-scoped QR-code extraction, typed capability clients,
    and service RPC.


xr-openxr-service  (services/openxr-service/)
    └── xr-ai-launcher [editable: ../../utils/xr-ai-launcher]
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-tools[services] [editable: ../../agent-sdk/xr-ai-tools]
    └── pyyaml >=6.0
    └── isaacteleop
    Owns the long-running headless OpenXR and DeviceIO sessions. Exposes plain
    dict head-pose and health messages over private msgpack/ZMQ at port 8330;
    xr-ai-tools owns the typed client contracts. Root pytest adds this source tree
    to its Python path only for CPU-only pose-math regression tests, avoiding a
    test-time isaacteleop installation.

xr-rag-service  (services/rag-service/)
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── xr-ai-models [editable: ../../agent-sdk/xr-ai-models]
    └── xr-ai-tools[services] [editable: ../../agent-sdk/xr-ai-tools]
    └── numpy >=1.24
    └── pyyaml >=6.0
    Recursively indexes Markdown and text documents, caches dense embeddings
    by content and index settings, and exposes private msgpack/ZMQ retrieval at
    port 8340. Applications consume the native ``RAGTools`` group.

xr-ai-launcher  (utils/xr-ai-launcher/)
    └── (stdlib only — zero runtime deps)
    `_cloudxr_env` owns the shared CloudXR env helpers (stdlib-only, os + re):
    `load_cloudxr_env`, plus the single source of truth for native device
    profiles: `NATIVE_DEVICE_PROFILES`, `is_native_profile(profile)`, and
    `read_device_profile(yaml_path)` (env-first NV_DEVICE_PROFILE read, regex
    YAML fallback). `load_model_deployment()` reads the selected wrapped JSON
    model profile using only stdlib to derive managed/reused services and
    required credential names, without adding a YAML or model-SDK dependency.

xr-ai-logging  (utils/xr-ai-logging/)
    └── loguru >=0.7

xr-ai-vllm  (utils/xr-ai-vllm/)
    └── (stdlib only — zero runtime deps)
    Pluggable vLLM hosting for vLLM-backed services. Dispatches to
    either pip-installed `vllm serve` or `docker run nvcr.io/nvidia/vllm:<tag>`
    based on each YAML's `vllm_backend:` key.  Stays stdlib-only so docker mode
    does not pull vllm/torch/etc. into the wrapper's venv just to manage a
    container. Imported by the vLLM wrappers and by the orchestrator
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

xr-media-hub  (services/xr-media-hub/)
    └── xr-ai-hub-client  [editable: ../../agent-sdk/xr-ai-hub]
    └── xr-ai-logging     [editable: ../../utils/xr-ai-logging]
    └── pyzmq >=27.0
    └── livekit >=1.0
    └── livekit-api >=1.0
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── httpx >=0.27
    └── websockets >=12.0
    └── numpy >=1.24
    └── pyyaml >=6.0
    └── cryptography >=42.0
    PyNvVideoCodec >=2.2 (NVENC H.264 encoder; used when video_recording.enabled: true)

xr-video-memory-service  (services/video-memory-service/)
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── xr-ai-tools[services] [editable: ../../agent-sdk/xr-ai-tools]
    └── PyNvVideoCodec >=2.2
    └── Pillow >=10.0
    └── numpy >=1.24
    └── pyyaml >=6.0
    Owns recorded H.264 chunk queries. Exposes typed msgpack/ZMQ at port 8310
    and performs historical decoding via NVDEC; it does not subscribe to hub IPC.

cloudxr-runtime  (services/cloudxr-runtime/)
    └── isaacteleop[cloudxr]
    └── pyyaml
    └── xr-ai-launcher  [editable: ../../utils/xr-ai-launcher] (is_native_profile + read_device_profile)
    └── xr-ai-logging   [editable: ../../utils/xr-ai-logging]

xr-render-scene  (agent-samples/xr-render-demo/scene/)
    └── xr-ai-launcher [editable: ../../../utils/xr-ai-launcher]
    └── xr-ai-logging [editable: ../../../utils/xr-ai-logging]
    └── xr-ai-tools[services] [editable: ../../../agent-sdk/xr-ai-tools]
    └── pyzmq >=27.0
    └── msgpack >=1.0
    └── pyyaml >=6.0
    Owns scene state, sample-local native tools, LOVR lifecycle, and the
    LOVR Lua app. Exposes typed msgpack/ZMQ at port 8320.

xr-ai-tests  (tests/)
    └── xr-ai-agent-runtime       [editable: ../agent-sdk/xr-ai-runtime]
    └── xr-ai-hub-client             [editable: ../agent-sdk/xr-ai-hub]
    └── xr-ai-models            [editable: ../agent-sdk/xr-ai-models]
    └── xr-ai-tools[frames,image-editing,qr-code,services,vision] [editable: ../agent-sdk/xr-ai-tools]
    └── xr-rag-service [editable: ../services/rag-service]
    └── xr-video-memory-service [editable: ../services/video-memory-service]
    └── xr-ai-voice             [editable: ../agent-sdk/xr-ai-voice]
    └── xr-media-hub            [editable: ../services/xr-media-hub]    (pulls in livekit, livekit-api for the wss /rtc proxy + room-client tests)
    └── xr-ai-launcher          [editable: ../utils/xr-ai-launcher]
    └── xr-ai-logging           [editable: ../utils/xr-ai-logging]
    └── xr-ai-vad               [editable: ../utils/xr-ai-vad]
    └── xr-ai-voicegate         [editable: ../utils/xr-ai-voicegate]
    └── xr-ai-vllm              [editable: ../utils/xr-ai-vllm]
    └── xr-render-scene         [editable: ../agent-samples/xr-render-demo/scene]
    └── pytest >=8.0
    └── pytest-asyncio >=0.23
    └── numpy >=1.24
    └── Pillow >=10.0   (CPU native-vision and GPU image tests)
    └── python-multipart >=0.0.9   (STT endpoint tests import stt-server via pythonpath)
    └── pyyaml >=6.0    (CPU subprocess/service configs and GPU service tests)
    The unmarked suite is multi-client / multi-agent integration tests over
    the IPC layer, driven via ZMQ `ipc://` only — no Docker / LiveKit /
    NVENC required. Also covers unit tests for the leaf util packages
    (launcher, logging, vllm), native spatial-math, text-memory, vision, QR-code,
    and typed service-tool tests, plus the sample-local scene native tools (LOVR
    is stubbed). Root pytest adds
    services/stt-server to its Python path (not a dependency) so the
    endpoint tests can import its FastAPI app with a mocked backend,
    avoiding a test-time NeMo installation.

    Tests marked `@pytest.mark.gpu` are the local-only set (skipped by
    `-m "not gpu"` in CI). They spawn real model services via `uv run` (e.g.
    `test_gpu_stt_server.py`), import
    `livekit.rtc` directly to drive `_room_client.py`, exercise NVENC /
    NVDEC via PyNvVideoCodec, and shell out to `docker` to manage a
    LiveKit container — `livekit`, `livekit-api`, `PyNvVideoCodec`, and
    `docker` all come in transitively via the media hub and video-memory
    service rather than being redeclared here.

vlm-server  (services/vlm-server/)
    └── vllm >=0.23.0
    └── pyyaml >=6.0
    └── hf-transfer >=0.1.4
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-vllm     [editable: ../../utils/xr-ai-vllm]
    Default model: the text-output Reasoner from nvidia/Cosmos3-Nano. Runtime
    selection details are documented in
    docs/source/components/ai-services.md#per-server-notes.
    nvidia/Cosmos-Reason1-7B remains configurable with the cosmos_vlm preset.
    Wrapper Popens `vllm serve` so the launcher's killpg() does not reach
    vLLM — model survives stack restarts.
    vllm_backend: pip|docker — pip path uses the wrapper's vllm; docker path
    runs `nvcr.io/nvidia/vllm:<tag> vllm serve …` instead.

embedding-server  (services/embedding-server/)
    └── vllm >=0.14.0
    └── pyyaml >=6.0
    └── hf-transfer >=0.1.4
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── xr-ai-vllm [editable: ../../utils/xr-ai-vllm]
    Model: nvidia/llama-nemotron-embed-1b-v2. Exposes OpenAI-compatible
    embeddings at port 8109 through the shared vLLM hosting wrapper.

stt-server  (services/stt-server/)
    └── nemo_toolkit[asr] >=2.5
    └── lightning >2.2.1,<=2.4.0    # routed to github.com/Lightning-AI/pytorch-lightning
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── python-multipart >=0.0.9
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    Model: nvidia/parakeet-tdt-0.6b-v3 (NeMo ASR, in-process)

magpie-tts-server  (services/magpie-tts/)
    └── nemo_toolkit[tts] >=2.5
    └── lightning >2.2.1,<=2.4.0    # routed to github.com/Lightning-AI/pytorch-lightning
    └── soundfile >=0.12
    └── numpy >=1.24
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    Model: nvidia/magpie_tts_multilingual_357m (NeMo TTS, in-process)

llama-nemotron-llm-server  (services/llama-nemotron-llm/)
    └── vllm >=0.12.0
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-vllm     [editable: ../../utils/xr-ai-vllm]
    Model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1 (vLLM).
    Native Llama-3.1 tool calling via vLLM's llama3_json parser
    (--enable-auto-tool-choice --tool-call-parser llama3_json) + per-turn
    reasoning toggle ("detailed thinking on/off") via system prompt.
    vllm_backend: pip|docker — same dispatch as the other vllm-backed services.

nemotron3-nano-llm-server  (services/nemotron3-nano-llm/)
    └── vllm >=0.12.0
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-vllm     [editable: ../../utils/xr-ai-vllm]
    Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8} (vLLM)
           (auto-selected by GPU compute capability — Blackwell SM>=10
           gets NVFP4 + FP8 KV cache, otherwise FP8 weights).
    Persistent wrapper around `vllm serve`; reuses an already-running
    instance if /health answers; survives stack restarts. Qwen3-Coder
    tool-call parser + nano_v3 reasoning parser handled server-side by
    vLLM (the parser plugin is auto-fetched into model_cache on first run).
    vllm_backend: pip|docker — same dispatch as vlm-server.

nemotron-omni-llm-server  (services/nemotron-omni-llm/)
    └── vllm >=0.12.0
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-vllm     [editable: ../../utils/xr-ai-vllm]
    Model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-{NVFP4,FP8,BF16} (vLLM)
    Multimodal (text + video). Non-persistent foreground wrapper; auto-selects
    quant by GPU compute capability (NVFP4 on Blackwell, FP8 on Ada/Hopper,
    BF16 via `use_bf16: true`). nemotron_v3 reasoning parser + qwen3_coder
    tool-call parser handled server-side by vLLM.
    vllm_backend: pip|docker — same dispatch as vlm-server.

piper-tts-server  (services/piper-tts/)
    └── piper-tts >=1.4.0
    └── huggingface-hub >=0.22
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    Voices: rhasspy/piper-voices on HuggingFace (ONNX, auto-downloaded)
    Trade-off vs magpie: ~100 ms/sentence on CPU vs. 2-5 s; no GPU needed.
```

---

## AI inference servers

| Server | Package | Command | Default port | Model | Backend |
|---|---|---|---|---|---|
| `services/vlm-server/` | `vlm-server` | `vlm_server` | 8100 | Cosmos3 Nano Reasoner | vLLM (pip or docker) |
| `services/stt-server/` | `stt-server` | `stt_server` | 8103 | parakeet-tdt-0.6b-v3 | NeMo ASR in-process |
| `services/magpie-tts/` | `magpie-tts-server` | `magpie_tts_server` | 8104 | magpie_tts_multilingual_357m | NeMo TTS in-process |
| `services/piper-tts/` | `piper-tts-server` | `piper_tts_server` | 8105 | rhasspy/piper-voices (ONNX) | piper-tts in-process |
| `services/llama-nemotron-llm/` | `llama-nemotron-llm-server` | `llama_nemotron_llm_server` | 8106 | Llama-3.1-Nemotron-Nano-8B-v1 | vLLM (pip or docker) |
| `services/nemotron3-nano-llm/` | `nemotron3-nano-llm-server` | `nemotron3_nano_llm_server` | 8107 | NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8} (GPU-selected) | vLLM (pip or docker) |
| `services/nemotron-omni-llm/` | `nemotron-omni-llm-server` | `nemotron_omni_llm_server` | 8108 | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-{NVFP4,FP8,BF16} | vLLM (pip or docker) — multimodal text+video |
| `services/embedding-server/` | `embedding-server` | `embedding_server` | 8109 | llama-nemotron-embed-1b-v2 | vLLM (pip or docker) |
| `services/video-memory-service/` | `xr-video-memory-service` | `video_memory_service` | 8310 | — | Typed msgpack/ZMQ → recorded H.264 queries |
| `agent-samples/xr-render-demo/scene/` | `xr-render-scene` | `xr_render_scene` | 8320 | — | Sample-local typed scene service → LOVR |
| `services/openxr-service/` | `xr-openxr-service` | `openxr_service` | 8330 | — | Typed msgpack/ZMQ → headless OpenXR / CloudXR |
| `services/rag-service/` | `xr-rag-service` | `rag_service` | 8340 | — | Typed msgpack/ZMQ → dense document retrieval |

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

| Layer | Language | External deps |
|---|---|---|
| `StreamKit` library | Swift | `livekit/client-sdk-swift` 2.13.0 (LiveKit WebRTC) |
| App target | Swift / SwiftUI | `livekit/client-sdk-swift` 2.13.0 (transitively via StreamKit), `NVIDIA/cloudxr-framework` 6.1.0 (CloudXRKit + NVIDIAStreamKit + NVTelemetry xcframeworks) |

Required entitlement on visionOS: `com.apple.developer.low-latency-streaming` (Apple Developer Program).

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
| Worker | `simple-vlm-example-worker` | `xr-ai-agent-runtime [editable]`, `xr-ai-hub-client [editable]`, `xr-ai-logging [editable]`, `xr-ai-models [editable]`, `xr-ai-tools[frames,vision] [editable]`, `xr-ai-voice [editable]`, `xr-ai-voicegate [editable]` | nemo-relay >=0.7.2,<0.8, loguru >=0.7, pyyaml >=6.0 (`xr-ai-voice` pulls in VAD, pipecat-ai, numpy, and scipy; `xr-ai-tools[frames]` pulls in numpy and Pillow) |

The packaged worker uses `CurrentFrameTool` for frame acquisition and passes its
opaque `ImageReference` to a transport-independent `StreamingImageQueryTool`
inside `SimpleVlmAgent`. Single-image, ordered multi-image, and timestamped
frame-sequence inference share the same list-based VLM path. Camera bytes stay
in a bounded in-process registry and image locations are redacted from VLM
telemetry while the provider receives the original frames.
`VoiceAgent` owns `VoiceSession`, readiness, hub transport, signals, and the
private Pipecat pipeline; it routes `"ping"` and ad-hoc text through the same
sample-named `UserQuery` topic as speech and publishes lifecycle events on
sample-named topics. `SimpleVlmAgent` handles cancellation and frame cleanup
inside its own subscriber methods. Voice-gate
behavior (magic phrases, follow-up grace, listening chime, stop acknowledgement),
VAD/STT, and sentence-batched TTS remain provided by the shared voice runtime.
The sample has no MCP or legacy agent-framework dependency.

Worker calls stt-server (8103), vlm-server (8100), and piper-tts-server
(8105) over HTTP via `xr-ai-models` SDK — no model weights loaded
in-process. The `models_config` key selects a structured deployment profile:
`models.local.json` manages the default Cosmos3 Nano Reasoner service,
`models.hosted.json` uses the hosted Cosmos3 Nano Reasoner NIM, and
`models.omni.json` reuses Nemotron-Omni on port 8108. These profiles separate
adapter behavior, endpoint readiness and
credentials, and launcher ownership. Voice-gate knobs are configured via
`yaml/voice_gate.yaml`.

### model-servers  (agent-samples/model-servers/)

Standalone launcher that starts the shared AI inference servers and keeps
them alive across stack restarts.  No hub, worker, or agent involved —
run this first to warm up model weights before starting any demo sample.

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `model-servers` | `xr-ai-launcher`, `xr-ai-logging`, `xr-ai-vllm` (for `--stop`) | — |

The default `--omni-stack` starts stt-server (8103),
nemotron-omni-llm-server (8108, `persistent=True`), and embedding-server
(8109, `persistent=True`). `--vlm-llm-stack` selects the legacy
nemotron3-nano-llm-server (8107) and vlm-server (8100) pair while retaining
stt-server and embedding-server. The stacks
are mutually exclusive; `--stop` shuts down every model-server port without
selecting one.
GPU profiles: `dual_48G_ada`, `spark`, `96G_blackwell` (auto-detected).

### xr-render-demo  (agent-samples/xr-render-demo/)

Voice-driven sphere rendered into a CloudXR session: web mic → STT → LLM
tool calls → Relay-managed native tools → typed scene/OpenXR/video services → LOVR.
The worker composes tracking and spatial-math tools in process for
user-relative requests such as "to my left".

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `xr-render-demo` | `xr-ai-launcher`, `xr-ai-logging` | loguru >=0.7 |
| Scene | `xr-render-scene` | `xr-ai-launcher`, `xr-ai-logging`, `xr-ai-tools[services]` | pyzmq >=27.0, msgpack >=1.0, pyyaml >=6.0 |
| Worker | `xr-render-demo-worker` | `xr-ai-agent-runtime` [editable], `xr-ai-hub-client`, `xr-ai-models` [editable], `xr-ai-tools[frames,services,vision]` [editable], `xr-ai-voice` [editable], `xr-ai-voicegate` [editable], `xr-ai-logging` [editable], `xr-render-scene` [editable] | pydantic >=2.12, pyyaml >=6.0 (native scene, tracking, spatial-math, video-memory, vision, and text-memory tools replace capability MCP clients; `xr-ai-voice` privately supplies VAD and speech-pipeline dependencies). |

Model endpoints (llm, agent_llm, stt, tts, vlm) are declared in
`yaml/models.yaml` and loaded via `xr-ai-models` `load_models_config` /
`make_llm` / `make_stt` / `make_tts` / `make_vlm`.  `httpx` is retained as
a transitive dep of `xr-ai-voice` and `xr-ai-tools[frames]`.

Requires `model-servers` to be running first — the Omni model server is
declared as `launch_mode="reuse"` so the launcher skips spawning it but the
dependency is explicit in the process list.
Starts: hub, cloudxr-runtime, piper-tts (8105), video-memory (8310),
scene (8320), openxr-service (8330), and worker. The model-server entry is
declared with `launch_mode="reuse"` and must already be healthy.
No MCP adapters run in the sample stack.
Web client must be a build that includes the bundled CloudXR JS SDK
(see `client-samples/web-xr-build/`).

---

## Change impact map

Keep non-obvious fan-out in the same change:

| Component changed | Also update |
|---|---|
| `agent-sdk/xr-ai-hub/` API or IPC types | [Agent SDK](docs/source/components/agent-sdk.md), [hub reference](docs/source/reference/agent-sdk-hub.md), and affected sample workers |
| `services/xr-media-hub/` configuration | Its reference YAML and every sample `xr_media_hub.yaml` |
| `utils/xr-ai-launcher/` process API | [Process model](docs/source/components/launcher-and-process-model.md) and sample orchestrators |
| `utils/xr-ai-vllm/` API or `vllm_backend` / `vllm_image` keys | Every vLLM service wrapper and YAML, every per-profile sample copy, and [AI services](docs/source/components/ai-services.md) |
| Model-service package, command, port, or container name | `services/README.md`, model-server orchestration and cleanup, this map, and [AI services](docs/source/components/ai-services.md) |
| vlm-server model or configuration | Its reference YAML, every model-server profile, model presets, and [AI services](docs/source/components/ai-services.md) |
| CloudXR configuration or native-profile helpers | `agent-samples/xr-render-demo/yaml/cloudxr_runtime.yaml`, its orchestrator, [Adding CloudXR](docs/source/guides/adding-cloudxr.md), and [xr-render reference](docs/source/reference/xr-render-demo.md) |
| Scene-service configuration | Scene YAML, xr-render orchestrator, and [xr-render reference](docs/source/reference/xr-render-demo.md) |
| Any `pyproject.toml` dependency | This dependency map and a local lock regeneration |
| New sample or reusable service | Root and local READMEs, this map, and the relevant Sphinx guide |
| `xr-ai-models` protocol, profile schema, or preset | Model package reference, preset registry, sample profiles, and architecture rules |

## Dependency rules (enforced)

- `utils/xr-ai-launcher/` — zero runtime dependencies. Stdlib only.
- `utils/xr-ai-logging/` — only `loguru`. Used by every process via `setup_logging()`.
- `utils/xr-ai-vllm/` — zero runtime dependencies. Stdlib only. Adding deps
  here would defeat docker mode (whose point is to keep heavy vllm-side deps
  out of the wrapper's venv).
- `agent-sdk/xr-ai-hub/` (`xr-ai-hub-client`) — only `pyzmq` + `msgpack`. No server-side packages.
- `agent-sdk/xr-ai-models/` — `xr-ai-logging` + `httpx` + `pyyaml` only. No
  vendor SDKs (no `openai`, no `anthropic`, no `litellm`). All in-tree backends speak OpenAI-compatible HTTP.
- `agent-sdk/xr-ai-tools/` — native tool contracts and only capability-specific
  optional dependencies. Spatial math remains CPU-only.
- Agent workers — `xr-ai-hub-client` + `xr-ai-models` + task-specific libs (numpy,
  torch, etc.). Must never import from `xr-media-hub` or `xr-ai-launcher`.
- New external deps require a note here explaining why they were added.
