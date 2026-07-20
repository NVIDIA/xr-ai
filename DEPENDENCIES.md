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
`xr-media-hub` and `video-memory-service` for NVENC encode / NVDEC decode) does not
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
    └── pipecat-ai >=1.3
    └── numpy >=1.24
    └── scipy >=1.11
    └── httpx >=0.27
    └── fastmcp >=2.0
    Unified Pipecat voice pipeline. Owns the transport bridge to
    ProcessorEndpoint (ZMQ IPC) plus the library FrameProcessors —
    VadSttProcessor, VoiceGateProcessor, BrainProcessor, StreamingTtsProcessor —
    composed by ``make_voice_pipeline``. Resamples hub float32 audio →
    16 kHz int16 for STT, converts TTS int16 PCM back to float32 AudioChunks
    for return. SttClient / TtsClient are thin wrappers around xr-ai-models'
    OpenAICompatSTT / OpenAICompatTTS; httpx is retained for http_probe()
    readiness checks.
    Not a dep of xr-ai-agent itself — import only in workers that use Pipecat.

xr-ai-capabilities  (agent-sdk/xr-ai-capabilities/)
    └── xr-ai-agent   [editable: ..]
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── xr-ai-models  [editable: ../xr-ai-models]
    └── numpy >=1.24
    └── Pillow >=10.0
    Framework-agnostic, reusable agent capabilities (see AGENTS.md → "Agent
    sample architecture"). A capability talks to the hub through a
    ``ProcessorEndpoint`` and depends only on the core SDK — NOT on pipecat or
    any pipeline framework — so both pipecat and non-pipecat agents can compose
    it. Hosts ``pixels`` (frame → PIL → JPEG; numpy + Pillow) and ``vision``
    (VisionModule — live-camera VLM Q&A with camera-on-demand, exposing ``ask``
    for streaming TTS and ``perceive`` for agentic tool loops), so vision
    samples no longer copy that code per-worker. A pipecat brain wires it up by
    passing ``transport.endpoint``.

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

xr-ai-nat  (agent-sdk/xr-ai-nat/)
    └── nvidia-nat-core ==1.8.0
    └── pydantic >=2.10
    └── [mcp] fastmcp >=3.4,<4
    └── [services] msgpack >=1.0, pyzmq >=27.0
    └── [vision] httpx >=0.27, Pillow >=10.0, xr-ai-models [editable: ../xr-ai-models]
    Typed, in-process NeMo Agent Toolkit functions for XR capabilities. The
    ``xr_spatial_math`` function group accepts explicit coordinate frames and
    performs deterministic spatial calculations without OpenXR, model, or MCP
    dependencies. ``xr_text_memory`` owns persistent per-source JSONL text
    history. Its optional MCP adapter exports an application's explicit native
    function list without routing native composition through MCP. Each
    capability module is its own ``nat.plugins`` discovery entry point; there
    is no package-wide registration aggregator. The spatial pure math core is
    also used by the transitional Vec and OpenXR MCP compatibility surfaces.
    ``xr_vision`` normalizes an acquired local image and calls an injected
    xr-ai-models VLM; it does not acquire frames itself. ``xr_tracking`` calls
    the typed OpenXR service and returns a complete user coordinate frame.
    ``xr_video_memory`` calls the typed video-memory service for recorded-video
    discovery, queries, and frame extraction. Live frames stay with the hub
    client owned by their caller.

xr-openxr-service  (services/openxr-service/)
    └── xr-ai-launcher [editable: ../../utils/xr-ai-launcher]
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-nat[services] [editable: ../../agent-sdk/xr-ai-nat]
    └── pyyaml >=6.0
    └── isaacteleop
    Owns the long-running headless OpenXR and DeviceIO sessions. Exposes plain
    dict head-pose and health messages over private msgpack/ZMQ at port 8330;
    xr-ai-nat owns the typed client contracts. Root pytest adds this source tree
    to its Python path only for CPU-only pose-math regression tests, avoiding a
    test-time isaacteleop installation.

xr-ai-launcher  (utils/xr-ai-launcher/)
    └── (stdlib only — zero runtime deps)
    `_cloudxr_env` owns the shared CloudXR env helpers (stdlib-only, os + re):
    `load_cloudxr_env`, plus the single source of truth for native device
    profiles: `NATIVE_DEVICE_PROFILES`, `is_native_profile(profile)`, and
    `read_device_profile(yaml_path)` (env-first NV_DEVICE_PROFILE read, regex
    YAML fallback).

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
    └── livekit >=1.0
    └── livekit-api >=1.0
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
    └── pyyaml >=6.0
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── xr-ai-nat[mcp] [editable: ../../agent-sdk/xr-ai-nat]
    Thin MCP compatibility process at /mcp (no REST). It republishes the four
    native ``xr_text_memory`` functions under their existing MCP tool names;
    JSONL storage and source identity handling live in xr-ai-nat.

vlm-mcp-server  (agent-mcp-servers/vlm-mcp/)
    └── uvicorn[standard] >=0.29
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-models   [editable: ../../agent-sdk/xr-ai-models]
    └── xr-ai-nat[mcp,vision] [editable: ../../agent-sdk/xr-ai-nat]
    Thin MCP compatibility process with one tool at /mcp (no REST). It
    republishes ``vision__ask_image`` under the existing ``ask_image`` name.
    Image normalization and the VLM call live in the native vision function;
    the legacy ``vlm_server:`` URL key remains accepted with a warning.

video-mcp-server  (agent-mcp-servers/video-mcp/)
    └── uvicorn[standard] >=0.29
    └── fastmcp >=2.0
    └── pyyaml >=6.0
    └── numpy >=1.24
    └── Pillow >=10.0
    └── xr-ai-agent [editable: ../../agent-sdk]
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── xr-ai-nat[services] [editable: ../../agent-sdk/xr-ai-nat]
    Pure FastMCP compatibility adapter at /mcp. Preserves the conditional
    legacy tool list; recorded operations delegate to video-memory-service and
    live compatibility operations acquire raw hub frames locally.

xr-video-memory-service  (services/video-memory-service/)
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── xr-ai-nat[services] [editable: ../../agent-sdk/xr-ai-nat]
    └── PyNvVideoCodec >=1.0
    └── Pillow >=10.0
    └── numpy >=1.24
    └── pyyaml >=6.0
    Owns recorded H.264 chunk queries. Exposes typed msgpack/ZMQ at port 8310
    and performs historical decoding via NVDEC; it does not subscribe to hub IPC.

cloudxr-runtime  (cloudxr-runtime/)
    └── isaacteleop[cloudxr]
    └── pyyaml
    └── xr-ai-launcher  [editable: ../utils/xr-ai-launcher] (is_native_profile + read_device_profile)
    └── xr-ai-logging   [editable: ../utils/xr-ai-logging]

render-mcp-server  (agent-mcp-servers/render-mcp/)
    └── xr-ai-logging [editable: ../../utils/xr-ai-logging]
    └── xr-render-scene [editable: ../../agent-samples/xr-render-demo/scene]
    └── pyyaml >=6.0
    └── uvicorn[standard] >=0.29
    └── fastmcp >=2.0
    FastMCP compatibility adapter at /mcp. Preserves the legacy render tool
    surface while delegating to the sample-local typed scene process.

xr-render-scene  (agent-samples/xr-render-demo/scene/)
    └── xr-ai-launcher [editable: ../../../utils/xr-ai-launcher]
    └── xr-ai-logging [editable: ../../../utils/xr-ai-logging]
    └── xr-ai-nat[services] [editable: ../../../agent-sdk/xr-ai-nat]
    └── pyzmq >=27.0
    └── msgpack >=1.0
    └── pyyaml >=6.0
    Owns scene state, sample-local NAT function groups, LOVR lifecycle, and the
    LOVR Lua app. Exposes typed msgpack/ZMQ at port 8320.

oxr-mcp-server  (agent-mcp-servers/oxr-mcp/)
    └── pyyaml >=6.0
    └── uvicorn[standard] >=0.29
    └── fastmcp >=2.0
    └── xr-ai-nat[services] [editable: ../../agent-sdk/xr-ai-nat]
    Pure FastMCP at /mcp. Preserves the existing OXR tool schemas while
    delegating pose acquisition to xr-openxr-service and coordinate operations
    to the native spatial-math core.

vec-mcp-server  (agent-mcp-servers/vec-mcp/)
    └── uvicorn[standard] >=0.29
    └── fastmcp >=2.0
    └── pyyaml >=6.0
    └── xr-ai-logging  [editable: ../../utils/xr-ai-logging]
    └── xr-ai-nat      [editable: ../../agent-sdk/xr-ai-nat]
    Pure FastMCP at /mcp. Deterministic spatial-math primitives
    (between_anchors, world_offset, along_direction, scale_value).
    Preserves the existing MCP tool names while delegating coordinate
    calculations to the shared native spatial-math core. ``scale_value``
    remains compatibility-only and is not part of the native function group.

xr-ai-tests  (tests/)
    └── xr-ai-agent             [editable: ../agent-sdk]
    └── xr-ai-capabilities      [editable: ../agent-sdk/xr-ai-capabilities]
    └── xr-ai-models            [editable: ../agent-sdk/xr-ai-models]
    └── xr-ai-nat               [editable: ../agent-sdk/xr-ai-nat]
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
    └── xr-render-scene         [editable: ../agent-samples/xr-render-demo/scene]
    └── video-mcp-server        [editable: ../agent-mcp-servers/video-mcp]
    └── vec-mcp-server          [editable: ../agent-mcp-servers/vec-mcp]
    └── pytest >=8.0
    └── pytest-asyncio >=0.23
    └── numpy >=1.24
    └── fastmcp >=3.4,<4 (MCP adapter contracts and tests marked `gpu`)
    └── Pillow >=10.0   (only used by tests marked `gpu`)
    └── pyyaml >=6.0    (only used by tests marked `gpu`)
    The unmarked suite is multi-client / multi-agent integration tests over
    the IPC layer, driven via ZMQ `ipc://` only — no Docker / LiveKit /
    NVENC required. Also covers unit tests for the leaf util packages
    (launcher, logging, vllm), a CI-viable subprocess test for
    CPU-viable subprocess smoke tests for transcript-mcp-server and
    vec-mcp-server (fastmcp pulled in transitively), native spatial-math and
    text-memory and vision function-group tests, generic NAT-to-MCP adapter
    tests, the vlm-mcp adapter, and the sample-local scene native groups plus
    render-mcp adapter surface (LOVR is stubbed). oxr-mcp is not
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
| `services/video-memory-service/` | `xr-video-memory-service` | `video_memory_service` | 8310 | — | Typed msgpack/ZMQ → recorded H.264 queries |
| `agent-mcp-servers/video-mcp/` | `video-mcp-server` | `video_mcp_server` | 8210 | — | FastMCP compatibility adapter → recorded service + live hub IPC |
| `agent-samples/xr-render-demo/scene/` | `xr-render-scene` | `xr_render_scene` | 8320 | — | Sample-local typed scene service → LOVR |
| `agent-mcp-servers/render-mcp/` | `render-mcp` | `render_mcp` | 8220 | — | FastMCP compatibility adapter → xr-render-scene |
| `services/openxr-service/` | `xr-openxr-service` | `openxr_service` | 8330 | — | Typed msgpack/ZMQ → headless OpenXR / CloudXR |
| `agent-mcp-servers/oxr-mcp/` | `oxr-mcp-server` | `oxr_mcp_server` | 8230 | — | FastMCP compatibility adapter → openxr-service |
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
| Worker | `simple-vlm-example-worker` | `xr-ai-agent`, `xr-ai-capabilities [editable]`, `xr-ai-logging [editable]`, `xr-ai-models [editable]`, `xr-ai-pipecat [editable]` | pyyaml >=6.0 (VisionModule + pixels now come from xr-ai-capabilities, which pulls numpy + Pillow; xr-ai-vad + xr-ai-voicegate + pipecat-ai + scipy + httpx + fastmcp pulled in via xr-ai-pipecat) |

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
action list (user-frame coords) → render-mcp → typed scene process → LOVR. Pose from oxr-mcp lets
the worker convert user-frame requests ("to my left") to world-frame before
forwarding.

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `xr-render-demo` | `xr-ai-launcher`, `xr-ai-logging` | loguru >=0.7 |
| Scene | `xr-render-scene` | `xr-ai-launcher`, `xr-ai-logging`, `xr-ai-nat` | pyzmq >=27.0, msgpack >=1.0, pyyaml >=6.0 |
| Worker | `xr-render-demo-worker` | `xr-ai-agent`, `xr-ai-capabilities` [editable], `xr-ai-models` [editable], `xr-ai-pipecat` [editable], `xr-ai-voicegate` [editable], `xr-ai-logging` [editable] | fastmcp >=2.0, pyyaml >=6.0, pipecat-ai >=1.3 (numpy + Pillow pulled in via xr-ai-capabilities; silero-vad via xr-ai-pipecat → xr-ai-vad). The `look_at_current_frame` perception tool reuses `xr_ai_capabilities.VisionModule` (live-frame VLM Q&A); the worker-local `pixels.py` and its frame/camera helpers were removed. |

Model endpoints (llm, agent_llm, stt, tts, vlm) are declared in
`yaml/models.yaml` and loaded via `xr-ai-models` `load_models_config` /
`make_llm` / `make_stt` / `make_tts` / `make_vlm`.  `httpx` is retained as
a transitive dep of `xr-ai-pipecat` and `fastmcp`.

Requires `model-servers` to be running first — model servers are declared as
`launch_mode="reuse"` so the launcher skips spawning them but the dependency
is explicit in the process list.
Starts: hub, cloudxr-runtime, piper-tts (8105), vlm-mcp (8240),
video-memory (8310), video-mcp (8210), scene (8320), render-mcp (8220),
openxr-service (8330), oxr-mcp (8230), vec-mcp (8250), worker.
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
| cloudxr-runtime YAML config keys | `agent-samples/xr-render-demo/yaml/cloudxr_runtime.yaml`, `docs/adding-cloudxr.md` |
| `utils/xr-ai-launcher/xr_ai_launcher/_cloudxr_env.py` API | xr-render-scene + oxr-mcp + cloudxr-runtime `__main__.py` imports, `agent-samples/xr-render-demo/main.py` (native-profile gate), `docs/adding-cloudxr.md`, `docs/xr-render-demo.md` (client-type section) |
| scene service YAML config keys | `agent-samples/xr-render-demo/scene/scene_service.yaml`, orchestrator process declaration, `docs/xr-render-demo.md` |
| render-mcp YAML config keys | `agent-mcp-servers/render-mcp/render_mcp.yaml`, worker URL constants |
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
- `agent-sdk/xr-ai-nat/` — NAT framework dependencies plus only the smallest
  capability-specific dependencies. Spatial math remains CPU-only and has no
  tracking, service, model, or MCP dependency.
- Agent workers — `xr-ai-agent` + `xr-ai-models` + task-specific libs (numpy,
  torch, etc.). Must never import from `xr-media-hub` or `xr-ai-launcher`.
- New external deps require a note here explaining why they were added.
