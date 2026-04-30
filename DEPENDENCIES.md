# Dependency Map

> **AGENTS: This file is mandatory to maintain.**
> Any change to a `pyproject.toml`, a YAML config/example, a documented
> interface, or an architectural decision **must** be reflected here in the
> same commit. A change is not complete until this file is up to date.

---

## Internal packages

```
xr-ai-agent  (agent-sdk/)
    └── pyzmq >=26.0
    └── msgpack >=1.0

xr-ai-launcher  (launcher/)
    └── (stdlib only — zero runtime deps)

xr-media-hub  (server-runtime/)
    └── xr-ai-agent  [editable: ../agent-sdk]
    └── pyzmq >=26.0
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

xr-ai-tests  (tests/)
    └── xr-ai-agent   [editable: ../agent-sdk]
    └── xr-media-hub  [editable: ../server-runtime]
    └── pytest >=8.0
    └── pytest-asyncio >=0.23
    └── numpy >=1.24
    Multi-client / multi-agent integration tests over the IPC layer.
    Driven via ZMQ `ipc://` only — no Docker / LiveKit / NVENC required.

vlm-server  (vlm-server/)
    └── torch >=2.2
    └── torchvision >=0.17
    └── transformers >=4.49
    └── accelerate >=0.30
    └── qwen-vl-utils >=0.0.8
    └── Pillow >=10.0
    └── numpy >=1.24
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    Model: nvidia/Cosmos-Reason1-7B (Qwen2.5-VL architecture, in-process)

stt-server  (stt-server/)
    └── nemo_toolkit[asr] >=2.0
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── python-multipart >=0.0.9
    └── pyyaml >=6.0
    Model: nvidia/parakeet-tdt-0.6b-v3 (NeMo ASR, in-process)

magpie-tts-server  (tts/magpie/)
    └── nemo_toolkit[tts] >=2.0
    └── soundfile >=0.12
    └── numpy >=1.24
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    Model: nvidia/magpie_tts_multilingual_357m (NeMo TTS, in-process)

llm-server  (llm-server/)
    └── torch >=2.2
    └── transformers >=4.49
    └── accelerate >=0.30
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── hf-transfer >=0.1.4
    └── pyyaml >=6.0
    Model: nvidia/Mistral-NeMo-Minitron-8B-Instruct (HuggingFace transformers, in-process)

piper-tts-server  (tts/piper/)
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
| `vlm-server/` | `vlm-server` | `vlm_server` | 8100 | Cosmos-Reason1-7B | transformers in-process |
| `llm-server/` | `llm-server` | `llm_server` | 8101 | Mistral-NeMo-Minitron-8B-Instruct | transformers in-process |
| `stt-server/` | `stt-server` | `stt_server` | 8103 | parakeet-tdt-0.6b-v3 | NeMo ASR in-process |
| `tts/magpie/` | `magpie-tts-server` | `magpie_tts_server` | 8104 | magpie_tts_multilingual_357m | NeMo TTS in-process |
| `tts/piper/` | `piper-tts-server` | `piper_tts_server` | 8105 | rhasspy/piper-voices (ONNX) | piper-tts in-process |
| `agent-mcp-servers/transcript-mcp/` | `transcript-mcp-server` | `transcript_mcp_server` | 8200 | — | Pure FastMCP (JSONL storage) |
| `agent-mcp-servers/video-mcp/` | `video-mcp-server` | `video_mcp_server` | 8210 | — | Pure FastMCP (reads NVENC chunks from disk) |

All model weights are cached under `models/` at the repo root (gitignored except
`.gitkeep`).  Cache path is configured via `model_cache` in each YAML, resolved
relative to the YAML file's directory.

---

## Agent samples

### echo-agent  (agent-samples/echo-agent/)

STT → TTS echo pipeline: audio → STT → TTS → audio; text → TTS → audio.

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `echo-agent` | `xr-ai-launcher` | — |
| Worker | `echo-agent-worker` | `xr-ai-agent` | numpy >=1.24, httpx >=0.27, pyyaml >=6.0 |

Uses stt-server (port 8103), tts-server (port 8104).

### vlm-agent  (agent-samples/vlm-agent/)

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `vlm-agent` | `xr-ai-launcher` | — |
| Worker | `vlm-agent-worker` | `xr-ai-agent` | numpy >=1.24, Pillow >=10.0, httpx >=0.27, pyyaml >=6.0 |

Worker calls the vlm-server HTTP API (`POST /v1/chat/completions`) and tts-server HTTP API
(`POST /v1/audio/speech`) — no model weights loaded in-process.

Uses tts-server (port 8104).

### cloudxr-agent  (agent-samples/cloudxr-agent/)

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `cloudxr-agent` | `xr-ai-launcher` | — |
| Worker | `cloudxr-agent-worker` | `xr-ai-agent` | — |

### mcp-agent  (agent-samples/mcp-agent/)

Continuous STT → transcript ingest + MCP-accessible transcript and video query.

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `mcp-agent` | `xr-ai-launcher` | — |
| Worker | `mcp-agent-worker` | `xr-ai-agent` | numpy >=1.24, httpx >=0.27, fastmcp >=0.4, pyyaml >=6.0 |
| MCP server | `mcp-server` | `transcript-mcp-server`, `video-mcp-server` | fastmcp >=0.4, uvicorn[standard] >=0.29, pyyaml >=6.0 |

Composed pure-FastMCP server at port 8200 mounts `transcript_*` and `video_*`
tools at `/mcp`. Worker reaches it via `fastmcp.Client`; uses STT (8103) for
transcription. Hub video recording requires `PyNvVideoCodec` (dep of
`xr-media-hub`; included in `uv sync`).

---

## Change impact map

When you change something in the left column, **all items on the right must be
updated in the same commit**.

| Component changed | Must also update |
|---|---|
| `agent-sdk/` API or types | `AGENTS.md` worker boilerplate, any sample worker that uses the changed API |
| `server-runtime/` config fields (`LiveKitConnectorConfig`) | `server-runtime/xr_media_hub.yaml` (reference copy), each sample's `xr_media_hub.yaml`, `AGENTS.md` Config section |
| `launcher/` `Process` / `run_stack` API | `AGENTS.md` orchestrator boilerplate and process model section |
| vlm-agent model class or supported architectures | `agent-samples/vlm-agent/vlm_agent_worker.yaml.example` comments |
| vlm-agent YAML config keys (`model`, `hf_token`, `system_prompt`, …) | `agent-samples/vlm-agent/vlm_agent_worker.yaml.example`, worker `__main__.py` docstring |
| cloudxr-runtime YAML config keys | `agent-samples/cloudxr-agent/cloudxr_runtime.yaml`, `AGENTS.md` CloudXR section |
| Any `pyproject.toml` dependency | `DEPENDENCIES.md` (this file) |
| Any new sample added | `DEPENDENCIES.md`, `AGENTS.md`, `README.md` |
| Any new shared component added (peer of `server-runtime/`) | `AGENTS.md` Architecture section, `DEPENDENCIES.md` |

---

## Dependency rules (enforced)

- `launcher/` — zero runtime dependencies. Stdlib only.
- `agent-sdk/` — only `pyzmq` + `msgpack`. No server-side packages.
- Agent workers — `xr-ai-agent` + task-specific libs (numpy, torch, etc.).
  Must never import from `xr-media-hub` or `xr-ai-launcher`.
- New external deps require a note here explaining why they were added.
