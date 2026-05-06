<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# AI inference servers

Read this when adding, calling, or operating an inference server. For the
orchestrator pattern that wires servers into a sample, see
`docs/adding-a-sample.md`.

Multiple reusable HTTP servers are available as launchable peers of
`server-runtime/`. All expose an OpenAI-compatible REST API so agent workers
can call them with any OpenAI SDK client or plain `httpx` / `requests`. Three
LLM backends ship side-by-side under `ai-services/llm/` — pick one per sample
based on the tool-calling / reasoning / hardware trade-offs documented below.

| Server | Command | Port | Model | Backend |
|---|---|---|---|---|
| `ai-services/vlm-server/` | `vlm_server` | 8100 | Cosmos-Reason1-7B | transformers in-process |
| `ai-services/stt-server/` | `stt_server` | 8103 | parakeet-tdt-0.6b-v3 | NeMo ASR in-process |
| `ai-services/tts/magpie/` | `magpie_tts_server` | 8104 | magpie_tts_multilingual_357m | NeMo TTS in-process |
| `ai-services/tts/piper/` | `piper_tts_server` | 8105 | rhasspy/piper-voices (ONNX) | piper-tts in-process |
| `ai-services/llm/llama_nemotron/` | `llama_nemotron_llm_server` | 8106 | Llama-3.1-Nemotron-Nano-8B-v1 | transformers in-process (+ LMFE) |
| `ai-services/llm/nemotron3_nano/` | `nemotron3_nano_llm_server` | 8107 | NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | vLLM (execvp shim) |
| `ai-services/llm/nemotron_omni/` | `nemotron_omni_llm_server` | 8108 | Nemotron-3-Nano-Omni-30B-A3B-Reasoning (NVFP4 / FP8 / BF16, GPU-selected) | vLLM — multimodal (text + video) |
| `agent-mcp-servers/transcript-mcp/` | `transcript_mcp_server` | 8200 | — | JSONL + FastMCP |
| `agent-mcp-servers/video-mcp/` | `video_mcp_server` | 8210 | — | FastMCP → hub |
| `agent-mcp-servers/vlm-mcp/` | `vlm_mcp_server` | 8220 | — | FastMCP → vlm-server (`ask_image` tool) |

All model weights land in `models/` at the repo root (gitignored, shared across
all servers). Each YAML configures `model_cache` — resolved relative to the
YAML file.

## Adding a server to a sample

**1 — Add the process to the orchestrator:**

```python
PROCESSES = [
    Process("hub",    "../../server-runtime",                     "xr_media_hub"),
    Process("vlm",    "../../ai-services/vlm-server",             "vlm_server"),   # ← add as needed
    # Pick ONE LLM backend per sample — they bind different default ports
    # (8106 / 8107) so running more than one at once is allowed but
    # usually unnecessary.
    Process("llm",    "../../ai-services/llm/llama_nemotron",     "llama_nemotron_llm_server"),
    # Process("llm",  "../../ai-services/llm/nemotron3_nano",     "nemotron3_nano_llm_server"),
    Process("stt",    "../../ai-services/stt-server",             "stt_server"),
    # Pick one TTS server
    Process("tts",    "../../ai-services/tts/piper",    "piper_tts_server"),
    # Process("tts",    "../../ai-services/tts/magpie",             "magpie_tts_server"),
    Process("worker", "worker",                                   "my_agent_worker"),
]
```

The agent samples in this repo (`simple-vlm-example`) default to Piper
TTS — it runs on CPU with ~100 ms/sentence latency and avoids the NeMo
dep tree. Magpie is still a supported option (better voice quality,
multilingual) when GPU is available; swap the `Process` row and YAML.

**2 — Copy the reference YAML to your sample's `yaml/` directory:**

```bash
mkdir -p yaml
cp ../../ai-services/vlm-server/vlm_server.yaml ./yaml/vlm_server.yaml
# Pick ONE LLM YAML — copy the one matching the Process you picked above.
cp ../../ai-services/llm/llama_nemotron/llama_nemotron_llm_server.yaml ./yaml/llama_nemotron_llm_server.yaml
# cp ../../ai-services/llm/nemotron3_nano/nemotron3_nano_llm_server.yaml ./yaml/nemotron3_nano_llm_server.yaml
cp ../../ai-services/stt-server/stt_server.yaml ./yaml/stt_server.yaml
cp ../../ai-services/tts/piper/piper_tts_server.yaml ./yaml/piper_tts_server.yaml
# Or for Magpie (multilingual, GPU, ~2-5 s/sentence):
cp ../../ai-services/tts/magpie/magpie_tts_server.yaml ./yaml/magpie_tts_server.yaml
# MCP servers:
cp ../../agent-mcp-servers/transcript-mcp/transcript_mcp_server.yaml ./yaml/transcript_mcp_server.yaml
cp ../../agent-mcp-servers/video-mcp/video_mcp_server.yaml ./yaml/video_mcp_server.yaml
```

Edit the YAML as needed (model, port, device, etc.). The launcher auto-discovers
`yaml/<command>.yaml` in the sample root and passes it as `--config`.

## Calling the servers from a worker

```python
import httpx

# STT — POST multipart/form-data
async with httpx.AsyncClient() as client:
    resp = await client.post(
        "http://localhost:8103/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={"response_format": "json"},
    )
    transcript = resp.json()["text"]

# TTS — POST JSON
async with httpx.AsyncClient() as client:
    resp = await client.post(
        "http://localhost:8104/v1/audio/speech",
        json={"input": "Hello from XR.", "response_format": "wav"},
    )
    wav_bytes = resp.content

# VLM — POST JSON with base64 image
async with httpx.AsyncClient() as client:
    resp = await client.post(
        "http://localhost:8100/v1/chat/completions",
        json={"model": "vlm", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": "What do you see?"},
        ]}]},
    )
    answer = resp.json()["choices"][0]["message"]["content"]

# LLM — POST JSON (pure-text chat completion)
# Ports: 8106 llama_nemotron | 8107 nemotron3_nano.
# The HTTP contract is identical across both; swap the port to swap
# backends with no worker-side code changes.
async with httpx.AsyncClient() as client:
    resp = await client.post(
        "http://localhost:8106/v1/chat/completions",
        json={"model": "llm", "messages": [
            {"role": "user", "content": "Say OK"},
        ], "max_tokens": 16},
    )
    answer = resp.json()["choices"][0]["message"]["content"]
```

## vLLM model persistence

The three vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`) **survive stack restarts by design**. Each wrapper
script checks its health endpoint before spawning vLLM:

- **Already running** → touch the ready file immediately, then idle. Stack is
  ready in seconds; no model reload.
- **Not running** → spawn vLLM normally, wait for `/health`, touch ready file.

vLLM is spawned with `start_new_session=True` so the launcher's `killpg()` does
not reach it on shutdown. The wrapper exits cleanly; vLLM keeps running.

**Stopping the persisted servers** — run from the sample directory:

```bash
uv run xr_render_demo --stop
```

This hits each model server's `/health` endpoint, finds the listening PID via
`ss` (or `lsof`), sends `SIGTERM`, and waits up to 20 s before `SIGKILL`. It
is safe to run while the stack is down — processes that are not running are
silently skipped.

The target ports are defined in `_PERSISTENT_SERVERS` in `main.py` and match the
defaults in the per-profile YAML files. Update that list if you change the port
in a YAML.

## Per-server notes

- **vlm-server** loads Cosmos-Reason1-7B in-process via HuggingFace transformers.
  Model warms up at startup; strips `<think>…</think>` blocks automatically.
- **llm/llama_nemotron** loads Llama-3.1-Nemotron-Nano-8B-v1 via HuggingFace
  transformers (no `trust_remote_code`). Native Llama-3.1 tool calling —
  `tools=[...]` in the request is rendered via the model's chat template and
  decoding is grammar-constrained by `lm-format-enforcer` so the tool-call JSON
  is always syntactically valid. Per-turn reasoning toggle via
  `"detailed thinking on"` / `"detailed thinking off"` in a system or user message;
  reasoning preamble is **not** stripped server-side. See
  [`ai-services/llm/llama_nemotron/README.md`](../ai-services/llm/llama_nemotron/README.md)
  for the full HTTP contract and tuning knobs.
- **llm/nemotron3_nano** is a ~200-line `execvp` shim into vLLM serving
  `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`. vLLM handles tool calling
  (`qwen3_coder` parser), reasoning extraction (`nano_v3` parser — auto-fetched
  into `model_cache`), and FlashInfer FP4 MoE kernels. Requires a Blackwell-class
  GPU (B200 / RTX PRO 6000) for native FP4; swap to the BF16 model variant for
  Hopper/Ampere. `enforce_eager: true` by default to avoid the silent 3–8 min
  CUDA graph + FlashInfer autotune on cold start. See
  [`ai-services/llm/nemotron3_nano/README.md`](../ai-services/llm/nemotron3_nano/README.md)
  for the vLLM flags it forwards and Blackwell prerequisites.
- **llm/nemotron_omni** is a vLLM-backed multimodal LLM serving
  `Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (text + video input) at port 8108.
  The YAML auto-selects between three model variants by detected GPU compute
  capability: NVFP4 on Blackwell (SM100+), FP8 on Ada/Hopper, BF16 forced via
  `use_bf16: true` for highest quality at the largest VRAM cost. Same
  OpenAI-compatible HTTP contract as the other LLM servers — swap the port to
  swap backends.
- **stt-server** loads parakeet-tdt-0.6b-v3 via NeMo ASR in-process.
  English-only; `language` / `temperature` form fields are accepted but ignored.
- **tts/magpie** loads magpie_tts_multilingual_357m via NeMo TTS in-process.
- **tts/piper** serves any rhasspy/piper-voices ONNX voice; ~100 ms/sentence on CPU.
  All inference runs in a thread pool so the asyncio loop is never blocked.
- **transcript-mcp-server** is pure FastMCP at `/mcp` on port 8200.
  Records are keyed by free-form `source_id` (live participant identity
  *or* an internal source name like `"agent-vlm"`). Tools:
  `query_transcripts`, `add_transcript` (worker ingest), `list_sources`,
  `get_transcript_stats`. Transcripts persist as JSONL alongside a
  `.identity` sidecar so list/query round-trip raw IDs cleanly even
  when sanitized filenames collide.
- **video-mcp-server** is pure FastMCP at `/mcp` on port 8210.
  Connects to the hub as a `ProcessorEndpoint` (`Subscribe.VIDEO`) for
  live frames. Tools exposed depend on whether `recordings_dir` is set
  in the YAML:
  - **Always**: `list_live_participants`, `get_latest_frame` (live IPC frame, no recording needed).
  - **Only when `recordings_dir` is configured**: `list_recorded_participants`,
    `get_video_stats`, `query_video`, `get_frame_from_time` (historical
    chunk lookup via NVDEC). Requires `video_recording.enabled: true`
    in `xr_media_hub.yaml` with a matching `out_dir`.
- Ports are configurable — avoid conflicts with LiveKit (7880–7882) and hub (8080, 8090).
- **Sample YAMLs** for each service ship in their own service directory.
  Copy them to your sample root and adjust `model_cache` (`../../models` resolves
  to `xr-ai/models/` from any `agent-samples/<name>/` directory).
