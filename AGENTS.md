# xr-ai — Working Conventions

Guidelines for developers and AI assistants working in this repo.

## Architecture

```
client-samples/     # Platform clients (Android, iOS/visionOS, Web)
server-runtime/     # XR-Media-Hub core + LiveKit transport
agent-sdk/          # xr-ai-agent: IPC client library (pyzmq + msgpack only)
launcher/           # stdlib-only process manager (used by samples)
cloudxr-runtime/    # Shared CloudXR OpenXR runtime + WSS proxy (opt-in per sample)
ai-services/        # OpenAI-compatible AI inference servers (VLM, STT, TTS)
agent-mcp-servers/  # MCP adapters: oxr, render, client, xr-media, transcript, video
agent-samples/      # End-to-end agent demos
tests/              # Multi-client / multi-agent integration tests
docs/               # Design docs
```

Key design decisions:
- **One hub, many clients, many agents.** A single hub instance fans the
  inbound stream out to every connected ``ProcessorEndpoint`` (agent) and
  routes return traffic back to the originating client only — never to peers.
- **XR-Media-Hub** is transport-agnostic at its IPC boundary. Agents connect via IPC only.
- **LiveKit** is an internal transport detail — not exposed to the agent layer.
  When LiveKit is the transport, return audio is published as one track per
  participant (`xr-hub-return-{pid}`) with subscribe permissions restricted to
  that participant; return data uses ``destination_identities`` for the same
  reason. Agents never need to know.
- **`agent-sdk/`** (`xr-ai-agent`) contains only the agent-facing IPC layer. Its sole
  runtime dependencies are `pyzmq` and `msgpack` — no LiveKit, FastAPI, or uvicorn.
- MCP servers are the agent's only interface to XR data and rendering.
- No API keys or tokens in source files — use env vars or `xr_media_hub.yaml`.

## Process model

Every sample is self-contained: running it starts the hub and all required
processes automatically. No separate server launch step.

Each sample has **two sub-projects**:

| Sub-project | Role | Dependencies |
|---|---|---|
| `<sample>/` | Orchestrator — declares process list in code, launches all | `xr-ai-launcher` only (stdlib) |
| `<sample>/worker/` | Agent worker — connects to hub via IPC, runs agent logic | `xr-ai-agent`, numpy, etc. |

**Launchable convention** — every sub-project that can be run is self-describing:
it has an entry-point command and optionally a YAML config named `<command>.yaml`
that lives in the sample root.  The launcher discovers the YAML automatically
and passes it as `--config`.  No separate launcher config file exists.

The orchestrator declares the process sequence in code:
```python
_BASE = Path(__file__).resolve().parents[1]   # sample root

PROCESSES = [
    Process("hub",     "../../server-runtime",  "xr_media_hub"),
    Process("worker",  "worker",                "my_agent_worker"),
    # Optional shared components — add as needed:
    # Process("cloudxr", "../../cloudxr-runtime",         "cloudxr_runtime"),
    # Process("mcp",     "../../agent-mcp-servers/oxr",   "oxr_mcp"),
]

def run() -> None:
    asyncio.run(run_stack(PROCESSES, _BASE))
```

Rules:
- **All processes start concurrently** — no ordering is required or expressed.
  Every process must tolerate its peers not being ready at startup.
  ZMQ reconnects automatically; `ProcessorEndpoint` works regardless of hub startup order.
- `xr_media_hub` always runs as its own process — never embedded in-process.
- The worker never imports anything from `server-runtime` or `launcher/`.
- Process management lives in `launcher/`, not inside any process it manages.
- `run_stack` is fail-fast: if any process exits, the rest are terminated.

## Credentials

The launcher manages HuggingFace and NGC API tokens so they are never stored
in source files or YAML configs.

Tokens are cached in `~/.config/xr-ai/credentials.json` — outside any repo,
no `.gitignore` entry required.  Values already in `os.environ` always take
priority (useful in CI or when you want to override the cache).

`HF_TOKEN` is additionally written to `~/.cache/huggingface/token`, the same
file used by `huggingface-cli login`.  This means:
- Child processes find it without relying on env-var inheritance.
- If you've already run `huggingface-cli login`, no prompt appears.

### Prompting for a token

Call `ensure_credentials` **before** `asyncio.run(run_stack(...))` in any
orchestrator that needs a token:

```python
from xr_ai_launcher import ensure_credentials, run_stack

def run() -> None:
    ensure_credentials("HF_TOKEN")          # prompts once, saves for future runs
    asyncio.run(run_stack(PROCESSES, _BASE))
```

Supported tokens: `HF_TOKEN`, `NGC_API_KEY`.  The user is shown a one-line
prompt (password-style, no echo) with a link to generate the token.  Pressing
Enter without typing skips the token (left unset, not saved).

### Automatic injection

`run_stack` always calls `load_credentials()` internally before spawning child
processes, so any token already saved in the credentials file is available to
every subprocess in the stack — even orchestrators that never call
`ensure_credentials` directly.

### Managing saved tokens

```bash
# View saved tokens
cat ~/.config/xr-ai/credentials.json

# Remove a token (re-run will prompt again)
python3 -c "
import json, pathlib
p = pathlib.Path.home() / '.config/xr-ai/credentials.json'
d = json.loads(p.read_text()); d.pop('HF_TOKEN', None); p.write_text(json.dumps(d, indent=2))
"
```

---

## Using AI inference servers

Multiple reusable HTTP servers are available as launchable peers of `server-runtime/`.
All expose an OpenAI-compatible REST API so agent workers can call them with any
OpenAI SDK client or plain `httpx` / `requests`. Three LLM backends ship side-by-side
under `ai-services/llm/` — pick one per sample based on the tool-calling /
reasoning / hardware trade-offs documented below.

| Server | Command | Port | Model | Backend |
|---|---|---|---|---|
| `ai-services/vlm-server/` | `vlm_server` | 8100 | Cosmos-Reason1-7B | transformers in-process |
| `ai-services/llm/mistral_minitron/` | `mistral_minitron_llm_server` | 8101 | Mistral-NeMo-Minitron-8B-Instruct | transformers in-process |
| `ai-services/stt-server/` | `stt_server` | 8103 | parakeet-tdt-0.6b-v3 | NeMo ASR in-process |
| `ai-services/tts/magpie/` | `magpie_tts_server` | 8104 | magpie_tts_multilingual_357m | NeMo TTS in-process |
| `ai-services/tts/piper/` | `piper_tts_server` | 8105 | rhasspy/piper-voices (ONNX) | piper-tts in-process |
| `ai-services/llm/llama_nemotron/` | `llama_nemotron_llm_server` | 8106 | Llama-3.1-Nemotron-Nano-8B-v1 | transformers in-process (+ LMFE) |
| `ai-services/llm/nemotron3_nano/` | `nemotron3_nano_llm_server` | 8107 | NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | vLLM (execvp shim) |
| `agent-mcp-servers/transcript-mcp/` | `transcript_mcp_server` | 8200 | — | JSONL + FastMCP |
| `agent-mcp-servers/video-mcp/` | `video_mcp_server` | 8210 | — | FastMCP → hub |

All model weights land in `models/` at the repo root (gitignored, shared across all
servers).  Each YAML configures `model_cache` — resolved relative to the YAML file.

### Adding a server to a sample

**1 — Add the process to the orchestrator:**

```python
PROCESSES = [
    Process("hub",    "../../server-runtime",                     "xr_media_hub"),
    Process("vlm",    "../../ai-services/vlm-server",             "vlm_server"),   # ← add as needed
    # Pick ONE LLM backend per sample — they bind different default ports
    # (8101 / 8106 / 8107) so running more than one at once is allowed but
    # usually unnecessary.
    Process("llm",    "../../ai-services/llm/mistral_minitron",   "mistral_minitron_llm_server"),
    # Process("llm",  "../../ai-services/llm/llama_nemotron",     "llama_nemotron_llm_server"),
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
dep tree.  Magpie is still a supported option (better voice quality,
multilingual) when GPU is available; swap the `Process` row and YAML.

**2 — Copy the reference YAML to your sample root:**

```bash
cp ../../ai-services/vlm-server/vlm_server.yaml ./vlm_server.yaml
# Pick ONE LLM YAML — copy the one matching the Process you picked above.
cp ../../ai-services/llm/mistral_minitron/mistral_minitron_llm_server.yaml ./mistral_minitron_llm_server.yaml
# cp ../../ai-services/llm/llama_nemotron/llama_nemotron_llm_server.yaml ./llama_nemotron_llm_server.yaml
# cp ../../ai-services/llm/nemotron3_nano/nemotron3_nano_llm_server.yaml ./nemotron3_nano_llm_server.yaml
cp ../../ai-services/stt-server/stt_server.yaml ./stt_server.yaml
cp ../../ai-services/tts/piper/piper_tts_server.yaml ./piper_tts_server.yaml
# Or for Magpie (multilingual, GPU, ~2-5 s/sentence):
cp ../../ai-services/tts/magpie/magpie_tts_server.yaml ./magpie_tts_server.yaml
# MCP servers:
cp ../../agent-mcp-servers/transcript-mcp/transcript_mcp_server.yaml ./transcript_mcp_server.yaml
cp ../../agent-mcp-servers/video-mcp/video_mcp_server.yaml ./video_mcp_server.yaml
```

Edit the YAML as needed (model, port, device, etc.).  The launcher auto-discovers
`<command>.yaml` in the sample root and passes it as `--config`.

### Calling the servers from a worker

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
# Ports: 8101 mistral_minitron | 8106 llama_nemotron | 8107 nemotron3_nano.
# The HTTP contract is identical across all three; swap the port to swap
# backends with no worker-side code changes.
async with httpx.AsyncClient() as client:
    resp = await client.post(
        "http://localhost:8101/v1/chat/completions",
        json={"model": "llm", "messages": [
            {"role": "user", "content": "Say OK"},
        ], "max_tokens": 16},
    )
    answer = resp.json()["choices"][0]["message"]["content"]
```

### Notes

- **vlm-server** loads Cosmos-Reason1-7B in-process via HuggingFace transformers.
  Model warms up at startup; strips `<think>…</think>` blocks automatically.
- **llm/mistral_minitron** loads Mistral-NeMo-Minitron-8B-Instruct in-process via
  HuggingFace transformers. Pure-text `/v1/chat/completions` only — no tool
  calling, no reasoning preamble. Default stop list handles Minitron's
  `<extra_id_*>` chat-template tokens. Swap models via `mistral_minitron_llm_server.yaml`.
- **llm/llama_nemotron** loads Llama-3.1-Nemotron-Nano-8B-v1 via HuggingFace
  transformers (no `trust_remote_code`). Native Llama-3.1 tool calling —
  `tools=[...]` in the request is rendered via the model's chat template and
  decoding is grammar-constrained by `lm-format-enforcer` so the tool-call JSON
  is always syntactically valid. Per-turn reasoning toggle via
  `"detailed thinking on"` / `"detailed thinking off"` in a system or user message;
  reasoning preamble is **not** stripped server-side.
- **llm/nemotron3_nano** is a ~200-line `execvp` shim into vLLM serving
  `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`. vLLM handles tool calling
  (`qwen3_coder` parser), reasoning extraction (`nano_v3` parser — auto-fetched
  into `model_cache`), and FlashInfer FP4 MoE kernels. Requires a Blackwell-class
  GPU (B200 / RTX PRO 6000 / Jetson Thor) for native FP4; swap to the BF16 model
  variant for Hopper/Ampere. `enforce_eager: true` by default to avoid the
  silent 3–8 min CUDA graph + FlashInfer autotune on cold start.
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
- **video-mcp-server** is pure FastMCP at `/mcp` on port 8210. Tools:
  `list_live_participants`, `list_recorded_participants`,
  `get_video_stats`, `query_video`, `get_latest_frame`,
  `get_frame_at_time`. `list_live_participants` returns the hub's IPC
  roster (the only pids `get_latest_frame` will succeed for);
  `list_recorded_participants` returns raw identities recovered from
  per-pid `.identity` sidecars in the recordings directory. Reads
  NVENC chunks from disk for historical queries; connects to the hub
  as a `ProcessorEndpoint` (`Subscribe.VIDEO`) for live frames.
  `get_frame_at_time` decodes via NVDEC and returns a PNG path.
  Requires hub video recording enabled via
  `video_recording.enabled: true` in `xr_media_hub.yaml`.
- Ports are configurable — avoid conflicts with LiveKit (7880–7882) and hub (8080, 8090).
- **Sample YAMLs** for each service ship with `cloudxr-agent` or `mcp-agent` as examples.
  Copy them to other sample roots and adjust `model_cache` (`../../models` resolves
  to `xr-ai/models/` from any `agent-samples/<name>/` directory).

---

## Adding a new sample

### Naming conventions

Choose a kebab-case sample name (e.g. `simple-vlm-example`,
`cloudxr-agent`).  Derive all other names from it mechanically:

| Thing | Convention | Example |
|---|---|---|
| Sample directory | `agent-samples/<kebab-name>/` | `simple-vlm-example/` |
| Orchestrator module | `<snake_name>.py` | `simple_vlm_example.py` |
| Orchestrator entry point | `<snake_name>` | `simple_vlm_example` |
| Worker entry module | `<snake_name>_worker.py` | `simple_vlm_example_worker.py` |
| Worker entry point | `<snake_name>_worker` | `simple_vlm_example_worker` |
| Agent class | `<CamelName>Agent` | `SimpleVlmAgent` |
| Logger name | `"<snake_name>"` | `"simple_vlm_example"` |
| pyproject name (orch) | `"<kebab-name>"` | `"simple-vlm-example"` |
| pyproject name (worker) | `"<kebab-name>-worker"` | `"simple-vlm-example-worker"` |

### Directory layout

Both sub-projects use **flat module layouts** — no nested package
directories, no `__main__.py`, no `__init__.py`.  Hatchling ships the
listed `.py` files as top-level modules in each sub-project's isolated
venv.

```
agent-samples/<name>/
├── pyproject.toml                  ← orchestrator project
├── xr_media_hub.yaml               ← hub config for this sample
├── <snake_name>.py                 ← orchestrator (declare PROCESSES, call run_stack)
└── worker/
    ├── pyproject.toml              ← worker project
    ├── <snake_name>_worker.py      ← entry point: parses config, runs main loop
    ├── agent.py                    ← <CamelName>Agent class
    └── …                           ← split helpers as needed (audio.py, services.py, …)
```

When the worker is small (≲ 100 lines) keep it as a single file —
`worker/<snake_name>_worker.py` containing everything.  Only split once
the file makes the agent logic harder to read; aim for a few focused
modules over one monolith *or* a swarm of tiny files.

Suggested split (used by `simple-vlm-example` and `mcp-agent`):

| File | Responsibility |
|---|---|
| `<snake>_worker.py` | Entry point: config parsing, signal handling, lifecycle |
| `agent.py` | The agent class — IPC callbacks and orchestration |
| `audio.py` | WAV/PCM helpers, RMS, pixel-format conversion (rename if not audio) |
| `services.py` | Thin HTTP/MCP clients for external services + readiness probe |
| `voice.py` | Per-participant VAD/streaming-STT bookkeeping (when applicable) |


### Orchestrator `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<kebab-name>"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["xr-ai-launcher"]

[tool.uv.sources]
xr-ai-launcher = { path = "../../launcher", editable = true }

[project.scripts]
<snake_name> = "<snake_name>:run"

[tool.hatch.build.targets.wheel]
only-include = ["<snake_name>.py"]
```

### Worker `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<kebab-name>-worker"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "xr-ai-agent",
    # add task-specific deps here: numpy, torch, etc.
]

[tool.uv.sources]
xr-ai-agent = { path = "../../../agent-sdk", editable = true }

[project.scripts]
<snake_name>_worker = "<snake_name>_worker:run"

[tool.hatch.build.targets.wheel]
only-include = [
    "<snake_name>_worker.py",
    "agent.py",
    # add other split modules here
]
```

When the worker is a single file, drop the extra entries:

```toml
[tool.hatch.build.targets.wheel]
only-include = ["<snake_name>_worker.py"]
```

### Orchestrator `<snake_name>.py`

Exact boilerplate — do not add logic here:

```python
"""
<Name> agent orchestrator.  Runs the process stack for this sample.

How to run (from agent-samples/<name>/):
    uv sync && uv run <snake_name>
"""
import asyncio
from pathlib import Path

from xr_ai_launcher import Process, run_stack

_BASE = Path(__file__).resolve().parent

PROCESSES = [
    Process("hub",    "../../server-runtime", "xr_media_hub"),
    Process("worker", "worker",               "<snake_name>_worker"),
]


def run() -> None:
    asyncio.run(run_stack(PROCESSES, _BASE))


if __name__ == "__main__":
    run()
```

### Worker `<snake_name>_worker.py`

Follow this structure exactly.  Fill in the sections marked `# ← FILL IN`.

```python
"""
<Name> agent worker — <one-line description>.

Launched as a subprocess by ``uv run <snake_name>`` (the orchestrator).
Do not run this directly.

Protocol                            # ← include only if the worker sends/receives data msgs
--------
Client → agent  (topic "<in.topic>"):
    <description>

Agent → client  (topic "<out.topic>"):
    <description>

Environment                         # ← include only if env vars are read
-----------
    ENV_VAR   description (default: value)
"""
from __future__ import annotations

import asyncio
import logging
import signal

from xr_ai_agent import (          # ← import only what you use
    AudioChunk, DataMessage, FrameSignal, ParticipantEvent, ProcessorEndpoint,
    Subscribe,                     # ← only needed when scoping subscriptions
)

log = logging.getLogger("<snake_name>")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


class <CamelName>Agent:            # ← FILL IN agent logic

    def __init__(self) -> None:
        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_frame(self._on_frame)       # ← remove callbacks you don't use
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

    # ── callbacks ─────────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None: ...
    async def _on_audio(self, chunk: AudioChunk) -> None: ...
    async def _on_data(self, msg: DataMessage) -> None: ...
    async def _on_participant(self, event: ParticipantEvent) -> None: ...

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()        # ← start any background tasks before this line

    def shutdown(self) -> None:
        # ← cancel any background tasks here first
        self._ep.stop()
        self._ep.close()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    agent = <CamelName>Agent()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("<snake_name> connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
```

### Rules for worker code

- **Only import from `xr_ai_agent`** for IPC types. Never import from
  `xr_media_hub`, `xr_ai_launcher`, or any server-side package.
- **`_HUB_PUB` / `_HUB_PUSH`** are module-level constants, not magic strings
  scattered through the code.
- **Signal handling**: wire `SIGINT` and `SIGTERM` to `agent.shutdown()` in
  `main()`.  Always wrap `await agent.run()` in `try/finally` that calls
  `shutdown()` — this covers the edge case where `run()` raises.
- **`shutdown()` is synchronous** — it must be safe to call from a signal
  handler.  Cancel asyncio tasks first, then call `ep.stop()` + `ep.close()`.
- **Background tasks** (e.g. a stats loop): create them in `run()` before
  calling `await ep.run()`, and cancel them at the top of `shutdown()`.
- **Callbacks are async** even if they do synchronous work — the signature
  must match `async def _on_*(self, ...)`.
- **CPU-bound or blocking work** (model inference, heavy image processing):
  use `await loop.run_in_executor(None, ...)` to avoid blocking the event loop.
- **One agent class per worker.** Keep the entry-point file thin —
  config parsing, signal handling, and `agent.run()`.  Put the agent
  class in `agent.py`; split further (e.g. `audio.py`, `services.py`)
  once the worker exceeds ~150 lines or mixes unrelated concerns.
- **Imports are absolute, not relative.** Workers ship as flat
  top-level modules — write `from agent import EchoAgent`, not
  `from .agent import EchoAgent`.  Each worker venv is isolated, so
  generic module names (`agent`, `audio`, `services`) don't conflict.
- **Don't add `__init__.py` or `__main__.py`.** Both are unnecessary
  with the flat-module layout and re-introduce nesting.

### Checklist

- [ ] `agent-samples/<name>/pyproject.toml` — orchestrator, deps: `xr-ai-launcher` only
- [ ] `agent-samples/<name>/worker/pyproject.toml` — worker, deps: `xr-ai-agent` + task libs (list every `.py` in `only-include`)
- [ ] `agent-samples/<name>/<snake_name>.py` — exact orchestrator boilerplate
- [ ] `agent-samples/<name>/worker/<snake_name>_worker.py` — entry point + (optional) split helpers next to it
- [ ] `agent-samples/<name>/xr_media_hub.yaml` — hub config (copy from `server-runtime/xr_media_hub.yaml`)
- [ ] `uv sync` in both `agent-samples/<name>/` and `agent-samples/<name>/worker/`
- [ ] `README.md` updated — architecture table and quickstart section

## Adding CloudXR to a sample

`cloudxr-runtime/` is a shared top-level component, like `server-runtime/`.
Any sample can stream XR content to a device by adding one line to its
orchestrator and a config file in the sample root.

### 1 — Add the process to the orchestrator

```python
PROCESSES = [
    Process("hub",     "../../server-runtime",  "xr_media_hub"),
    Process("cloudxr", "../../cloudxr-runtime", "cloudxr_runtime"),  # ← add this
    Process("worker",  "worker",                "my_agent_worker"),
]
```

### 2 — Add `cloudxr_runtime.yaml` to the sample root

The launcher auto-discovers this file and passes it as `--config`.

```yaml
# CloudXR runtime configuration.
cloudxr_install_dir: ~/.cloudxr

# Accept the NVIDIA CloudXR EULA non-interactively.
# View: https://github.com/NVIDIA/IsaacTeleop/blob/main/deps/cloudxr/CLOUDXR_LICENSE
# Written once to <cloudxr_install_dir>/run/eula_accepted; ignored on subsequent runs.
accept_eula: true

# Device profile — controls transport and XR device defaults.
# Valid: auto-native | auto-webrtc | apple-vision-pro | ipad-pro | quest3
cloudxr_env:
  NV_DEVICE_PROFILE: auto-webrtc

# ── Ports (do not conflict with LiveKit) ──────────────────────────────────────
# CloudXR native service:  localhost:49100  (internal)
# WSS proxy (TLS):         0.0.0.0:48322   (XR clients connect here; auto-webrtc only)
```

### Notes

- CloudXR and the hub are **independent stacks**. CloudXR streams sim/render
  content directly to XR devices over WebRTC; the hub handles agent media via
  LiveKit. They share no ports.
- `auto-webrtc` profile starts a WSS proxy on port 48322 for WebRTC signaling.
  `auto-native` uses a direct native transport and does not need the proxy.
- After CloudXR is ready, activate its environment in a separate terminal to
  run an OpenXR app against it:
  ```bash
  source ~/.cloudxr/run/cloudxr.env
  ```
- Full list of supported `NV_*` env vars: `cloudxr-openxr-runtime` source,
  `env_config` / `nv_config.h`.

## Adding a new managed process type

Add `launcher/xr_ai_launcher/_<name>.py` following the pattern in `_hub.py`.
Use `ManagedProcess` as the base. Export from `__init__.py`.

## Documentation rule

**Update `README.md` (and relevant sub-repo docs) in the same task as the code
change.** A change is not done until the docs reflect it. This applies to: new
packages, changed entry points, new quickstart flows, renamed commands, new
config files.

## Dependency discipline

**`DEPENDENCIES.md` at the repo root is the authoritative dependency map.**
Any change to a `pyproject.toml` must update `DEPENDENCIES.md` in the same
commit. This applies to adding, removing, or updating any dependency — internal
or external. A change is not complete until `DEPENDENCIES.md` reflects it.

Hard rules (also documented in `DEPENDENCIES.md`):
- `launcher/` has zero runtime dependencies — stdlib only. Keep it that way.
- `agent-sdk/` (`xr-ai-agent`) depends only on `pyzmq` + `msgpack`. No server deps.
- Agent workers import only from `xr_ai_agent` (and task-specific libs like numpy/torch).
- Agent workers must never import from `xr_media_hub` or `xr_ai_launcher`.
- Don't add abstractions until needed by two concrete use-cases.

## Config

Each sample provides its own `xr_media_hub.yaml` in its project directory
(e.g. `agent-samples/simple-vlm-example/xr_media_hub.yaml`). `server-runtime/` also
contains a reference copy documenting all available fields.

Paths inside the YAML (e.g. `web_client_dir`) resolve relative to the YAML
file's own directory, not CWD. `HubLauncher` finds the YAML automatically by
searching upward from CWD when the orchestrator runs.

### Known limitations

**LiveKit always uses plain `ws://` (no TLS)**

The web server (`web_server_tls: true`) and token endpoint both support HTTPS,
but LiveKit itself always runs over plain WebSocket (`ws://`).  This means:

- The `/token` response returns `url: ws://<host>:<lk_port_ws>`.
- Browsers loaded over HTTPS will block the `ws://` connection as mixed content.
- Native clients (iOS, visionOS, Android) are unaffected — they accept both.

**Workarounds until LiveKit TLS is added:**

1. Use a reverse proxy (nginx, Caddy) in front of LiveKit to terminate TLS and
   forward as plain WebSocket internally.  Point `web_client_dir` at a build
   that targets the proxy URL.
2. Run the web client over plain HTTP (`web_server_tls: false`), which avoids
   the mixed-content restriction.  Camera/mic access requires a secure context,
   so this only works on `localhost` or with a browser flag override.
3. Add native LiveKit TLS: set `tls.cert` and `tls.key` in the generated
   `livekit.yaml` (see `_docker.py`) and change the token URL scheme to `wss://`.
   This is the correct long-term fix but has not been implemented yet.

---

## Decisions & change log

Significant decisions, in reverse-chronological order. Update this whenever a
non-trivial architectural or design decision is made so the rationale is
preserved and not re-litigated.

### 2026-04-30 — Unified MCP IDs: identity sidecars, live vs recorded splits, transcript source_id

The MCP servers had two consistency gaps: (1) `list_*` tools returned
sanitized filesystem names rather than the original LiveKit identities,
so a caller round-tripping a value through `list_recording_participants`
→ `get_latest_frame` could miss; (2) the transcript store named its
key `participant_id` even though transcripts can come from non-
participant sources (e.g. an agent's own TTS).

Changes:

- **Recorder + stores write a `.identity` sidecar per source.** The
  hub's `_recorder.py` writes `<recordings_dir>/<safe>/.identity`;
  transcript-mcp writes `<transcripts_dir>/<safe>.identity` next to
  the JSONL. Sidecar contents are the raw caller-supplied ID verbatim.
  Collisions between distinct raw IDs that happen to share a
  `_safe_name` get a counter suffix (`alice_home`, `alice_home_2`, …).
- **List tools return raw IDs.** `list_recorded_participants` (renamed
  from `list_recording_participants`) and `list_sources` (renamed from
  `list_participants` on transcript-mcp) read sidecars and return
  exactly what the writer passed in.
- **New tool `list_live_participants`** on video-mcp — surfaces
  `ep.connected_participants` from the ProcessorEndpoint so callers
  can ask "who's actually live right now?". This is the only set
  `get_latest_frame` will succeed for.
- **Transcript-mcp renames `participant_id` → `source_id`** in tool
  signatures, response keys, and stored identity sidecars. The store
  treats `source_id` as opaque, allowing agents to write under
  internal names (`"agent-vlm"`, `"tts"`) alongside live participant
  records. video-mcp keeps `participant_id` since video really does
  come from real participants.
- mcp-agent worker updated to use `source_id` when calling transcript
  tools.

**Why:** the underlying storage was always string-keyed and didn't
care, but the API leaked sanitized filenames and overloaded
"participant" semantics onto things that aren't participants. The
sidecar lifts the raw name back out cleanly; the rename names the
field for what it actually is.

### 2026-04-29 — Video recording on tmpfs; video-mcp gains live-frame + frame-at-time

`server-runtime/xr_media_hub/video/_recorder.py`:
- Default `out_dir` flipped from `/tmp/xr_recordings` (disk) to
  `/dev/shm/xr-ai/recordings` (tmpfs — RAM-backed). Writes don't touch
  disk by default.
- Eviction policy is now **size-based, global**: `max_total_bytes`
  (default 500 MB) caps total chunk size across all participants.
  When the cap is exceeded, oldest chunks are evicted FIFO. Replaces
  the prior per-participant `max_chunks` count.

`agent-mcp-servers/video-mcp/`:
- Now connects to the hub as a `ProcessorEndpoint` with
  `filter=Subscribe.VIDEO`. A small `FrameProvider` tracks the most
  recent `FrameSignal` per pid; pixel bytes are pulled on demand via
  `request_frame()`. No on-disk side-channel — the live path is
  entirely IPC-based.
- New MCP tool `get_latest_frame(participant_id)` — calls into the
  provider, converts the returned `FrameData` to RGB, writes a PNG to
  `out_dir`, returns `{path, width, height, timestamp_us, track_id}`.
- New MCP tool `get_frame_at_time(participant_id, timestamp_us)` —
  finds the chunk covering the timestamp, decodes it with NVDEC via
  PyNvVideoCodec, picks the frame closest to the timestamp by linear
  interpolation across the chunk, encodes PNG, returns `{path, width,
  height, timestamp_us, chunk_path}`.
- video-mcp gains `xr-ai-agent`, `PyNvVideoCodec`, `Pillow`, and
  `numpy` runtime deps. mcp-agent's composed `mcp_server` adopts the
  same model — owns its own `ProcessorEndpoint` and lifecycle.

**Why:** disk IO was wasted overhead for chunks that almost always get
evicted within minutes; `/dev/shm` cuts the IO cost to RAM bandwidth
without changing the file-based interface that the video-mcp uses for
historical queries. Live frames bypass the chunk store entirely — the
hub already has the most recent SHM slot held open per (pid, track),
so a `request_frame()` is a single zero-copy memcpy at the hub plus a
pixel-format conversion at the consumer.

### 2026-04-29 — MCP servers go pure FastMCP

`transcript-mcp-server`, `video-mcp-server`, and the composed `mcp-server`
in `agent-samples/mcp-agent/` no longer wrap a FastAPI app. Each runs the
`FastMCP.http_app(path="/mcp")` Starlette app directly under uvicorn.

- All worker ingress is now an MCP tool call. Transcript ingest is the new
  `transcript_add_transcript` tool (replaces `POST /ingest`); stats fetches
  use the existing `transcript_get_transcript_stats` / `video_get_video_stats`
  tools (replace the `/transcript/stats/{pid}` and `/video/stats/{pid}` REST
  routes).
- The composed `mcp-server` no longer reads a `skills:` config block — both
  sub-servers are always mounted; their per-server config lives at the top
  level of `mcp_server.yaml` under `transcript:` and `video:`.
- `/health` is gone. The mcp-agent worker's readiness probe now uses
  `fastmcp.Client.list_tools()` against `/mcp` to confirm the server is
  serving (a stronger guarantee than a 200 from `/health` ever was).
- Drops the `fastapi` and `pydantic` runtime dependencies on transcript-mcp,
  video-mcp, and the composed mcp-server. Worker gains a `fastmcp>=0.4`
  dependency.

**Why:** the dual REST + MCP surface had no value once workers got an MCP
client. Two interfaces meant two contracts to keep in sync, two error-
handling paths, and two readiness checks. Pure FastMCP is one contract,
one error model, and a stronger readiness check via `list_tools()`.

### 2026-04-29 — Participant-keyed agent subscriptions

`ProcessorEndpoint` now models subscriptions as **participants**, not topic
prefixes. The unit of opt-in is "I want everything for participant X";
categories (data / audio / video) are an opt-out filter inside that.

- New `Subscribe` flag enum (`DATA`, `AUDIO`, `VIDEO`, `ALL`) replaces the
  prior raw-bytes `topics=` parameter.
- `ProcessorEndpoint(auto_subscribe=True, filter=Subscribe.ALL)` is the
  default. The endpoint installs an internal participant handler that
  calls `subscribe(pid)` on join and `unsubscribe(pid)` on leave —
  agents see every client's full inbound stream out of the box.
- `ep.subscribe(pid, filter=...)` and `ep.unsubscribe(pid)` are the
  primitives. Idempotent. Calling subscribe with a different filter
  diffs the active subscriptions. Subscribing before the pid joins is
  fine — ZMQ holds the SUBSCRIBE.
- `auto_subscribe=False` is the escape hatch for single-client agents:
  the agent only sees `participant` + `control` until it explicitly
  subscribes. Use `ep.subscribed_participants` to introspect live state.
- New `ROSTER_REQUEST` IPC type (`MsgType.ROSTER_REQUEST = 12`).
  `request_roster()` (called automatically once at the start of
  `run()` when auto-subscribe is on) makes the hub re-publish
  `PARTICIPANT_EVENT(joined=True)` for every current pid so endpoints
  started mid-session catch up. Replays go on the regular `participant`
  topic, so other endpoints' `on_participant` callbacks may fire again
  for known pids — keep them idempotent.
- Topic prefixes always include the trailing `.` so `data.alice.` does
  not bleed into `data.alice2.chat`. The helper centralises this so
  individual agents never type the prefix themselves.

**Why:** the previous bytes-tuple `topics=` parameter forced agents to
know hub-internal topic conventions, made per-pid scoping awkward
(write your own join handler, remember the trailing dot), and didn't
solve the mid-session catch-up problem. The new model treats the
participant as the unit of subscription, which is what every real
agent actually wants — most just take the default broadcast; the few
that need scoping flip `auto_subscribe=False` and call `subscribe`.

### 2026-04-29 — Multi-client / multi-agent isolation; topic surface; tests

The hub formally supports many clients and many agents at once. The IPC and
LiveKit transport layers were extended so:

- **Per-participant return audio** — `RoomClient` now publishes one
  `xr-hub-return-{pid}` audio track per active participant, with subscribe
  permissions restricted via `set_track_subscription_permissions` so a
  participant can only hear their own return audio. Tracks are unpublished
  on participant leave.
- **Targeted return data** — `RoomClient.send_return_data` passes
  `destination_identities=[participant_id]` so return text/binary is no
  longer broadcast to other participants in the room.
- **`ReturnAudioFlush` control message** (`MsgType.RETURN_AUDIO_FLUSH = 11`)
  added to `xr-ai-agent`. `ProcessorEndpoint.flush_return_audio(pid)`
  routes through the hub on `return_audio_flush.<pid>` to the connector,
  which calls `AudioSource.clear_queue()` for that pid only. Used to
  cleanly interrupt agent TTS playback when a new query arrives.
- **StreamKit `onDataReceived(topic, data)`** — the previously dropped
  data-channel `topic` is now surfaced to the application across web and
  iOS/visionOS. The reserved `_agent.status` topic is still intercepted
  internally and never reaches `onDataReceived`.
- **`tests/` top-level suite** — multi-client / multi-agent coverage over
  the real IPC layer (no Docker / LiveKit needed). CI workflow at
  `.github/workflows/tests.yml` runs the suite on every push and PR
  across Python 3.11 and 3.12.

### 2026-04-30 — LLM servers reorganized into per-model packages

`ai-services/llm-server/` (single package, single model) is split into three
sibling packages under `ai-services/llm/`, each with its own entry-point
command, YAML, default port, and dependency set. This lets a sample pick the
LLM that matches its tool-calling / reasoning / hardware requirements without
dragging in the dependencies of the others (notably vLLM and
`lm-format-enforcer`, neither of which the Minitron backend needs).

| New package | Command | Port | Model | Backend |
|---|---|---|---|---|
| `llm/mistral_minitron/` | `mistral_minitron_llm_server` | 8101 | `nvidia/Mistral-NeMo-Minitron-8B-Instruct` | HF transformers (in-process) — chat only |
| `llm/llama_nemotron/` | `llama_nemotron_llm_server` | 8106 | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | HF transformers + `lm-format-enforcer` — native tool calls, reasoning toggle |
| `llm/nemotron3_nano/` | `nemotron3_nano_llm_server` | 8107 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | vLLM (execvp shim) — Blackwell FP4 MoE |

- **HTTP contract is identical** across all three (OpenAI-compatible
  `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`). Workers point
  at a different port to swap backends — no worker-side code changes.
- **Ports** chosen to be non-overlapping so two LLM backends can coexist in
  the same stack if a sample actually wants that (unusual; typically pick one).
  8101 slot is kept for Mistral-Minitron to preserve wire compatibility with
  the previous `llm-server` deployment.
- **`llama_nemotron`** adds grammar-constrained tool-call decoding via
  `lm-format-enforcer`. When `tools=[...]` is present in the request, a
  `UnionParser([tool_call_grammar, free_text])` is fed as
  `prefix_allowed_tokens_fn` so the model's vocabulary is masked every step to
  either valid `<TOOLCALL>[{...}]</TOOLCALL>` JSON or plain assistant text.
  Rationale: the native Llama-3.1 chat template instructs the model to emit
  tool calls as JSON, but sampling noise / schema drift can still produce
  syntactically broken output. LMFE eliminates that entirely.
- **`nemotron3_nano`** is intentionally thin (~200 lines). vLLM already
  exposes the OpenAI API, parses Nemotron-3-Nano's XML tool-call format via
  `--tool-call-parser qwen3_coder`, and splits the `<think>…</think>` preamble
  via `--reasoning-parser nano_v3` (custom plugin auto-fetched from the model
  card into `model_cache`). The shim reads the YAML, sets
  `VLLM_USE_FLASHINFER_MOE_FP4=1`, and `os.execvp`s into `vllm serve` so the
  launcher's signals go straight to vLLM with no intermediate wiring.
- **`enforce_eager: true`** is the default for `nemotron3_nano` — CUDA graph
  capture plus FlashInfer FP4 MoE autotune are silent and take 3–8 min on
  first run, which is a bad UX for a voice agent waiting to become healthy.
  Eager mode starts in ~5 s after weight load and is 10–20% slower per token
  (imperceptible at <250 tokens/turn where STT+VAD+TTS already dominate).

Dependency fan-out stays contained: only `llama_nemotron` pulls
`lm-format-enforcer`, only `nemotron3_nano` pulls `vllm>=0.12.0`. Mistral
Minitron remains a plain FastAPI + transformers box.

### 2026-04-28 — llm-server added (pure-text LLM)

Fourth AI inference server added under `ai-services/llm-server/`, filling the
pure-text LLM gap alongside the existing VLM / STT / TTS servers.

- **Model**: `nvidia/Mistral-NeMo-Minitron-8B-Instruct` (HuggingFace causal LM,
  ~16 GB VRAM at BF16) loaded in-process via `AutoModelForCausalLM` +
  `AutoTokenizer`. Any `AutoModelForCausalLM`-compatible HF model works — swap
  by editing `llm_server.yaml` (`model`, `max_new_tokens`, `dtype`, `stop`).
- **API**: OpenAI-compatible `GET /health`, `GET /v1/models`,
  `POST /v1/chat/completions` on default port **8101** (avoiding the VLM 8100
  and the STT/TTS 8103/8104 slots). Pure-text messages only — multi-modal
  content blocks are flattened to text (non-`text` blocks are dropped).
- **No strict Pydantic schema** on the request body — the endpoint parses
  raw JSON. This sidesteps FastAPI/Pydantic v2 ForwardRef issues with
  closure-defined models and tolerates the many optional fields OpenAI
  clients send (`n`, `frequency_penalty`, `seed`, `reasoning_effort`, …).
- **Stop handling**: the request's `stop` list is honored; if absent, the
  YAML's `stop` default is used. For Minitron-8B this defaults to
  `["<extra_id_1>", "<extra_id_0>"]` so chat-template tokens don't leak
  into replies. A `StringStopCriteria` hook also halts generation mid-stream
  once a stop string appears in decoded output.
- **Threading**: blocking `generate()` runs in the default thread pool
  executor via `loop.run_in_executor` so the asyncio loop is never blocked.
  The backend is loaded lazily under a lock (warmed at startup from `_run`).
- **No continuous batching.** Single-user voice-agent workloads only; for
  higher concurrency swap in vLLM behind the same HTTP contract.

### 2026-04-27 — MCP example: transcript + video MCP servers; NVENC recording in hub

`agent-samples/mcp-agent/` added as a demonstration of MCP integration with XR data.

**Transcript MCP server** (`agent-mcp-servers/transcript-mcp/`, port 8200):
- Single FastAPI process hosts both the non-MCP HTTP ingest endpoint (`POST /ingest`)
  and the FastMCP tools (`/mcp`) so agents can query historical transcripts.
- Agent workers POST transcripts over plain HTTP; MCP is for LLM tool-use only.
- JSONL storage persists across server restarts; one file per participant.

**Video MCP server** (`agent-mcp-servers/video-mcp/`, port 8210):
- Thin FastMCP wrapper around the hub video HTTP API (`GET /video`).
  Fetches the concatenated H.264 byte stream, writes it to a temp file, returns path.
- Kept separate from the transcript server so either can be used independently.

**Hub NVENC video recording** (`server-runtime/xr_media_hub/video/_recorder.py`):
- Opt-in via `video_recording.enabled: true` in `xr_media_hub.yaml`.
- Uses `PyNvVideoCodec` (on PyPI) for NVENC encoding; included in the standard `uv sync`.
  The config guard (`enabled: true`) prevents instantiation when recording is not needed.
- VBR mode, no B-frames (`bf=0`), `repeat_sps_pps=1`.  Each chunk uses a fresh encoder
  session so it always begins with SPS+PPS+IDR and is independently decodable.
  Chunks are binary-concatenable with `cat`.
- Hub exposes a video query HTTP API on port 8090 (`GET /video?pid=&start_us=&end_us=`).

**PyNvVideoCodec pitfalls (hard-won)**:
- `Encode()` must receive a **2D numpy array** of shape `(H*3//2, W)` — do **not** call
  `.flatten()`.  NVENC reads the array using numpy strides to determine the row pitch.
  A 1D array causes NVENC to assume an internally aligned pitch (e.g. 512 for W=320),
  producing a circular horizontal shift in every decoded frame.
- `GetSequenceParams()` does not exist in PyNvVideoCodec 2.x.  Use `repeat_sps_pps=1`
  in `CreateEncoder` kwargs instead; it prepends SPS+PPS automatically before each IDR.
- WebRTC adaptive bitrate changes the frame resolution mid-stream.  The encoder must be
  recreated (and the current chunk flushed) whenever `width` or `height` changes.  Feeding
  wrong-sized frames to the encoder silently corrupts all subsequent output.
- There is no reliable option that forces repeated IDR frames in PyNvVideoCodec 2.x.
  `gopLength`, `gop`, `idrPeriod` were all tested — NVENC only emits one IDR at the start
  of a session regardless.  Use per-chunk fresh encoders (`EndEncode` → `CreateEncoder`)
  to guarantee IDR boundaries; each new encoder session always begins its output with IDR.

**mcp-agent worker** (`agent-samples/mcp-agent/worker/`):
- Runs continuous STT (same VAD logic as echo-agent).
- POSTs each final utterance to the transcript-mcp-server over HTTP.
- Does not speak TTS — pure observation/logging pipeline.

### 2026-04-24 — AI inference servers added; NVIDIA models; shared model cache

`ai-services/` added as a sibling of `server-runtime/`, containing three reusable
OpenAI-compatible HTTP inference servers.

Model choices — all NVIDIA:
- **vlm-server**: `nvidia/Cosmos-Reason1-7B` in-process via HuggingFace
  transformers (Qwen2.5-VL architecture).  Accepts base64 image_url in messages.
- **stt-server**: `nvidia/parakeet-tdt-0.6b-v3` in-process via NeMo ASR.
  English-only TDT model, CC-BY-4.0.  ~1.5 GB VRAM.
- **tts/magpie**: `nvidia/magpie_tts_multilingual_357m` in-process via NeMo TTS.
  Multilingual, NVIDIA Open Model License.  ~1 GB VRAM.
- **tts/piper**: any rhasspy/piper-voices ONNX voice; ~100 ms/sentence on CPU.

Shared model cache: all weights land in `models/` at the repo root (gitignored).
Each YAML configures `model_cache` (resolved relative to the YAML file) so the
same physical directory is used regardless of which sample root the YAML is in.

Sample YAMLs for all four services ship with `cloudxr-agent` as a template.

OpenAI-compatible APIs chosen so workers never need to know backend details —
swap models by changing the YAML only.

### 2026-04-22 — CloudXR runtime extracted to top-level shared component

`cloudxr-runtime/` added as a peer of `server-runtime/`, wrapping
`isaacteleop[cloudxr]` (NVIDIA IsaacTeleop SDK).  Samples opt-in by adding
`Process("cloudxr", "../../cloudxr-runtime", "cloudxr_runtime")` to their
`PROCESSES` list and providing a `cloudxr_runtime.yaml` in the sample root.

The native CloudXR service runs entirely as a local process (no Docker).
`isaacteleop`'s Python `wss_run()` provides a TLS WebSocket proxy on port 48322
required for `auto-webrtc` profile; `auto-native` does not need it.
CloudXR and the hub are fully independent: CloudXR streams rendered/sim content
to XR devices over WebRTC while the hub handles agent media via LiveKit.

### 2026-04-22 — Launchable convention + StackLauncher

Each runnable sub-project (hub, worker, future CloudXR runtime, MCP servers) is a
**launchable**: an entry-point command + an optional `<command>.yaml` config.
The launcher discovers YAML files automatically by convention — no separate
launcher config file (the previous `stack.toml` idea was dropped).

The orchestrator code declares the process sequence using `Process` + `run_stack`.
All processes start concurrently; startup order does not matter because every
launchable must be resilient to peers not being ready (ZMQ reconnects, etc.).
`run_stack` is fail-fast: any process exit terminates the whole stack.

`launcher/` gained `Process`, `StackLauncher`, and `run_stack` (all stdlib-only).
`HubLauncher` / `ProjectLauncher` remain as lower-level building blocks.

### 2026-04-21 — Agent-SDK extracted; samples use orchestrator + worker subprocess model

`agent-sdk/` (`xr-ai-agent`) was extracted as a standalone package with only
`pyzmq` + `msgpack` as runtime dependencies. The four IPC client modules
(`_types`, `_codec`, `_shm`, `_processor`) moved there from `server-runtime`.
`server-runtime/xr_media_hub/ipc/__init__` re-exports everything for backwards compat.

Each sample now has two entry points:
- **Orchestrator** (`<name>`): stdlib + `xr-ai-launcher` only. Uses `HubLauncher`
  (which runs the hub via `uv run --project server-runtime`) and `ProjectLauncher`
  (which runs the worker via `uv run --project .`). Waits for the worker to exit.
- **Worker** (`<name>_worker`): imports only from `xr_ai_agent`. Contains all
  agent logic. Launched as a subprocess by the orchestrator.

`launcher/` gained `ProjectLauncher` — a generic context manager that runs any
uv project command as a managed subprocess in its own isolated venv, yielding
the `asyncio.subprocess.Process` for lifecycle control.

**Why:** complete venv isolation between hub (server-runtime), agent (sample), and
orchestrator (launcher-only). No cross-contamination of server deps into agent
venvs and vice versa. `uv run --project` is the mechanism — uv resolves and caches
each project's venv independently.

### 2026-04-21 — VLM agent sample added

`agent-samples/vlm-agent/` — answers natural-language queries about live XR
video using a locally-hosted vision-language model.
**Model:** `nvidia/Cosmos-Reason1-7B` (NVIDIA Open Model License + Apache 2.0,
commercial use permitted; ~16 GB VRAM at BF16). Architecture:
`Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` + `qwen-vl-utils`.
**Protocol:** client sends `vlm.query` data message (raw text or
`{"query":"…","track_id":"…"}`); agent replies on `vlm.response`.
**Frame flow:** `on_frame()` tracks latest `FrameSignal` per (participant,
track); on query, `request_frame(signal)` pulls a pixel copy, converts to PIL
via numpy (I420/NV12/RGB24/RGBA/BGRA), then calls `_VlmBackend.infer()` in a
thread pool so the asyncio loop is not blocked. Model is loaded lazily on the
first query. Override model via `VLM_MODEL` env var.

### 2026-04-21 — Process management moved to `launcher/`

`HubLauncher` lives in `launcher/xr_ai_launcher/`, not in `server-runtime`.
**Why:** process management should not be part of the processes it manages.
The launcher will eventually start MCP servers, CloudXR runtime, and other
components — keeping it separate keeps dependency chains lean and the boundary
clean. `launcher/` has zero runtime dependencies (stdlib only).

### 2026-04-21 — NVDEC/NVENC required; OpenH264 must not be used

`LiveKitConnector.start()` calls `require_nvidia_video_codecs()` before doing
anything else. It checks for `libnvcuvid.so` (NVDEC) and `libnvidia-encode.so`
(NVENC) via ctypes and raises `RuntimeError` if either is absent (Linux only).
**Why:** `livekit-rtc` bundles `libwebrtc` which includes OpenH264 as a software
fallback. OpenH264 is royalty-bearing for end users and must not ship in this
product. The guard prevents silent fallback at the cost of a hard startup failure.
In Docker: `--gpus all` or `--device /dev/nvidia*` must be passed.

### 2026-04-21 — Video frame delivery: metadata push, pixel pull

Processors receive `FrameSignal` metadata at full frame rate via `on_frame()`.
Pixel data is only copied when the processor calls `await ep.request_frame(signal)`.
The hub holds one SHM slot per (participant, track) — always the latest frame.
The slot stays `_STATE_READY` (not released to the connector) until the next frame
arrives for the same track, so `bytes(view.data)` in FRAME_REQUEST is safe.
**Why:** avoids copying every frame over IPC; agents sample at their own rate.
Concurrent `request_frame()` calls for the same track are coalesced into one
FRAME_REQUEST; all waiters receive the same FRAME_DATA response.

### 2026-04-21 — `AgentEndpoint` + `ConsumerEndpoint` → `ProcessorEndpoint`

`ipc/_agent.py` and `ipc/_consumer.py` are deleted. Both are replaced by a
single `ProcessorEndpoint` in `ipc/_processor.py`.
**Why:** `ConsumerEndpoint` was unused scaffolding; `AgentEndpoint` was too
narrow a name (the endpoint suits analytics, recording, etc. — not just agents).
`ProcessorEndpoint` auto-maintains `connected_participants: frozenset[str]` so
processors always know who is present without manual event tracking.

### 2026-04-21 — Agent return path through hub

Agents push `RETURN_DATA`/`RETURN_AUDIO` on the hub's PULL socket.
The hub's `_dispatch` routes them to `send_return_data`/`send_return_audio`,
which PUBs them on `return_data.<pid>` / `return_audio.<pid>` topics.
The `ConnectorEndpoint` SUBs these topics and calls registered callbacks
→ `RoomClient` → LiveKit → client.
**Why:** closes the loop so agents can send audio and data back to participants.

### 2026-04-21 — Echo-agent sample added

`agent-samples/echo-agent/` — echoes audio back to the originating participant
and sends a JSON stats ping (`topic="agent.stats"`) every 5 s to each
connected participant. Demonstrates `ProcessorEndpoint` usage end-to-end.

### 2026-04-20 — Track task management keyed by track SID

`RoomClient._track_tasks` changed from `list[Task]` to `dict[str, Task]`
keyed by track SID. A `track_unsubscribed` handler cancels the exact task.
**Why:** without this, stop/start camera caused a new streaming task to start
while the old one kept running, doubling (then tripling) fps counts.

### 2026-04-20 — Audio format: float32 on the wire, int16 in LiveKit

LiveKit delivers audio as int16 PCM. The hub's IPC layer (`AudioChunk`) uses
float32 LE interleaved. Conversion happens in `_room_client.py`:
- Inbound: `int16 / 32768.0 → float32`
- Outbound (return audio): `clip(float32, -1, 1) * 32767 → int16`

### 2026-04-20 — `xr_media_hub.yaml` config file

Flat YAML at repo root. Fields map 1:1 to `LiveKitConnectorConfig` dataclass.
Relative paths (e.g. `web_client_dir`) resolve relative to the YAML file's
own directory, not CWD. `HubLauncher` searches upward from CWD to find it
automatically.
