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
can call them with any OpenAI SDK client or plain `httpx` / `requests`.
Reference services cover vision-language reasoning, speech recognition,
text-to-speech, and large language models. Three LLM backends ship
side-by-side under `ai-services/llm/` — pick one per sample based on the
tool-calling / reasoning / hardware trade-offs documented below.

| Server | Command | Port | Model | Backend |
|---|---|---|---|---|
| `ai-services/vlm-server/` | `vlm_server` | 8100 | Cosmos-Reason1-7B | vLLM (pip or docker) |
| `ai-services/stt-server/` | `stt_server` | 8103 | parakeet-tdt-0.6b-v3 | NeMo ASR in-process |
| `ai-services/tts/magpie/` | `magpie_tts_server` | 8104 | magpie_tts_multilingual_357m | NeMo TTS in-process |
| `ai-services/tts/piper/` | `piper_tts_server` | 8105 | rhasspy/piper-voices (ONNX) | piper-tts in-process |
| `ai-services/llm/llama_nemotron/` | `llama_nemotron_llm_server` | 8106 | Llama-3.1-Nemotron-Nano-8B-v1 | vLLM (pip or docker) |
| `ai-services/llm/nemotron3_nano/` | `nemotron3_nano_llm_server` | 8107 | NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8} | vLLM (pip or docker) |
| `ai-services/llm/nemotron_omni/` | `nemotron_omni_llm_server` | 8108 | Nemotron-3-Nano-Omni-30B-A3B-Reasoning (NVFP4 / FP8 / BF16, GPU-selected) | vLLM (pip or docker) — multimodal (text + video) |
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

The agent samples in this repo (`simple-vlm-example` and `xr-render-demo`)
default to Piper TTS — it runs on CPU with ~100 ms/sentence latency and avoids
the NeMo dep tree. Magpie is still a supported NVIDIA TTS option with better
voice quality and multilingual support when GPU is available; swap the
`Process` row and YAML.

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

## Calling these from a worker

Workers do not hand-roll `httpx` clients against these endpoints.  They
depend on [`agent-sdk/xr-ai-models`](../agent-sdk/xr-ai-models/README.md),
load a per-sample `yaml/models.yaml`, and construct service clients via
`make_llm` / `make_vlm` / `make_stt` / `make_tts`.  The SDK encapsulates the
OpenAI-compatible wire format and the per-model quirks (reasoning-field
aliasing, `chat_template_kwargs`, served-model-name strings) so callers
never branch on backend.

```python
from xr_ai_models import load_models_config, make_llm, ChatMessage

config = load_models_config("yaml/models.yaml")
async with make_llm(config, "agent_llm") as llm:
    resp = await llm.chat(
        [ChatMessage(role="user", content="hello")],
        max_tokens=128,
        enable_thinking=True,
    )
    print(resp.content, resp.reasoning)
```

A matching `models.yaml` for the four built-in service backends:

```yaml
agent_llm:
  kind:     preset:nemotron3_nano
  base_url: http://localhost:8107

vlm:
  kind:     preset:cosmos_vlm
  base_url: http://localhost:8100

stt:
  kind:     preset:parakeet_stt
  base_url: http://localhost:8103

tts:
  kind:     preset:piper_tts
  base_url: http://localhost:8105
```

Swapping a backend is a `kind:` + `base_url:` edit in YAML; worker code does
not change.  Full protocol surface, the preset table, and the explicit
(no-preset) spec are in
[`agent-sdk/xr-ai-models/README.md`](../agent-sdk/xr-ai-models/README.md).

## Hosting models on NVIDIA NIM

The LLM and VLM can run on [NVIDIA NIM](https://build.nvidia.com) instead of
local vLLM — NIM exposes the same OpenAI-compatible `/v1/chat/completions`
API, so this is a `models.yaml` change with no worker code edits. STT and TTS
stay local: hosted NIM speech (Riva) is not OpenAI `/v1/audio`-compatible.

A NIM model entry differs from a local one in three fields:

```yaml
vlm:
  kind:        openai_compat
  category:    vlm
  base_url:    https://integrate.api.nvidia.com   # client appends /v1/...
  model_name:  nvidia/cosmos-reason1-7b           # confirm slug at build.nvidia.com
  api_key_env: NGC_API_KEY                         # → Authorization: Bearer
  health_check: false                              # hosted NIM has no /health
  capabilities: { vision: true, streaming: true }
```

- **`api_key_env: NGC_API_KEY`** sends the key as a bearer token. The key is a
  managed credential — `run_stack` injects a saved `NGC_API_KEY` into every
  subprocess (see [`docs/credentials.md`](credentials.md)); or export it.
- **`health_check: false`** is required for hosted endpoints — they have no
  local `/health` route, so the worker readiness gate must not probe them.
  (Default is `true` for local servers.)
- **`model_name`** is the hosted model id from [build.nvidia.com](https://build.nvidia.com).

Each sample ships a ready-made `yaml/models.nim.yaml` overlay, selected by a
**single key** — no `main.py` edits. To switch a sample to NIM:

1. Set `model_backend: nim` in the sample's `*_worker.yaml` (default
   `local`). The worker then loads `models.nim.yaml`, and the orchestrator
   (which reads the same key) skips the local model server(s) NIM replaces —
   for xr-render-demo it also points `vlm-mcp` at
   `yaml/vlm_mcp_server.nim.yaml`.
2. Provide `NGC_API_KEY` — in NIM mode the orchestrator prompts for it once
   if it isn't already saved or exported.
3. For xr-render-demo, run the demo without the local `llm` / `agent-llm` /
   `vlm` model-servers (they're `launch_mode="reuse"`, so just don't start
   them in the model-servers stack).

Set `model_backend: local` to switch back. (The orchestrator reads
`model_backend` from the worker YAML with a stdlib regex, so it stays
pyyaml-free.)

**Self-hosted NIM containers** work the same way — point `base_url` at the
container (e.g. `http://localhost:8000`) and set `health_check: true` if it
exposes `/v1/health`.

## vLLM model persistence

The persistent vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`) **survive stack restarts by design**.
`nemotron_omni_llm_server` is foreground (dies with the wrapper). Each
persistent wrapper script checks its health endpoint before spawning vLLM:

- **Already running** → touch the ready file immediately, then idle. Stack is
  ready in seconds; no model reload.
- **Not running** → spawn vLLM normally, wait for `/health`, touch ready file.

In pip mode, vLLM is spawned with `start_new_session=True` so the launcher's
`killpg()` does not reach it on shutdown. In docker mode, the container is
launched detached (`docker run -d --name xr-ai-vllm-<service>`) so it
similarly outlives the wrapper. Either way the wrapper exits cleanly and
vLLM keeps running.

**Stopping the persisted servers** — run from the sample directory:

```bash
uv run xr_render_demo --stop
```

This hits each model server's `/health` endpoint, then either runs
`docker stop <container_name>` (docker-mode servers) or finds the listening
PID via `ss`/`lsof` and sends `SIGTERM` (pip-mode), escalating to
`docker kill` / `SIGKILL` after 20 s. It is safe to run while the stack is
down — processes/containers that are not running are silently skipped.

The target ports and container names are defined in `_PERSISTENT_SERVERS` in
`main.py` and match the defaults in the per-profile YAML files. Update that
list if you change the port or container name.

## Choosing the vLLM runtime (pip vs Docker)

All four vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`, `nemotron_omni_llm_server`) accept a
`vllm_backend:` key in their YAML to pick how vLLM is hosted:

| `vllm_backend` | Runtime | Default | Use when |
|---|---|---|---|
| `pip` | `vllm serve` from the wrapper's venv | yes | Standard development; fastest iteration; works offline once weights are cached. |
| `docker` | `docker run nvcr.io/nvidia/vllm:<tag> vllm serve …` | no | Trying NVIDIA's optimized vLLM container; pinning a specific NGC release; reproducing a deployment image. |

Both modes honor identical config keys — same model, same port, same vLLM
flags. The dispatcher lives in `utils/xr-ai-vllm/`. Switching is one YAML edit:

```yaml
vllm_backend: docker
vllm_image:   nvcr.io/nvidia/vllm:26.04-py3
```

`vllm_image:` defaults to `nvcr.io/nvidia/vllm:26.04-py3`; override to pin
another tag, an internal mirror, or a custom build.

### docker mode — prerequisites

- **Docker Engine** with the user in the `docker` group (`docker version`
  must succeed without `sudo`).
- **NVIDIA Container Toolkit** so `--gpus` works:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- **NGC pull access** for `nvcr.io/nvidia/vllm`. The wrapper auto-runs
  `docker login nvcr.io` if `NGC_API_KEY` is in the environment (loaded by
  `load_credentials()` from `~/.config/xr-ai/credentials.json` per
  [`docs/credentials.md`](credentials.md)). Otherwise log in manually once:

  ```bash
  docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
  ```

Existing `~/.docker/config.json` entries take priority and are not overwritten.

### docker mode — runtime details

- Container is launched with `--network host --ipc host --runtime nvidia`
  (forwarding `NVIDIA_VISIBLE_DEVICES`). The nvidia runtime is used instead of
  the legacy `--gpus` flag so the launch works under both legacy and CDI
  container-toolkit modes; `--ipc host` gives vLLM the shared-memory region
  its workers expect.
- The host `model_cache` is bind-mounted at the same path inside the
  container and `HF_HOME` is set to it, so weights cached by pip mode are
  reused by docker mode and vice versa.
- Container name is deterministic per service: `xr-ai-vllm-vlm-server`,
  `xr-ai-vllm-llama-nemotron-llm-server`,
  `xr-ai-vllm-nemotron3-nano-llm-server`,
  `xr-ai-vllm-nemotron-omni-llm-server`.
- Persistence parity: `vlm_server`, `llama_nemotron_llm_server`, and
  `nemotron3_nano_llm_server` run detached (`docker run -d --rm --name …`) so
  the container survives stack restarts, mirroring their pip-mode
  `start_new_session=True` behavior. `nemotron_omni_llm_server` runs
  foreground (container exits with the wrapper) — same as its pip-mode
  semantics.

### Cleanup

`uv run xr_render_demo --stop` works for both modes. The cleanup path probes
`/health` first; for docker mode it then runs `docker stop <container_name>`
(escalating to `docker kill` after 20 s); for pip mode it falls back to the
port → PID → SIGTERM/SIGKILL path. Same UX for both.

## Per-server notes

- **vlm-server** is a thin launcher around `vllm serve` for Cosmos-Reason1-7B
  (or any Qwen2.5-VL-compatible VLM). vLLM handles weight loading, image
  decoding, and the OpenAI-compatible HTTP API. Hosting backend is selectable
  per YAML — see *Choosing the vLLM runtime* above.
- **llm/llama_nemotron** is a thin wrapper around `vllm serve` for
  `Llama-3.1-Nemotron-Nano-8B-v1`. vLLM handles native Llama-3.1 tool calling
  via the `llama3_json` parser — `tools=[...]` in the request is rendered via
  the model's chat template and the resulting tool calls come back in OpenAI
  wire format (`finish_reason: "tool_calls"`). Per-turn reasoning toggle via
  `"detailed thinking on"` / `"detailed thinking off"` in a system or user
  message; reasoning preamble is **not** stripped server-side. Hosting backend
  is selectable per YAML (see *Choosing the vLLM runtime*). See
  [`ai-services/llm/llama_nemotron/README.md`](../ai-services/llm/llama_nemotron/README.md)
  for the full HTTP contract and tuning knobs.
- **llm/nemotron3_nano** is a thin wrapper around `vllm serve` for
  `NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8}` (auto-selected by GPU compute
  capability). vLLM handles tool calling (`qwen3_coder` parser), reasoning
  extraction (`nano_v3` parser — auto-fetched into `model_cache`), and
  FlashInfer FP4 MoE kernels. Requires a Blackwell-class GPU (B200 / RTX PRO
  6000) for native FP4; swap to FP8 / BF16 variants for Hopper/Ampere.
  `enforce_eager: true` by default to avoid the silent 3–8 min CUDA graph +
  FlashInfer autotune on cold start. Hosting backend is selectable per YAML
  (see *Choosing the vLLM runtime*). See
  [`ai-services/llm/nemotron3_nano/README.md`](../ai-services/llm/nemotron3_nano/README.md)
  for the vLLM flags it forwards and Blackwell prerequisites.
- **llm/nemotron_omni** is a vLLM-backed multimodal LLM serving
  `Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (text + video input) at port 8108.
  The YAML auto-selects between three model variants by detected GPU compute
  capability: NVFP4 on Blackwell (SM100+), FP8 on Ada/Hopper, BF16 forced via
  `use_bf16: true` for highest quality at the largest VRAM cost. Same
  OpenAI-compatible HTTP contract as the other LLM servers — swap the port to
  swap backends. Hosting backend is selectable per YAML (see *Choosing the
  vLLM runtime*); runs foreground in both pip and docker modes (no
  cross-restart persistence).
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
