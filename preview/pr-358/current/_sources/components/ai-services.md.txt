<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# AI inference servers

Read this when calling or operating an inference server. For the
orchestrator pattern that wires servers into a sample, refer to
{doc}`/guides/adding-a-sample`.

Multiple reusable HTTP servers are available as launchable peers of
`services/xr-media-hub/`. All expose an OpenAI-compatible REST API so agent workers
can call them with any OpenAI SDK client or plain `httpx` or `requests`.
Reference services cover vision-language reasoning, speech recognition,
text-to-speech, embeddings, and large language models. The projects are direct
children of the [services source index](https://github.com/NVIDIA/xr-ai/blob/main/services/README.md);
pick an LLM per sample based on the tool-calling, reasoning, and hardware
trade-offs documented below.

| Server | Command | Port | Model | Backend |
|---|---|---|---|---|
| `services/vlm-server/` | `vlm_server` | 8100 | Cosmos-Reason1-7B | vLLM (pip or docker) |
| `services/stt-server/` | `stt_server` | 8103 | parakeet-tdt-0.6b-v3 | NeMo ASR in-process |
| `services/magpie-tts/` | `magpie_tts_server` | 8104 | magpie_tts_multilingual_357m | NeMo TTS in-process |
| `services/piper-tts/` | `piper_tts_server` | 8105 | rhasspy/piper-voices (ONNX) | piper-tts in-process |
| `services/llama-nemotron-llm/` | `llama_nemotron_llm_server` | 8106 | Llama-3.1-Nemotron-Nano-8B-v1 | vLLM (pip or docker) |
| `services/nemotron3-nano-llm/` | `nemotron3_nano_llm_server` | 8107 | NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8} | vLLM (pip or docker) |
| `services/nemotron-omni-llm/` | `nemotron_omni_llm_server` | 8108 | Nemotron-3-Nano-Omni-30B-A3B-Reasoning (NVFP4, FP8, or BF16, GPU-selected) | vLLM (pip or docker) — multimodal (text + video) |
| `services/embedding-server/` | `embedding_server` | 8109 | llama-nemotron-embed-1b-v2 | vLLM (pip or docker) |
| `services/video-memory-service/` | `video_memory_service` | 8310 | — | Typed recorded-video capability |
| `services/rag-service/` | `rag_service` | 8340 | — | Typed dense document retrieval capability |

All model weights land in the service's `model_cache` directory, set per YAML
and resolved relative to the YAML file (every `models/` tree is excluded from
version control). The model-servers profiles share `models/` at the
repository root; the exact layout per launch style is below.

## Two HuggingFace cache roots

The servers use two different `HF_HOME` values, so HuggingFace weights live in
two separate trees under the service's resolved `model_cache`:

| Consumer | `HF_HOME` | Hub cache |
|---|---|---|
| vLLM-backed servers (pip and docker) | `<model_cache>/` | `<model_cache>/hub/` |
| STT and Magpie TTS (NeMo host processes) | `<model_cache>/huggingface/` | `<model_cache>/huggingface/hub/` |

(The NeMo servers additionally cache non-HF artifacts under `<model_cache>/nemo/`.)

`model_cache` itself is set per YAML and resolved relative to the YAML file.
Both the model-servers profiles and the standalone service YAMLs resolve it to
`models/` at the repository root. For a manual `hf download`, set `HF_HOME` to
match the consumer:

```bash
# vLLM-served model, launched via a model-servers profile
# (model_cache resolves to models/ at the repository root):
HF_HOME=models hf download nvidia/Cosmos-Reason1-7B

# NeMo STT server launched from its standalone YAML:
HF_HOME=models/huggingface hf download nvidia/parakeet-tdt-0.6b-v3
```


## Adding a server to a sample

**1 — Add the process to the orchestrator:**

```python
PROCESSES = [
    Process("hub",    "../../services/xr-media-hub",                    "xr_media_hub"),
    Process("vlm",    "../../services/vlm-server",               "vlm_server"),   # ← add as needed
    # Pick ONE LLM backend per sample — they bind different default ports
    # (8106 / 8107) so running more than one at once is allowed but
    # usually unnecessary.
    Process("llm",    "../../services/llama-nemotron-llm",       "llama_nemotron_llm_server"),
    # Process("llm",  "../../services/nemotron3-nano-llm",       "nemotron3_nano_llm_server"),
    Process("stt",    "../../services/stt-server",               "stt_server"),
    # Add these together when the application uses native document retrieval.
    Process("embedding", "../../services/embedding-server",      "embedding_server"),
    Process("rag",    "../../services/rag-service",               "rag_service",
            config="yaml/rag_service.yaml"),
    # Pick one TTS server
    Process("tts",    "../../services/piper-tts",                 "piper_tts_server"),
    # Process("tts",  "../../services/magpie-tts",                "magpie_tts_server"),
    Process("worker", "worker",                                   "my_agent_worker"),
]
```

The agent samples in this repository (`simple-vlm-example` and `xr-render-demo`)
default to Piper TTS — it runs on CPU with ~100 ms/sentence latency and avoids
the NeMo dep tree. Magpie is still a supported NVIDIA TTS option with better
voice quality and multilingual support when GPU is available; swap the
`Process` row and YAML.

**2 — Copy the reference YAML to your sample's `yaml/` directory:**

```bash
mkdir -p yaml
cp ../../services/vlm-server/vlm_server.yaml ./yaml/vlm_server.yaml
# Pick ONE LLM YAML — copy the one matching the Process you picked above.
cp ../../services/llama-nemotron-llm/llama_nemotron_llm_server.yaml ./yaml/llama_nemotron_llm_server.yaml
# cp ../../services/nemotron3-nano-llm/nemotron3_nano_llm_server.yaml ./yaml/nemotron3_nano_llm_server.yaml
cp ../../services/stt-server/stt_server.yaml ./yaml/stt_server.yaml
cp ../../services/embedding-server/embedding_server.yaml ./yaml/embedding_server.yaml
cp ../../services/rag-service/rag_service.yaml ./yaml/rag_service.yaml
cp ../../services/piper-tts/piper_tts_server.yaml ./yaml/piper_tts_server.yaml
# Or for Magpie (multilingual, GPU, ~2-5 s/sentence):
cp ../../services/magpie-tts/magpie_tts_server.yaml ./yaml/magpie_tts_server.yaml
cp ../../services/video-memory-service/video_memory_service.yaml ./yaml/video_memory_service.yaml
```

The standalone model-service YAMLs contain `model_cache: ../../models` for their
original `services/<project>/` location. After copying them one level deeper
into `agent-samples/<name>/yaml/`, change that value to `../../../models` so
the cache still resolves to the repository-root `models/` directory. Capability
and capability configurations without a `model_cache` key need no change.

Edit the YAML as needed (model, port, device, etc.). The launcher auto-discovers
`yaml/<command>.yaml` in the sample root and passes it as `--config`.
For RAG, also point `rag_service.yaml` at an application-owned document
directory and a model profile containing an `embedding` role.

## Calling these from a worker

Workers do not hand-roll `httpx` clients against these endpoints.  They
depend on [`agent-sdk/xr-ai-models`](https://github.com/NVIDIA/xr-ai/blob/main/agent-sdk/xr-ai-models/README.md),
load a per-sample model profile, and construct service clients via
`make_llm`, `make_vlm`, `make_stt`, `make_tts`, and `make_embedding`. The SDK encapsulates the
OpenAI-compatible wire format and the per-model quirks (reasoning-field
aliasing, `chat_template_kwargs`, served-model-name strings) so callers
never branch on backend.

```python
from xr_ai_models import load_models_config, make_llm, ChatMessage

config = load_models_config("yaml/models.local.json")
async with make_llm(config, "agent_llm") as llm:
    resp = await llm.chat(
        [ChatMessage(role="user", content="hello")],
        max_tokens=128,
        enable_thinking=True,
    )
    print(resp.content, resp.reasoning)
```

A model profile separates adapter behavior, endpoint connectivity, and
deployment ownership:

```json
{
  "models": {
    "agent_llm": {
      "category": "llm",
      "adapter": {"preset": "nemotron3_nano"},
      "endpoint": {"base_url": "http://localhost:8107", "readiness": "health"},
      "deployment": {"ownership": "reused", "service": "agent-llm"}
    }
  }
}
```

The worker-side `xr-ai-models` loader accepts JSON or YAML, including flat
legacy entries and direct role mappings. A profile shared with the stdlib-only
launcher must use the wrapped nested `.json` contract; non-`.json` profiles are
rejected before deployment metadata is read. Full protocol surface, the preset
table, and the profile contract are in
[`agent-sdk/xr-ai-models/README.md`](https://github.com/NVIDIA/xr-ai/blob/main/agent-sdk/xr-ai-models/README.md).

## Hosting models on NVIDIA NIM

The LLM and VLM can run on [NVIDIA NIM](https://build.nvidia.com) instead of
local vLLM — NIM exposes the same OpenAI-compatible `/v1/chat/completions`
API, so this is a model-profile change with no worker code edits. STT and TTS
stay local: hosted NIM speech (Riva) is not OpenAI `/v1/audio`-compatible.

A hosted entry uses an environment-variable reference for its credential,
disables endpoint health probing, and declares external ownership:

```json
{
  "models": {
    "vlm": {
      "category": "vlm",
      "adapter": {
        "kind": "openai_compat",
        "model_name": "nvidia/cosmos-reason1-7b",
        "capabilities": {"vision": true, "streaming": true}
      },
      "endpoint": {
        "base_url": "https://integrate.api.nvidia.com",
        "api_key_env": "NGC_API_KEY",
        "readiness": "none"
      },
      "deployment": {"ownership": "external"}
    }
  }
}
```

- **`api_key_env: NGC_API_KEY`** sends the environment value as a bearer
  token. The key is a
  managed credential — `run_stack` injects a saved `NGC_API_KEY` into every
  subprocess (refer to [`docs/source/getting_started/credentials.md`](https://github.com/NVIDIA/xr-ai/blob/main/docs/source/getting_started/credentials.md)); or export it.
- **`readiness: none`** is required when the hosted endpoint has no local
  `/health` route.
- **`ownership: external`** keeps the launcher from starting or stopping the
  hosted service.
- **`model_name`** is the hosted model id from [build.nvidia.com](https://build.nvidia.com).

For `simple-vlm-example`, set `models_config: models.hosted.json` in the worker
YAML. This wrapped JSON profile is consumed by both the worker and orchestrator,
so the local VLM process is omitted and `NGC_API_KEY` is requested
automatically. Select `models.local.json` to switch back.

`xr-render-demo` retains its `model_backend: nim` selector and
`models.nim.yaml` overlay. Run it without the local `agent-llm` / `vlm`
model-servers and provide `NGC_API_KEY`.

**Self-hosted NIM containers** work the same way: point `base_url` at the
container (e.g. `http://localhost:8000`), set `readiness: health`, and choose
`managed` or `reused` ownership if the launcher owns that service. Legacy flat
profiles may continue to set `health_check: true` when it
exposes `/v1/health`.

## vLLM model persistence

The persistent vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`, `nemotron_omni_llm_server`, `embedding_server`)
**survive stack restarts by design**. Each persistent wrapper script checks its
health endpoint before spawning vLLM:

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

Cleanup locates labelled Docker containers before inspecting ports, then
stops them with `docker stop` (escalating to `docker kill` after 20 s).
Pip-mode processes must carry the `XR_AI_VLLM_MANAGED` and
`XR_AI_VLLM_PORT` ownership markers before cleanup sends `SIGTERM` or
`SIGKILL`. Unknown listeners and failed inspection abort cleanup without
sending a signal; absent servers are silently skipped.

The target ports and container names match the defaults in the per-profile YAML files.

## Choosing the vLLM runtime (pip vs Docker)

All five vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`, `nemotron_omni_llm_server`, `embedding_server`) accept a
`vllm_backend:` key in their YAML to pick how vLLM is hosted:

| `vllm_backend` | Runtime | Default | Use when |
|---|---|---|---|
| `pip` | `vllm serve` from the wrapper's venv | yes | Standard development; fastest iteration; works offline once weights are cached. |
| `docker` | `docker run nvcr.io/nvidia/vllm:<tag> vllm serve …` | no | Trying NVIDIA's optimized vLLM container; pinning a specific NGC release; reproducing a deployment image. |

Both modes honor identical configuration keys — same model, same port, same vLLM
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
- **NVIDIA Container Toolkit** so the `nvidia` runtime can expose GPUs:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- **NGC pull access** for `nvcr.io/nvidia/vllm`. The wrapper auto-runs
  `docker login nvcr.io` if `NGC_API_KEY` is in the environment (loaded by
  `load_credentials()` from `~/.config/xr-ai/credentials.json` per
  [`docs/source/getting_started/credentials.md`](https://github.com/NVIDIA/xr-ai/blob/main/docs/source/getting_started/credentials.md)). Otherwise, log in manually once:

  ```bash
  docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
  ```

Existing `~/.docker/config.json` entries take priority and are not overwritten.

### docker mode — runtime details

- Container is launched with `--network host --ipc host --runtime nvidia`
  (forwarding `NVIDIA_VISIBLE_DEVICES`), and `/bin/bash` overrides the image
  entrypoint so setup installs run before `vllm serve`.
- Failed stopped containers are recreated because Docker cannot change their
  recorded entrypoint or command.
- The host `model_cache` is bind-mounted at the same path inside the
  container and `HF_HOME` is set to it, so weights cached by pip mode are
  reused by docker mode and vice versa.
- Container name is deterministic per service: `xr-ai-vllm-vlm-server`,
  `xr-ai-vllm-llama-nemotron-llm-server`,
  `xr-ai-vllm-nemotron3-nano-llm-server`,
  `xr-ai-vllm-nemotron-omni-llm-server`.
- Persistence parity: `vlm_server`, `llama_nemotron_llm_server`,
  `nemotron3_nano_llm_server`, and `nemotron_omni_llm_server` launch their
  Docker processes in separate sessions, so they survive launcher shutdowns
  like their pip-mode `start_new_session=True` counterparts.

### Cleanup

`uv run xr_render_demo --stop` works for both modes. Cleanup locates labelled
Docker containers before inspecting ports, then stops them with `docker stop`
(escalating to `docker kill` after 20 s). Pip-mode processes carry an
`xr-ai-vllm` ownership marker; unknown listeners and failed inspection abort
cleanup without sending a signal.

Pip-mode vLLM processes started before the ownership markers were introduced
cannot be identified safely. After upgrading, stop each unmarked process
manually once; subsequent launches include the markers and support managed
cleanup.

## Per-server notes

- **vlm-server** is a thin launcher around `vllm serve` for Cosmos-Reason1-7B
  (or any Qwen2.5-VL-compatible VLM). vLLM handles weight loading, image
  decoding, and the OpenAI-compatible HTTP API. Hosting backend is selectable
  per YAML — refer to *Choosing the vLLM runtime* above.
- **llama-nemotron-llm** is a thin wrapper around `vllm serve` for
  `Llama-3.1-Nemotron-Nano-8B-v1`. vLLM handles native Llama-3.1 tool calling
  via the `llama3_json` parser — `tools=[...]` in the request is rendered via
  the model's chat template and the resulting tool calls come back in OpenAI
  wire format (`finish_reason: "tool_calls"`). Per-turn reasoning toggle via
  `"detailed thinking on"` or `"detailed thinking off"` in a system or user
  message; reasoning preamble is **not** stripped server-side. Hosting backend
  is selectable per YAML (refer to *Choosing the vLLM runtime*). Refer to
  [`services/llama-nemotron-llm/README.md`](https://github.com/NVIDIA/xr-ai/blob/main/services/llama-nemotron-llm/README.md)
  for the full HTTP contract and tuning knobs.
- **nemotron3-nano-llm** is a thin wrapper around `vllm serve` for
  `NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8}` (auto-selected by GPU compute
  capability). vLLM handles tool calling (`qwen3_coder` parser), reasoning
  extraction (`nano_v3` parser — auto-fetched into `model_cache`), and
  FlashInfer FP4 MoE kernels. Requires a Blackwell-class GPU (B200 or RTX PRO
  6000) for native FP4; swap to the FP8 or BF16 variants for Hopper and Ampere.
  `enforce_eager: true` by default to avoid the silent 3–8 min CUDA graph and
  FlashInfer autotune on cold start. Hosting backend is selectable per YAML
  (refer to *Choosing the vLLM runtime*). Refer to
  [`services/nemotron3-nano-llm/README.md`](https://github.com/NVIDIA/xr-ai/blob/main/services/nemotron3-nano-llm/README.md)
  for the vLLM flags it forwards and Blackwell prerequisites.
- **nemotron-omni-llm** is a vLLM-backed multimodal LLM serving
  `Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (text + video input) at port 8108.
  The YAML auto-selects between three model variants by detected GPU compute
  capability: NVFP4 on Blackwell (SM100+), FP8 on Ada and Hopper, BF16 forced via
  `use_bf16: true` for highest quality at the largest VRAM cost. Same
  OpenAI-compatible HTTP contract as the other LLM servers — swap the port to
  swap backends. Hosting backend is selectable per YAML (refer to *Choosing the
  vLLM runtime*); persists across stack restarts in both pip and docker modes.
- **stt-server** loads parakeet-tdt-0.6b-v3 via NeMo ASR in-process.
  English-only; the `language` and `temperature` form fields are accepted but ignored.
- **magpie-tts** loads magpie_tts_multilingual_357m via NeMo TTS in-process.
- **piper-tts** serves any rhasspy/piper-voices ONNX voice; ~100 ms/sentence on CPU.
  All inference runs in a thread pool so the asyncio loop is never blocked.
- **video-memory-service** owns recorded chunk queries, NVDEC, and PNG output
  behind typed msgpack/ZMQ on port 8310. Set `recordings_dir` in its YAML to
  enable recorded-video operations; the path must match the hub's
  `video_recording.out_dir`. Current frames stay with the caller's hub client.
- Ports are configurable — avoid conflicts with LiveKit (7880–7882) and hub (8080, 8090).
- **Sample YAMLs** for each service ship in their own service directory.
  Copy them to your sample's `yaml/` directory and set `model_cache` to
  `../../../models`, which resolves to `xr-ai/models/` from
  `agent-samples/<name>/yaml/`.
