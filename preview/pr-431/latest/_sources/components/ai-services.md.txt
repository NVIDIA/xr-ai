<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# AI inference servers

Read this when calling or operating an inference server. For the
orchestrator pattern that wires servers into a sample, refer to
{doc}`/guides/adding-a-sample`. For an end-to-end procedure covering custom
deployment profiles, hardware YAML, and reuse-only sample configuration, refer to
{doc}`/guides/customizing-model-servers`.

Multiple reusable inference and typed capability services are available as
launchable peers of `services/device-io-hub/`. Model HTTP is encapsulated by
the typed factories in `xr-ai-models`; workers must not add vendor SDKs or
hand-written HTTP clients. Reference services cover vision-language reasoning,
speech recognition, text-to-speech, embeddings, large language models,
recorded video, and document retrieval.

| Server | Command | Port | Model | Backend |
|---|---|---|---|---|
| `services/vlm-server/` | `vlm_server` | 8100 | Cosmos3 Nano Reasoner | vLLM (pip or docker) |
| `services/stt-server/` | `stt_server` | 8103 | parakeet-tdt-0.6b-v3 | NeMo ASR in-process |
| `services/magpie-tts/` | `magpie_tts_server` | 8104 | magpie_tts_multilingual_357m | NeMo TTS in-process |
| `services/piper-tts/` | `piper_tts_server` | 8105 | rhasspy/piper-voices (ONNX) | piper-tts in-process |
| `services/llama-nemotron-llm/` | `llama_nemotron_llm_server` | 8106 | Llama-3.1-Nemotron-Nano-8B-v1 | vLLM (pip or docker) |
| `services/nemotron3-nano-llm/` | `nemotron3_nano_llm_server` | 8107 | NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8} | vLLM (pip or docker) |
| `services/nemotron-omni-llm/` | `nemotron_omni_llm_server` | 8108 | Nemotron-3-Nano-Omni-30B-A3B-Reasoning (NVFP4, FP8, or BF16, GPU-selected) | vLLM (pip or docker) — multimodal (text + video) |
| `services/embedding-server/` | `embedding_server` | 8109 | llama-nemotron-embed-1b-v2 | vLLM (pip or docker) |
| `services/nim-server/` | `nim_server` | configured per YAML | selected NVIDIA NIM | persistent Docker container |
| `services/video-memory-service/` | `video_memory_service` | 8310 | — | Typed recorded-video capability |
| `services/rag-service/` | `rag_service` | 8340 | — | Typed dense document retrieval capability |

Local model artifacts land in the service's `model_cache` directory, set per
YAML and resolved relative to the YAML file. Self-hosted NIM containers use
`nim_cache` for their optimized engines and weights; hosted endpoints keep no
weights in this repository. Every `models/` tree is excluded from version
control. The model-servers profiles share `models/` at the repository root;
the exact layout per launch style is below.

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
HF_HOME=models hf download nvidia/Cosmos3-Nano

# NeMo STT server launched from its standalone YAML:
HF_HOME=models/huggingface hf download nvidia/parakeet-tdt-0.6b-v3
```

(migrating-model-caches-from-ai-services)=

## Migrating model caches from `ai-services/`

The service-directory rename does not move ignored model weights. Existing
installations may still hold VLM, STT, and LLM weights in
`ai-services/models/`, and TTS weights in `ai-services/tts/models/`; all
current service profiles resolve to the repository-root `models/` directory.

Stop model services, then merge the caches from the repository root:

```bash
mkdir -p models
if [ -d ai-services/models ]; then
  cp -a -l -n ai-services/models/. models/
fi
if [ -d ai-services/tts/models ]; then
  cp -a -l -n ai-services/tts/models/. models/
fi
```

The `-l` option avoids duplicating large weight files and requires both paths
to use the same filesystem. Omit `-l` when copying across filesystems and
verify free space first. Keep the old directories until the relocated services
start successfully without network access. Recreate project environments with
`uv sync`; do not copy `.venv` directories across the move.


## Adding a server to a sample

**1 — Add the process to the orchestrator:**

```python
PROCESSES = [
    Process("hub",    "../../services/device-io-hub",                    "device_io_hub"),
    Process("vlm",    "../../services/vlm-server",               "vlm_server",
            config="yaml/vlm_server.yaml"),   # ← add as needed
    # Pick ONE LLM backend per sample — they bind different default ports
    # (8106 or 8107) so running more than one at once is allowed but
    # usually unnecessary.
    Process("llm",    "../../services/llama-nemotron-llm",       "llama_nemotron_llm_server",
            config="yaml/llama_nemotron_llm_server.yaml"),
    # Process("llm",  "../../services/nemotron3-nano-llm",       "nemotron3_nano_llm_server",
    #         config="yaml/nemotron3_nano_llm_server.yaml"),
    Process("stt",    "../../services/stt-server",               "stt_server",
            config="yaml/stt_server.yaml"),
    # Add these together when the application uses native document retrieval.
    Process("embedding", "../../services/embedding-server",      "embedding_server",
            config="yaml/embedding_server.yaml"),
    Process("rag",    "../../services/rag-service",               "rag_service",
            config="yaml/rag_service.yaml"),
    # Pick one TTS server
    Process("tts",    "../../services/piper-tts",                 "piper_tts_server",
            config="yaml/piper_tts_server.yaml"),
    # Process("tts",  "../../services/magpie-tts",                "magpie_tts_server",
    #         config="yaml/magpie_tts_server.yaml"),
    Process("worker", "worker",                                   "my_agent_worker"),
]
```

The agent samples in this repository (`simple-vlm-example`,
`tea-making-sample`, and `xr-render-demo`) support Piper TTS — it runs on CPU
with ~100 ms/sentence latency and avoids the NeMo dep tree. Magpie is still a
supported NVIDIA TTS option with better voice quality and multilingual support
when GPU is available; samples that own TTS can select its process and YAML.

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

`simple-vlm-example` does not copy model-service YAML or launch model servers.
Its fixed `yaml/models.json` points at operator-owned STT, VLM, and TTS
endpoints; the sample orchestrator only launches its hub and worker.

Edit the YAML as needed (model, port, device, etc.). Set every copied path
explicitly with `Process(config=...)`; the launcher does not discover files by
command name.
For RAG, also point `rag_service.yaml` at an application-owned document
directory and a model profile containing an `embedding` role.

## Calling these from a worker

Workers do not hand-roll `httpx` clients against these endpoints.  They
depend on {doc}`/reference/agent-sdk-models`,
load a per-sample model profile, and construct service clients via
`make_llm`, `make_vlm`, `make_stt`, `make_tts`, and `make_embedding`. The SDK encapsulates the
OpenAI-compatible wire format and the per-model quirks (reasoning-field
aliasing, `chat_template_kwargs`, served-model-name strings) so callers
never branch on backend.

```python
from xr_ai_models import load_models_config, make_llm, ChatMessage

config = load_models_config("yaml/models.json")
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
      "adapter": {"preset": "nemotron_omni"},
      "endpoint": {"base_url": "http://localhost:8108", "readiness": "health"},
      "deployment": {"ownership": "reused", "service": "omni"}
    }
  }
}
```

The worker-side `xr-ai-models` loader accepts JSON or YAML, including flat
legacy entries and direct role mappings. A profile shared with the stdlib-only
launcher must use the wrapped nested `.json` contract; non-`.json` profiles are
rejected before deployment metadata is read. Full protocol surface, the preset
table, and the profile contract are in
{doc}`/reference/agent-sdk-models`.

## Hosting models on NVIDIA NIM

The LLM and VLM can run on [NVIDIA NIM](https://build.nvidia.com) instead of
local vLLM — NIM exposes the same OpenAI-compatible `/v1/chat/completions`
API, so this is a model-profile change with no worker code edits. STT and TTS
stay local: hosted NIM speech (Riva) is not OpenAI `/v1/audio`-compatible.
Self-hosted speech NIMs are covered below.

A hosted entry uses an environment-variable reference for its credential,
disables endpoint health probing, and declares external ownership:

```json
{
  "models": {
    "vlm": {
      "category": "vlm",
      "adapter": {
        "kind": "openai_compat",
        "model_name": "nvidia/cosmos3-nano-reasoner",
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
  subprocess (refer to {doc}`/getting_started/credentials`); or export it.
- **`readiness: none`** is required when the hosted endpoint has no local
  `/health` route.
- **`ownership: external`** keeps the launcher from starting or stopping the
  hosted service.
- **`model_name`** is the hosted model id from [build.nvidia.com](https://build.nvidia.com).

To adapt a sample, copy its active model profile, replace the local model entry
with the hosted entry, and point `models_config` in the worker YAML at the new
file. The same wrapped JSON profile is consumed by the worker and orchestrator,
so the orchestrator omits externally owned services and requests the referenced
credential automatically.

### Self-hosted NIM containers (`models.vlm_llm_nim.json`)

Compatible models can be pulled from NGC and served as **optimized NIM
containers on your own GPUs**: same APIs as hosted NIM and no network hop.

The NIM containers are owned by the shared model-servers stack, exactly
like the local vLLM servers. Its deployment profiles pick the mix: every
`managed` entry launches as a `nim_server` process (`services/nim-server`, a
generic wrapper; the per-GPU-profile `nim_<role>_server.yaml` picks the
image and ports) or as a local server:

```bash
uv run --project agent-samples/model-servers model_servers --models vlm_llm_nim
```

- `vlm_llm_nim`: Nemotron-3 Nano Omni and Cosmos3-Nano Reasoner as NIM
  containers, with STT, Piper TTS, and embedding served locally. Samples reuse
  these endpoints; they never launch or stop the containers.

To adapt a sample, copy the relevant `llm` and `vlm` entries from
`models.vlm_llm_nim.json` into the sample's active models JSON and change only
their deployment ownership from `managed` to `reused`. Samples with an
`agent_llm` role duplicate the `llm` entry under that name. The adjacent
`nim_llm_server.yaml` and `nim_vlm_server.yaml` comments repeat this mapping
beside the container configuration.

The container `image:` is the model, so swapping models is a
`nim_<role>_server.yaml` edit plus the matching profile entry. Selection is
per entry, not per profile: each model role independently picks a local
server, a self-hosted NIM container, or a hosted endpoint through its own
`adapter`/`endpoint`/`deployment` sections. The shipped profiles are
presets, not a closed set; a mixed setup is a copy of a shipped profile
with the relevant entries changed, saved under any name and selected with
`--models` (model-servers) or `models_config` (workers). When mixing, mind
port overlaps: give a NIM container a free port or drop the overlapping
local server from the profile.

A custom model-server profile can still launch self-hosted Riva speech NIMs.
Workers reach them through the optional `riva_grpc` model kind:

```yaml
stt:
  kind:      riva_grpc
  category:  stt
  base_url:  localhost:50051   # the container's gRPC port
  language:  en-US
```

TTS additionally takes `voice:` (a Riva voice name) and `sample_rate:`
(default 44100). `health_check: true` (the default) runs a gRPC channel-ready
probe. No shipped model-server profile or sample selects Riva speech.

Requirements: docker + NVIDIA Container Toolkit, `NGC_API_KEY` (used for the
`nvcr.io` image pull *and* by the container itself to download the
GPU-matched optimized engine from NGC on first start; multi-GB, cached
under `models/nim/` for later runs), and GPU capacity for every container.
`cuda_visible_devices` placement lives in the per-GPU-profile
`nim_*_server.yaml` files. Their adjacent comments record profile-specific
validation status and hardware cautions; verify startup and capacity on the
target host. Readiness gates on each container's `/v1/health/ready`.

A NIM container serving something the samples don't ship is the same
mechanism by hand: point an `openai_compat` entry's `base_url` at its port
(its health route is `/v1/health/ready`, so keep `health_check: false` and
let the container gate readiness), or a `riva_grpc` entry at its gRPC port.
With `ownership: external` (you run the container yourself) that is the
whole change. For an orchestrator to launch or expect it, the entry's
`deployment.service` must name a process row in that orchestrator's service
table (`_MODEL_SERVICES` in model-servers, `_MODEL_PROCESSES` in a sample);
a service name with no row fails fast at startup, and adding one row plus
its config YAML is the only orchestrator edit the profile system ever
needs.

## Model-server persistence

The persistent vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`, `nemotron_omni_llm_server`, `embedding_server`)
and self-hosted NIM containers (`nim_server`)
**survive stack restarts by design**, including when a deployment profile
marks them `managed`: the stack starts them, but a clean shutdown leaves them
serving so the next start reuses hot weights. `model_servers --stop` is the
teardown. Switching profiles needs no manual teardown: at startup a wrapper
that finds a *different* persistent xr-ai container holding its port (found
by the `xr-ai-vllm.port=<port>` label) stops and removes it before launching
its own. Each persistent wrapper script checks its
health endpoint before spawning vLLM:

- **Already running with a matching launch fingerprint** → touch the ready
  file immediately, then idle. Stack is ready in seconds; no model reload.
- **Matching container still starting** → attach to its lifecycle and keep
  waiting for `/health` instead of issuing a conflicting second `docker run`.
- **Already running with changed or legacy configuration** → stop, remove, and
  recreate the repository-owned container from the current YAML.
- **Healthy endpoint without the expected running container** → fail without
  stopping the unowned listener.
- **Stopped Docker container with matching launch fingerprint** → restart it,
  wait for `/health`, then touch the ready file.
- **Stopped Docker container with changed or legacy configuration** → remove
  and recreate it from the current YAML before waiting for `/health`.
- **Not running** → spawn vLLM normally, wait for `/health`, then touch the
  ready file.

In pip mode, vLLM is spawned with `start_new_session=True` so the launcher's
`killpg()` does not reach it on shutdown. In docker mode, Docker owns the
container while the foreground `docker run` client uses its own session.
Either way vLLM keeps running after the orchestrator exits.

The STT server follows the same pattern without Docker: `stt_server` spawns
its persistent process with `start_new_session=True`, reuses a healthy server
that survived a previous stack run, and is stopped by the same
`model_servers --stop` cleanup. The CPU Piper server is also launched as a
persistent shared service and is stopped by the same command.

Docker containers carry a fingerprint of their image, GPU assignment, model
cache, environment, bootstrap packages, complete vLLM command, and a versioned
launcher-controlled Docker contract. This prevents a failed container created
by one sample profile—or by older launcher behavior—from being restarted later
with stale memory limits, entrypoint, setup commands, or model arguments.

**Stopping the persisted servers**, from the repo root:

```bash
uv run --project agent-samples/model-servers model_servers --stop
```

Cleanup locates labelled Docker containers before inspecting ports, then
stops them with `docker stop` (escalating to `docker kill` after 20 s).
Locally persisted processes (pip-mode vLLM and Piper) must carry the
`XR_AI_VLLM_MANAGED` and `XR_AI_VLLM_PORT` ownership markers before cleanup
sends `SIGTERM` or `SIGKILL`. Unknown listeners and failed inspection abort
cleanup without sending a signal; absent servers are silently skipped.

The target ports and container names match the defaults in the per-profile YAML files.

## Choosing the vLLM runtime (pip vs Docker)

All five vLLM-backed servers (`vlm_server`, `llama_nemotron_llm_server`,
`nemotron3_nano_llm_server`, `nemotron_omni_llm_server`, `embedding_server`) accept a
`vllm_backend:` key in their YAML to pick how vLLM is hosted:

| `vllm_backend` | Runtime | Code fallback | Shipped standalone YAMLs | Use when |
|---|---|---|---|---|
| `pip` | `vllm serve` from the wrapper's venv | yes | no | Developing the wrapper in its local environment or using a custom pip installation. |
| `docker` | `docker run <vllm_image> vllm serve …` | no | yes | Running the configured vLLM container used by the checked-in configurations. |

Both modes honor identical configuration keys — same model, same port, same vLLM
flags. The dispatcher lives in `utils/xr-ai-vllm/`. Switching is one YAML edit:

```yaml
# vlm-server (Cosmos3)
vllm_backend: docker
vllm_image:   nvcr.io/nvidia/vllm:26.07-py3
```

`vllm_image:` defaults to `nvcr.io/nvidia/vllm:26.07-py3` for vlm-server,
whose Cosmos3 support requires vLLM 0.23 or newer. The other wrappers retain
their `26.04-py3` default. Override either to pin another tag, an internal
mirror, or a custom build.

### docker mode — prerequisites

- **Docker Engine** with the user in the `docker` group (`docker version`
  must succeed without `sudo`).
- **NVIDIA Container Toolkit** so the `nvidia` runtime can expose GPUs:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- **NGC pull access**, when the configured `vllm_image` is restricted or
  requires authentication on `nvcr.io`. The wrapper auto-runs
  `docker login nvcr.io` if `NGC_API_KEY` is in the environment (loaded by
  `load_credentials()` from `~/.config/xr-ai/credentials.json` per
  {doc}`/getting_started/credentials`). Otherwise, log in manually once:

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
  `xr-ai-vllm-nemotron-omni-llm-server`, and
  `xr-ai-vllm-embedding-server`.
- Persistence parity: all five vLLM-backed wrappers launch their Docker
  processes in separate sessions, so they survive launcher shutdowns like
  their pip-mode `start_new_session=True` counterparts.

### Cleanup

`model_servers --stop` works for both modes. Cleanup locates labelled
Docker containers before inspecting ports, then stops them with `docker stop`
(escalating to `docker kill` after 20 s). Pip-mode processes carry an
`xr-ai-vllm` ownership marker; unknown listeners and failed inspection abort
cleanup without sending a signal.

Pip-mode vLLM processes started before the ownership markers were introduced
cannot be identified safely. After upgrading, stop each unmarked process
manually once; subsequent launches include the markers and support managed
cleanup.

## Per-server notes

- **vlm-server** defaults to the Cosmos3 Nano Reasoner. Hugging Face publishes
  Reasoner and Generator weights in the unified `nvidia/Cosmos3-Nano`
  checkpoint. The required `Cosmos3ForConditionalGeneration` architecture
  override selects vLLM's native Reasoner loader: despite its generic class
  name, it maps only the understanding tower and vision encoder and drops the
  Generator weights. The Generator requires vLLM's separate `--omni` path,
  which xr-ai intentionally does not enable. The checkpoint's official chat
  template emits the assistant answer directly and has no `enable_thinking`
  branch or `<think>` delimiters, so the client preset needs no reasoning-field
  mapping. Cosmos-Reason1 remains available by pairing
  `model: nvidia/Cosmos-Reason1-7B` with the `cosmos_vlm` client preset. Hosting
  backend is selectable per YAML — refer to *Choosing the vLLM runtime* above.
- **llama-nemotron-llm** is a thin wrapper around `vllm serve` for
  `Llama-3.1-Nemotron-Nano-8B-v1`. vLLM handles native Llama-3.1 tool calling
  via the `llama3_json` parser — `tools=[...]` in the request is rendered via
  the model's chat template and the resulting tool calls come back in OpenAI
  wire format (`finish_reason: "tool_calls"`). Per-turn reasoning toggle via
  `"detailed thinking on"` or `"detailed thinking off"` in a system or user
  message. Without either phrase, this model reasons by default; the reasoning
  preamble is **not** stripped server-side. To swap checkpoints, change `model`
  only to one with a compatible chat template, and update `tool_call_parser`
  for the replacement model's tool syntax. Revisit `max_model_len`,
  `gpu_memory_utilization`, and `tensor_parallel_size` for its context and GPU
  footprint. Keep `served_model_name: llm` to retain the built-in
  `llama_nemotron` adapter, or update the model profile's adapter when changing
  that name or any wire behavior. Hosting backend is selectable per YAML
  (refer to *Choosing the vLLM runtime*). The
  [model card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1)
  recommends `temperature=0.6` and `top_p=0.95` with reasoning enabled, and
  greedy decoding with reasoning disabled. It identifies the model as ready
  for commercial use under the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
  and the
  [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/).
- **nemotron3-nano-llm** is a thin wrapper around `vllm serve` for
  `NVIDIA-Nemotron-3-Nano-30B-A3B-{NVFP4,FP8}` (auto-selected by GPU compute
  capability). vLLM handles tool calling (`qwen3_coder` parser), reasoning
  extraction (`nano_v3` parser — auto-fetched into `model_cache`), and
  FlashInfer FP4 MoE kernels. `model_blackwell` selects the NVFP4 checkpoint
  on SM100+; `model_ada` selects FP8 on earlier supported GPUs. Parsed
  reasoning is returned in the `reasoning` field. The `nemotron3_nano` client
  preset disables thinking by default so short calls retain an answer token
  budget; pass `enable_thinking=True` on a call to opt in. Native FP4 requires a
  Blackwell-class GPU such as B200 or RTX PRO 6000; FP8 is used on Ada,
  Hopper, and Ampere.
  `enforce_eager: true` by default to avoid the silent 3–8 min CUDA graph and
  FlashInfer autotune on cold start. Hosting backend is selectable per YAML
  (refer to *Choosing the vLLM runtime*). The
  [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
  recommends `temperature=1.0` and `top_p=1.0` for reasoning, and
  `temperature=0.6` and `top_p=0.95` for tool calling. It identifies the model
  as ready for commercial use under the
  [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).
- **nemotron-omni-llm** is a vLLM-backed multimodal LLM serving
  `Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (text + video input) at port 8108.
  The YAML auto-selects between three model variants by detected GPU compute
  capability: NVFP4 on Blackwell (SM100+), FP8 on Ada, Hopper, and Ampere, BF16 forced via
  `use_bf16: true` for highest quality at the largest VRAM cost. Same
  OpenAI-compatible HTTP contract as the other LLM servers — swap the port to
  swap backends. Hosting backend is selectable per YAML (refer to *Choosing the
  vLLM runtime*); persists across stack restarts in both pip and docker modes.
- **stt-server** loads parakeet-tdt-0.6b-v3 via NeMo ASR in-process.
  English-only; the `language` and `temperature` form fields are accepted but
  ignored. Set `startup_timeout_s` to a positive finite number to override the
  600-second cold-start budget.
- **magpie-tts** loads magpie_tts_multilingual_357m via NeMo TTS in-process.
- **piper-tts** serves any rhasspy/piper-voices ONNX voice; ~100 ms/sentence on CPU.
  All inference runs in a thread pool so the asyncio loop is never blocked.
- **embedding-server** serves `nvidia/llama-nemotron-embed-1b-v2` through
  `/v1/embeddings`. It emits 2048-dimensional Matryoshka embeddings and can
  truncate them to 384, 512, 768, 1024, or 2048 dimensions. The checked-in
  configuration reserves 20% of a GPU and uses Docker.
- **rag-service** is a typed dense document-retrieval capability. Point
  `documents_dir` at an application-owned tree and `models_config` at a profile
  with an `embedding` role. It chunks and embeds supported documents at
  startup, caches its index, and returns matches above `min_score`. Start the
  embedding service first, RAG second, and the consuming worker last. The cache
  includes document content, indexing settings, and the model profile; change
  `cache_key` when a remote endpoint changes its backing model without changing
  that profile.
- **video-memory-service** owns recorded chunk queries, NVDEC, and PNG output
  behind typed msgpack over ZMQ on port 8310. Set `recordings_dir` in its YAML to
  enable recorded-video operations; the path must match the hub's
  `video_recording.out_dir`. Latest video and sampling windows end at the
  newest recorded timestamp and require only a duration. Historical frame,
  video, and sampling requests share an absolute `start_us`; video windows add
  a duration. Sampling also accepts a hard total frame budget, decodes each
  selected chunk once, skips unavailable or corrupt chunks when other frames
  remain, and can bound exported PNG dimensions. The sampled timestamps are
  estimates interpolated from chunk metadata. The selection budget may be up
  to 256, but the shipped Cosmos VLM accepts no more than four selected images
  per inference request. Current frames stay with the caller's hub client.
  When `recordings_dir` is empty, participant discovery returns an empty list
  and recorded-media operations return `recording_disabled`.
- Ports are configurable — avoid conflicts with LiveKit (7880–7882) and hub (8080, 8090).
- Standalone service YAMLs live beside the services that support direct local
  launch. When copying one into `agent-samples/<name>/yaml/`, set
  `model_cache` to `../../../models` so it still resolves to the repository's
  `models/` directory.
- The generic NIM wrapper has no service-local YAML. Use a hardware profile
  under `agent-samples/model-servers/yaml/<gpu-profile>/`; its
  `nim_<role>_server.yaml` files use `nim_cache`, normally
  `../../../../models/nim` from that location.
