<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-models

<a id="contract"></a>
(agent-sdk-models-contract)=
`xr-ai-models` defines typed LLM, VLM, STT, TTS, and embedding protocols and
constructs concrete clients from deployment profiles. Workers depend on those
protocols instead of hand-written HTTP or vendor SDK calls. Refer to
{doc}`python/index` for exact classes, methods, fields, and defaults. Refer to
{doc}`/components/ai-services` for server operation.

<a id="quickstart"></a>
(agent-sdk-models-quickstart)=
## Construct a model client

```python
from xr_ai_models import ChatMessage, load_models_config, make_llm

config = load_models_config("yaml/models.json")
async with make_llm(config, "agent_llm") as llm:
    response = await llm.chat(
        [ChatMessage(role="user", content="hello")],
        max_tokens=128,
        enable_thinking=True,
    )
    print(response.content, response.reasoning)
```

(profile-contract)=
A client profile names logical roles and declares adapters and endpoints:

```json
{
  "models": {
    "agent_llm": {
      "category": "llm",
      "adapter": {"preset": "nemotron_omni"},
      "endpoint": {
        "base_url": "http://localhost:8108",
        "timeout": 60.0
      }
    }
  }
}
```

- `adapter` owns the model name, wire quirks, capabilities, default request
  extras, and reasoning-field normalization.
- `endpoint` owns connectivity, timeouts, and environment-variable credentials.
  Optional health settings control explicit SDK `health()` calls; they do not
  cause consumer workers to poll endpoints at startup.
- Optional `deployment` metadata selects the processes a shared model-server
  orchestrator manages. Consumer profiles omit it: their endpoints are operated
  outside the sample, whether locally or remotely. Existing explicit `reused`
  and `external` entries remain supported for compatibility.

(deployment-profiles)=
Workers may load JSON or YAML. For compatibility, the loader accepts a direct
role mapping, legacy flat entries, `health_check: true` or `health_check: false`,
and `kind: preset:<name>`. The public role-spec classes also retain their legacy
flat constructors and read-only flat properties. Profiles shared with the
stdlib-only launcher must use the wrapped nested JSON form with `adapter`,
`endpoint`, and `deployment` objects. Only the worker-side loader accepts omitted
`deployment` metadata. Launcher credentials are explicit: endpoint credentials
use `api_key_env`, while credentials needed by a managed service itself use
`deployment.credentials`.

## Request failures

Applications handle errors from model requests. For worker startup behavior,
refer to {ref}`consumer-model-readiness`.

## Built-in adapters

| Preset | Target | Important behavior |
|---|---|---|
| `cosmos3_nano_reasoner` | Cosmos3 Nano VLM | Image; video requires `max_videos_per_prompt >= 1`; no reasoning-field mapping |
| `cosmos_vlm` | Cosmos-Reason1 compatibility | Image; video requires `max_videos_per_prompt >= 1`; thinking disabled by default |
| `llama_nemotron` | Llama Nemotron LLM | Server-side `llama3_json` tool calls |
| `nemotron3_nano` | Nemotron 3 Nano LLM | Normalizes the `reasoning` field; thinking disabled unless requested |
| `nemotron_omni` | Nemotron Omni | Tool calls, image and video, `reasoning_content` normalization; thinking disabled unless requested |
| `nemotron_embedding` | Embedding server | OpenAI-compatible dense vectors |
| `parakeet_stt` | STT server | OpenAI-compatible transcription |
| `pocket_tts` | Pocket TTS | OpenAI-compatible synthesis plus native PCM streaming |
| `magpie_tts` | Magpie TTS | OpenAI-compatible speech synthesis |

The Cosmos adapter capability describes the supported request shape, but the
server controls whether video input is enabled. Every checked-in `vlm-server`
profile sets `max_videos_per_prompt: 0` to avoid reserving unused activation
memory. Set it to at least `1` and restart the persistent VLM server before
sending a video request.

<a id="protocols"></a>
(agent-sdk-models-protocols)=
`ChatResponse.reasoning` is the canonical post-normalization field. Model
adapters absorb whether the provider calls it `reasoning` or
`reasoning_content`. LLM and VLM calls accept controlled per-request headers for
Relay lineage, but callers cannot replace the profile's `Authorization` header.

Single-image `ask_image()` and `stream()` calls are wrappers over the ordered
multi-image methods. All images are placed in one user message in caller order.

(remote-and-hosted-nim-endpoints)=
<a id="remote--hosted-nim-endpoints"></a>
(remote-hosted-nim-endpoints)=
## Hosted endpoints

A hosted OpenAI-compatible endpoint changes only the profile:

```json
{
  "models": {
    "vlm": {
      "category": "vlm",
      "adapter": {
        "kind": "openai_compat",
        "model_name": "nvidia/cosmos3-nano-reasoner"
      },
      "endpoint": {
        "base_url": "https://integrate.api.nvidia.com",
        "api_key_env": "NGC_API_KEY"
      }
    }
  }
}
```

For code that calls `health()` explicitly, `readiness: none` makes that call
succeed without a request; `health_path` selects the HTTP route when probing is
enabled. Tea making's RAG service requires this configuration for hosted embedding
endpoints; refer to {ref}`rag-embedding-health`.

(riva-grpc-speech-nim-stt-tts)=
## Riva speech over gRPC

Riva speech NIMs use `kind: riva_grpc`, not OpenAI `/v1/audio`. Install the
`riva` extra; its `nvidia-riva-client` import is deferred until `make_stt()` or
`make_tts()` selects that kind.

```yaml
stt:
  kind: riva_grpc
  category: stt
  base_url: localhost:50051
  language: en-US
```

STT accepts 16-bit PCM WAV or raw int16 PCM with an explicit sample rate. TTS
also accepts `voice` and `sample_rate`. A hosted NVCF endpoint uses TLS,
`api_key_env`, and its `function_id`. If custom code explicitly calls `health()`,
set `health_check: false` for a hosted endpoint without a channel-ready health
surface. Existing gRPC request deadlines and error propagation are unchanged.

## Tests

The CPU-only `tests/test_models_*.py` modules exercise model wire formats with
the `tests/_stub_openai.StubOpenAI` httpx mock transport. Run them from the
repository root:

```bash
uv run --project tests pytest -q tests/test_models_*.py
```
