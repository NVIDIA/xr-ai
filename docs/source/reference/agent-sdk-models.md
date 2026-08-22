<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-models

`xr-ai-models` defines typed LLM, VLM, STT, TTS, and embedding protocols and
constructs concrete clients from deployment profiles. Workers depend on those
protocols instead of hand-written HTTP or vendor SDK calls. Exact classes,
methods, fields, and defaults are in {doc}`python/index`; server operation is in
{doc}`/components/ai-services`.

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

A profile names logical roles and separates three concerns:

```json
{
  "models": {
    "agent_llm": {
      "category": "llm",
      "adapter": {"preset": "nemotron_omni"},
      "endpoint": {
        "base_url": "http://localhost:8108",
        "timeout": 60.0,
        "readiness": "health"
      },
      "deployment": {"ownership": "reused", "service": "omni"}
    }
  }
}
```

- `adapter` owns the model name, wire quirks, capabilities, default request
  extras, and reasoning-field normalization.
- `endpoint` owns connectivity, readiness, timeouts, and environment-variable
  credentials.
- `deployment` tells an orchestrator whether the process is managed, reused, or
  external.

Workers may load JSON or YAML, including legacy flat entries. Profiles shared
with the stdlib-only launcher must use the wrapped nested JSON form. Launcher
credentials are explicit: endpoint credentials use `api_key_env`, while
credentials needed by a managed service itself use `deployment.credentials`.

## Built-in adapters

| Preset | Target | Important behavior |
|---|---|---|
| `cosmos3_nano_reasoner` | Cosmos3 Nano VLM | Image and video; no reasoning-field mapping |
| `cosmos_vlm` | Cosmos-Reason1 compatibility | Image and video; thinking disabled by default |
| `llama_nemotron` | Llama Nemotron LLM | Server-side `llama3_json` tool calls |
| `nemotron3_nano` | Nemotron 3 Nano LLM | Normalizes the `reasoning` field |
| `nemotron_omni` | Nemotron Omni | Tool calls, image/video, `reasoning_content` normalization |
| `nemotron_embedding` | Embedding server | OpenAI-compatible dense vectors |
| `parakeet_stt` | STT server | OpenAI-compatible transcription |
| `piper_tts` | Piper TTS | OpenAI-compatible speech synthesis |
| `magpie_tts` | Magpie TTS | OpenAI-compatible speech synthesis |

`ChatResponse.reasoning` is the canonical post-normalization field. Model
adapters absorb whether the provider calls it `reasoning` or
`reasoning_content`. LLM and VLM calls accept controlled per-request headers for
Relay lineage, but callers cannot replace the profile's `Authorization` header.

Single-image `ask_image()` and `stream()` calls are wrappers over the ordered
multi-image methods. All images are placed in one user message in caller order.

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
        "api_key_env": "NGC_API_KEY",
        "readiness": "none"
      },
      "deployment": {"ownership": "external"}
    }
  }
}
```

Use `readiness: none` only when the remote provider has no compatible health
route. It makes `health()` succeed without a request, preventing an impossible
local readiness gate.

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
`api_key_env`, its `function_id`, and `health_check: false` because it has no
Riva channel-ready health surface.
