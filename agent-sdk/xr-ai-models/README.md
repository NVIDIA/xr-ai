<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-models

Unified service protocols and OpenAI-compatible HTTP clients for the xr-ai
model layer. Worker code depends on the typed protocols `LLMService`,
`VLMService`, `STTService`, `TTSService`, and `EmbeddingService`, and constructs
concrete clients from a model deployment profile — no hand-rolled httpx calls
in callers, no model quirks leaking out of this package.

## Contract

Model profiles name logical roles and keep endpoint and model-specific behavior
inside this package. Callers depend on the typed protocol and can change models
through configuration.

## Quickstart

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

`models.local.json`:

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

Built-in presets — see `xr_ai_models/presets/`:

| Preset | Service it targets | Notes |
|---|---|---|
| `cosmos3_nano_reasoner` | vlm-server          | default; Cosmos3 Nano text-output Reasoner, image + video; video requires `max_videos_per_prompt >= 1` |
| `cosmos_vlm`     | vlm-server               | Cosmos-Reason1 compatibility option; image + video; `enable_thinking=false` by default; video requires `max_videos_per_prompt >= 1` |
| `llama_nemotron` | llama-nemotron-llm-server | OpenAI tool calling via llama3_json (server-side) |
| `nemotron3_nano` | nemotron3-nano-llm-server | reasoning field: `reasoning` |
| `nemotron_omni`  | nemotron-omni-llm-server  | default LLM; reasoning field: `reasoning_content`, thinking off unless requested, vision + video |
| `nemotron_embedding` | embedding-server | OpenAI-compatible dense embeddings |
| `parakeet_stt`   | stt-server               | |
| `piper_tts`      | tts/piper                | |
| `magpie_tts`     | tts/magpie               | |

## Profile contract

```json
{
  "models": {
    "agent_llm": {
      "category": "llm",
      "adapter": {
        "kind": "openai_compat",
        "model_name": "llm",
        "capabilities": {"tool_calls": true, "reasoning": true},
        "reasoning_field": "reasoning"
      },
      "endpoint": {
        "base_url": "http://localhost:8107",
        "timeout": 60.0,
        "readiness": "health"
      },
      "deployment": {"ownership": "reused", "service": "agent-llm"}
    }
  }
}
```

`category` selects the service protocol. `adapter` owns model and wire quirks;
`endpoint` owns connectivity, environment-based credentials, timeouts, and
readiness; `deployment` tells a launcher whether the service is `managed`,
`reused`, or `external`.

Profiles may be JSON or YAML. The loader also accepts a direct role mapping,
flat entries, `health_check: true|false`, and `kind: preset:<name>` for
backward compatibility. The public role-spec classes also retain their legacy
flat positional and keyword constructors and expose flat fields as read-only
properties.

## Deployment profiles

A profile may separate model behavior, endpoint connectivity, and process
ownership. The existing flat YAML format remains supported for workers. A
profile shared with the stdlib-only launcher must be wrapped in `models`, use
the nested shape below, and be JSON so both consumers read the same file.

```json
{
  "models": {
    "agent_llm": {
      "adapter": {"preset": "nemotron3_nano"},
      "endpoint": {"base_url": "http://localhost:8107", "readiness": "health"},
      "deployment": {"ownership": "reused", "service": "agent-llm"}
    }
  }
}
```

Workers pass the profile to `load_models_config()` as usual. A stdlib-only
orchestrator can call `load_model_deployment(worker_config)` from
`xr-ai-launcher` to map `managed` to `launch_mode="own"`, `reused` to
`launch_mode="reuse"`, and `external` to no local process. Credentials used by
the launcher must be declared explicitly as `endpoint.api_key_env`; launcher
profiles do not inherit credentials from adapter presets. A deployment may
additionally list `credentials` the launched service itself needs (e.g.
`NGC_API_KEY` for a NIM container's nvcr.io pull and engine download) even
when the endpoint takes no API key; only the launcher collects these, and
`ModelsConfig.required_credentials` stays endpoint keys only.

The launcher validates the `.json` suffix before reading a profile; selecting a
YAML profile is supported only for worker-side `xr-ai-models` loading. Model
roles compose `AdapterSpec`, `EndpointSpec`, and `DeploymentSpec`, while their
legacy flat attributes remain available as read-only compatibility aliases.
The simple VLM sample provides complete local and hosted profiles.

## Protocols

The versioned documentation site generates exact protocol methods, parameters,
types, and defaults from the public Python declarations. This README keeps only
the behavioral rules that span multiple calls or implementations.

`ChatResponse.reasoning` is the canonical reasoning field — the
`reasoning_field` knob normalizes `reasoning_content` (nemotron_v3 parser)
into the same surface.

`LLMService` and `VLMService` request methods accept optional string-valued
per-call headers for execution context such as Relay session lineage. The model
profile remains the authority for credentials: callers cannot supply an
`Authorization` header.

`ask_image()` and `stream()` are one-image wrappers over `ask_images()` and
`stream_images()`. Multi-image calls preserve caller order and place every
image in one OpenAI-compatible user message before the question.

## Remote / hosted-NIM endpoints

Cloud / remote endpoints (e.g. hosted [NVIDIA NIM](https://build.nvidia.com))
are a profile change:

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

`readiness: health` makes `health()` probe `base_url/health`. Remote endpoints
without that route use `readiness: none`, which makes
`health()` return `True` without a request — otherwise a worker's readiness
gate would block forever. See
`docs/source/components/ai-services.md` for hosted endpoint operation.

## Riva gRPC speech (NIM STT/TTS)

NIM speech is Riva over gRPC, not OpenAI `/v1/audio`. The `riva_grpc` kind
covers it, for self-hosted Riva/NIM speech containers and for hosted NVCF
endpoints alike. It requires the `riva` extra (`xr-ai-models[riva]` →
`nvidia-riva-client`); the import is deferred to `make_stt`/`make_tts` so
the base install stays gRPC-free.

```yaml
stt:
  kind:      riva_grpc
  category:  stt
  base_url:  localhost:50051   # self-hosted container's gRPC port
  language:  en-US
```

STT input must be 16-bit PCM: `transcribe` accepts a 16-bit PCM WAV (any
other sample width raises `ValueError`, since the frames go to Riva labelled
LINEAR_PCM and would transcribe as garbage) or raw int16 PCM with an
explicit `sample_rate=`.

TTS additionally takes `voice:` and `sample_rate:` (default 44100).
`health_check: true` (the default) runs a gRPC channel-ready probe. For a
hosted NVCF endpoint, set `base_url: grpc.nvcf.nvidia.com:443`,
`use_ssl: true`, `api_key_env`, the model's `function_id:` from
build.nvidia.com, and `health_check: false` (no health surface).

Future non-OpenAI-compat backends (LiteLLM, vendor SDKs) plug in as new
`kind`s in `_factory.py::make_*`; the protocols and callers do not change.

## Tests

`tests/test_models_*.py` exercise the wire format against a
`tests/_stub_openai.StubOpenAI` httpx MockTransport — no GPU required.
