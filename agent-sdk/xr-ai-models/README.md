<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-models

Unified service protocols and typed HTTP clients for the xr-ai model layer.
Worker code depends on `LLMService`, `VLMService`, `STTService`, `TTSService`,
`StreamingTTSService`, and `EmbeddingService`, and constructs concrete clients
from a model deployment profile — no hand-rolled HTTP calls in callers and no
model quirks leaking out of this package.

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
| `magpie_tts_nim` | magpie-tts-nim            | online signed-16-bit PCM plus offline WAV; English voice defaults to Mia |

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
profiles do not inherit credentials from adapter presets.

The launcher validates the `.json` suffix before reading a profile; selecting a
YAML profile is supported only for worker-side `xr-ai-models` loading. Model
roles compose `AdapterSpec`, `EndpointSpec`, and `DeploymentSpec`, while their
legacy flat attributes remain available as read-only compatibility aliases.
The simple VLM sample provides complete local and hosted profiles.

## Protocols

```python
class LLMService(Protocol):
    capabilities: Capabilities
    async def chat(self, messages, *, tools=None, max_tokens=None,
                   temperature=None, enable_thinking=False,
                   thinking_budget=None, timeout=None,
                   headers=None) -> ChatResponse: ...
    def stream(self, messages, *, ...) -> AsyncIterator[str]: ...
    async def health(self) -> bool: ...
    async def close(self) -> None: ...

class VLMService(Protocol):
    capabilities: Capabilities
    async def ask_image(self, image, question, *, system_prompt="",
                        max_tokens=None, temperature=None,
                        timeout=None, headers=None) -> ChatResponse: ...
    async def ask_images(self, images, question, *, system_prompt="",
                         max_tokens=None, temperature=None,
                         timeout=None, headers=None) -> ChatResponse: ...
    async def ask_video(self, video, question, *, system_prompt="",
                        max_tokens=None, temperature=None,
                        timeout=None, headers=None) -> ChatResponse: ...
    def stream(self, image, question, *, system_prompt="",
               max_tokens=None, temperature=None,
               timeout=None, headers=None) -> AsyncIterator[str]: ...
    def stream_images(self, images, question, *, system_prompt="",
                      max_tokens=None, temperature=None,
                      timeout=None, headers=None) -> AsyncIterator[str]: ...
    async def health(self) -> bool: ...

class STTService(Protocol):
    async def transcribe(self, audio: bytes, *, sample_rate=None,
                         channels=1, timeout=None) -> str: ...
    async def health(self) -> bool: ...

class TTSService(Protocol):
    async def synthesize(self, text: str, *, response_format="wav",
                         timeout=None) -> bytes: ...
    async def health(self) -> bool: ...

class StreamingTTSService(TTSService, Protocol):
    def stream_synthesize(self, text: str, *,
                          timeout=None) -> AsyncIterator[TTSAudioChunk]: ...

class EmbeddingService(Protocol):
    async def embed(self, texts, *, timeout=None) -> list[list[float]]: ...
    async def health(self) -> bool: ...
```

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

## Tests

`tests/test_models_*.py` exercise the wire format against a
`tests/_stub_openai.StubOpenAI` httpx MockTransport — no GPU required.
