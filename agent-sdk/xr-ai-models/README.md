<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-models

Unified service protocols and OpenAI-compatible HTTP clients for the xr-ai
model layer.  Worker code depends on the four protocols
(`LLMService`, `VLMService`, `STTService`, `TTSService`) and constructs
concrete clients from a `models.yaml` config — no hand-rolled httpx calls
in callers, no model quirks leaking out of this package.

## Why

Before this package, every consumer (vlm-mcp, simple-vlm-example worker,
xr-render-demo worker, xr-ai-pipecat) rolled its own httpx wrappers and
hard-coded model quirks (`chat_template_kwargs`, served-model-name strings,
the `reasoning` vs `reasoning_content` field difference between
nano_v3 and nemotron_v3 parsers).  Swapping an LLM meant editing N files.

After: one `models.yaml` per sample names the logical models the worker
needs; `make_llm(config, "agent_llm")` returns something that satisfies
`LLMService` regardless of which backend or quirks are involved.

## Quickstart

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

`models.yaml`:

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

Built-in presets — see `xr_ai_models/presets/`:

| Preset | Service it targets | Notes |
|---|---|---|
| `cosmos_vlm`     | vlm-server               | image + video; `enable_thinking=false` by default. Video requires vlm-server's `max_videos_per_prompt >= 1` |
| `llama_nemotron` | llama-nemotron-llm-server | OpenAI tool calling via llama3_json (server-side) |
| `nemotron3_nano` | nemotron3-nano-llm-server | reasoning field: `reasoning` |
| `nemotron_omni`  | nemotron-omni-llm-server  | reasoning field: `reasoning_content`, vision + video |
| `parakeet_stt`   | stt-server               | |
| `piper_tts`      | tts/piper                | |
| `magpie_tts`     | tts/magpie               | |

## Explicit (no-preset) spec

```yaml
agent_llm:
  kind:       openai_compat
  category:   llm
  base_url:   http://localhost:8107
  model_name: llm
  capabilities: { tool_calls: true, reasoning: true }
  reasoning_field: reasoning
  default_extras:
    chat_template_kwargs: { enable_thinking: false }
  timeout: 60.0
```

`category:` is required when not using a preset.

## Protocols

```python
class LLMService(Protocol):
    capabilities: Capabilities
    async def chat(self, messages, *, tools=None, max_tokens=None,
                   temperature=None, enable_thinking=False,
                   thinking_budget=None, timeout=None) -> ChatResponse: ...
    def stream(self, messages, *, ...) -> AsyncIterator[str]: ...
    async def health(self) -> bool: ...
    async def close(self) -> None: ...

class VLMService(Protocol):
    capabilities: Capabilities
    async def ask_image(self, image, question, *, system_prompt="",
                        max_tokens=None, temperature=None,
                        timeout=None) -> ChatResponse: ...
    async def ask_video(self, video, question, *, system_prompt="",
                        max_tokens=None, temperature=None,
                        timeout=None) -> ChatResponse: ...
    async def health(self) -> bool: ...

class STTService(Protocol):
    async def transcribe(self, audio: bytes, *, sample_rate=None,
                         channels=1, timeout=None) -> str: ...
    async def health(self) -> bool: ...

class TTSService(Protocol):
    async def synthesize(self, text: str, *, response_format="wav",
                         timeout=None) -> bytes: ...
    async def health(self) -> bool: ...
```

`ChatResponse.reasoning` is the canonical reasoning field — the
`reasoning_field` knob normalizes `reasoning_content` (nemotron_v3 parser)
into the same surface.

## Phase B preview

Cloud / remote endpoints are a `base_url` change — any spec can point at a
non-localhost OpenAI-compatible URL and pass `api_key_env:
OPENAI_API_KEY` for `Authorization: Bearer …`.  Future non-OpenAI-compat
backends (LiteLLM, vendor SDKs) plug in as new `kind`s in
`factory.py::make_*`; the protocols and callers do not change.

## Tests

`tests/test_models_*.py` exercise the wire format against a
`tests/_stub_openai.StubOpenAI` httpx MockTransport — no GPU required.
