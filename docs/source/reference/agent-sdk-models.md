<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-models

## Overview

The model that works for your prototype may change as you tune quality,
hardware use, or deployment. Shared model interfaces let your application call
the capabilities it needs while configuration selects the supported model
adapter and endpoint.

## Model interfaces

<a id="contract"></a>
(agent-sdk-models-contract)=
The protocols cover LLM, VLM, STT, TTS, embedding, decision, and OCR services.
Workers construct clients through the factories in `xr-ai-models`; model HTTP
and vendor SDK calls stay inside the adapters. Refer to
{doc}`python/index` for exact classes, methods, fields, and defaults. Refer to
{doc}`/components/ai-services` for server operation.

<a id="quickstart"></a>
(agent-sdk-models-quickstart)=
## Construct a model client

```python
from xr_ai_models import ChatMessage, load_models_config, make_llm

config = load_models_config("yaml/models.json")
async with make_llm(config, "llm") as llm:
    response = await llm.chat(
        [ChatMessage(role="user", content="hello")],
        max_tokens=128,
        enable_thinking=True,
    )
    print(response.content, response.reasoning)
```

Choice-only decision clients use `make_decision` and the SystemOne adapter:

```python
from xr_ai_models import DecisionQuestion, load_models_config, make_decision

config = load_models_config("yaml/models.json")
async with make_decision(config, "triage") as decisions:
    result = await decisions.decide(
        "The participant asked to start the timer.",
        {
            "approval": DecisionQuestion(
                instructions="Did the participant explicitly approve the action?",
                criteria={"yes": "An explicit approval", "no": "No explicit approval"},
            )
        },
    )
    print(result.answers["approval"].choice)
```

The profile entry uses `category: decision`, `kind: systemone`, a model name,
and an endpoint. A request the service rejects as more input than its model
accepts (HTTP 413) raises `DecisionInputTooLargeError`. The limit belongs to the
service, so a caller handles this by retrying with less input. The standalone
{doc}`Clef-Flash guide </reference/clef-flash>` documents the local service and
its pinned model release.

(profile-contract)=
Model entry names are application-defined. Applications can configure multiple
LLMs under distinct names with `make_llm(config, name)`, using the same endpoint
or different endpoints.

A client profile names logical roles and declares adapters and endpoints:

```json
{
  "models": {
    "llm": {
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

## Annotated consumer configuration

The package reference below shows local endpoint examples, not a default launch
stack. Copy `models.yaml.sample` to your application's `models.yaml`, keep only
the roles it uses, and pass that file explicitly to `load_models_config`.
Loading it does not connect to or launch a model server.
The launcher requires JSON deployment profiles; this worker-side YAML reference
does not replace a launcher's `models.json`.
Refer to {doc}`/guides/deploying-with-nim` for the separate annotated NIM profile
and its transport-specific settings.

```{literalinclude} ../../../agent-sdk/xr-ai-models/models.yaml.sample
:language: yaml
```

(deployment-profiles)=
Workers may load JSON or YAML. For compatibility, the loader accepts a direct
role mapping, legacy flat entries, `health_check: true` or `health_check: false`,
and `kind: preset:<name>`. The public role-spec classes also retain their legacy
flat constructors and read-only flat properties. Profiles shared with the
launcher must use the wrapped nested JSON form with `adapter`,
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
| `cosmos3_nano_reasoner` | Cosmos3 Nano VLM | Image; video requires `max_videos_per_prompt >= 1`; no reasoning switch |
| `cosmos_vlm` | Cosmos-Reason1 compatibility | Image; video requires `max_videos_per_prompt >= 1`; thinking disabled by default |
| `nemotron3_nano` | Nemotron 3 Nano LLM | Normalizes the `reasoning` field; thinking disabled unless requested |
| `nemotron_omni` | Nemotron Omni | Tool calls, forced tool calls and JSON Schema output, image and video, `reasoning_content` normalization; thinking disabled unless requested |
| `nemotron_embedding` | Embedding server | OpenAI-compatible dense vectors |
| `nemotron_ocr` | Hugging Face OCR server or OCR NIM 2.0 | Multilingual text, confidence, normalized polygons; `/v1/ocr` |
| `parakeet_stt` | STT server | OpenAI-compatible transcription |
| `pocket_tts` | Pocket TTS | OpenAI-compatible synthesis plus native PCM streaming |
| `magpie_tts` | Magpie TTS | OpenAI-compatible speech synthesis |

The Cosmos adapter capability describes the supported request shape, but the
server controls whether video input is enabled. Every checked-in `vlm-server`
profile sets `max_videos_per_prompt: 0` to avoid reserving unused activation
memory. Set it to at least `1` and restart the persistent VLM server before
sending a video request.

(agent-sdk-models-thinking)=
## Reasoning control

Callers request reasoning per call with `enable_thinking`. `True` or `False`
always applies to that call; omitting it (`None`) uses the endpoint default.
Code written this way runs unchanged when a deployment swaps models, because
the adapter's `thinking` setting decides how the switch reaches the model:

| `control` | Wire behavior | Used by |
|---|---|---|
| `chat_template` | `chat_template_kwargs.enable_thinking`; `thinking_budget` as vLLM's `thinking_token_budget` | `nemotron3_nano`, `nemotron_omni`, `cosmos_vlm` |
| `system_prompt` | A `detailed thinking on` or `off` line at the start of the system message | Profiles for Llama-Nemotron endpoints |
| `none` | Nothing is sent; the model has no switch or rejects template arguments | `cosmos3_nano_reasoner` |
| unset | `chat_template` when `default` is set; otherwise only an explicit `enable_thinking=True` sends the switch. A supplied budget is sent unless reasoning is explicitly off. | Profiles without a preset |

The per-call `thinking_budget` is sent as `thinking_token_budget` for
`chat_template` and unset controls unless reasoning is off. Budget enforcement
requires support in the deployed model server, independently of support for
the thinking switch. It has been verified on Nemotron Omni served by vLLM with
`--reasoning-parser nemotron_v3`; NIM budget support has not been qualified.
Other endpoints may reject or ignore the field. The SDK does not detect budget
support or retry without it. Omitting `thinking_budget` (or passing `None`)
leaves the thinking switch available without requesting a reasoning cap.
Sample requests that explicitly pass a budget continue to send it; selecting
a different endpoint profile does not independently disable that budget.

`default` is the endpoint's reasoning state when a call does not choose one;
`null` sends no switch and leaves the model's own default. With `system_prompt`,
a complete `detailed thinking on` or `off` line already in the system message
stands for the default, and an explicit `enable_thinking` replaces it. A profile
overrides either key on top of its preset:

```yaml
adapter:
  preset: nemotron3_nano
  thinking:
    default: true
```

Set `control: none` for a hosted endpoint that rejects unknown request fields;
such a model ignores `enable_thinking`. `capabilities.reasoning` reports only
whether responses carry a separate reasoning field, not whether the switch is
honored. A legacy `default_extras.chat_template_kwargs.enable_thinking`
boolean is still accepted: with `control: chat_template` or unset it becomes
`thinking.default`, overriding the preset's, and a call can override it. With
another control it is sent unchanged. A profile that sets both keys to
different values is rejected.

(agent-sdk-models-structured-output)=
## Structured output and forced tool calls

`LLMService.chat`, `stream` and `stream_events` accept two constraints:

- `response_schema`, a JSON Schema mapping (for example a Pydantic model's
  `model_json_schema()`), limits the response text to one JSON value matching
  it. It is sent as the OpenAI-compatible `response_format` of type
  `json_schema`. It cannot be combined with tools.
- `tool_choice` is `"auto"`, `"none"`, `"required"` (some tool must be called)
  or the name of one supplied tool (that tool must be called).

Both are applied while the model generates, but they promise different things.
A `response_schema` answer is limited to the schema. A forced tool call is
always made, but servers may constrain only the call and not every rule of its
argument schema: on Nemotron Omni with vLLM's `qwen3_coder` tool parser, 6 of
1,728 forced calls exceeded a `maxLength` or omitted a required field. Native
tools validate their arguments and return the error to the model, so a tool loop
recovers; for one value that must match a schema, prefer `response_schema`.

Callers still validate every result. A response cut off at `max_tokens` is not a
complete value, and a constrained answer can be cut off: vLLM's xgrammar JSON
grammar allows any amount of whitespace, and 1 of 1,152 Nemotron Omni answers
filled the rest of its tokens with it (`finish_reason` `length`). vLLM's server
option `--structured-outputs-config '{"disable_any_whitespace": true}'` removes
that freedom; the repository's model servers do not set it yet.

Support depends on the model and its server, so an adapter declares it in
`capabilities.json_schema` and `capabilities.tool_choice`. A request using an
undeclared constraint raises `ValueError` before anything is sent, instead of
being silently ignored. `"auto"` and `"none"` need no declaration.

Two rules apply to every model:

- When reasoning may precede the answer, the client also states the schema in
  the system message. Without it, Nemotron Omni with reasoning on sometimes
  filled the constrained answer with whitespace until the token limit. Without
  reasoning the client leaves the prompt alone: there the constraint alone kept
  every answer valid, and stating the schema biased an enum answer toward its
  first value. Tools already carry their schema, so forced tool calls add
  nothing.
- With reasoning on, a constrained request requires `thinking_budget`, and
  `max_tokens` must exceed it. Reasoning comes before the constrained answer
  and can otherwise use every token, leaving no answer. An endpoint whose
  reasoning switch cannot carry a budget (`system_prompt`, or `none` on a model
  that declares `capabilities.reasoning`) rejects constrained requests that may
  reason.

### Qualify a model

Before declaring either capability for a new model or server, run the live eval
in [`agent-sdk/xr-ai-models/eval`](https://github.com/NVIDIA/xr-ai/tree/main/agent-sdk/xr-ai-models/eval)
against it. It passes only when, with reasoning off and on and through both
`chat` and `stream_events`, every `response_schema` answer matches its schema
and every forced tool call is made. Then set the
capabilities in the preset, or in a deployment profile:

```yaml
adapter:
  preset: nemotron3_nano
  capabilities:
    json_schema: true
    tool_choice: true
```

`nemotron_omni` declares both. The other presets declare neither until they
have been evaluated.

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

## Deployment profiles

Use the same factories and typed interfaces for local development and native
NIM deployment. The example in `deployment/nim/models.yaml` selects native HTTP
chat and embedding endpoints and Riva gRPC speech, without compatibility servers.
The embedding factory honors existing request defaults, including asymmetric
`input_type` batches. Refer to {doc}`/guides/deploying-with-nim` for deployment
ownership, sample configuration paths, and validation commands.

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
surface. Recognition timeout and cancellation cancel the native RPC and join
its pending client read; synthesis uses the same cleanup rule for streamed
audio.

## Tests

The CPU-only `tests/test_models_*.py` modules exercise model wire formats with
the `tests/_stub_openai.StubOpenAI` httpx mock transport. Run them from the
repository root:

```bash
uv run --project tests pytest -q tests/test_models_*.py
```
