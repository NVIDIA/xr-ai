<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Customizing model servers

Use this workflow to change which shared models run, where they run, or how a
sample connects to them. Model-server customization has two separate parts:

1. A `model-servers` deployment profile declares which shared services the
   operator starts and owns.
2. Each sample's active models JSON declares compatible client adapters and
   endpoints. Consumer entries do not need a `deployment` object.

Samples do not start or stop model services. Keep that ownership boundary when
adapting a sample: customize the shared stack first, start it, and then point
the sample at the resulting endpoints.

## Configuration layers

| Layer | Location | Responsibility |
|---|---|---|
| Deployment profile | `agent-samples/model-servers/yaml/models.<name>.json` | Logical roles, client adapters, endpoints, credentials, and shared-service ownership |
| Hardware profile | `agent-samples/model-servers/yaml/<gpu-profile>/` | Image or checkpoint, ports, GPU placement, cache paths, and runtime memory settings |
| Sample models JSON | `agent-samples/<sample>/yaml/models*.json` | The roles that worker consumes and the endpoints it reuses |
| Sample worker YAML | `agent-samples/<sample>/yaml/<worker>.yaml` | Application behavior such as prompts, voice gating, timeouts, and capability-service endpoints |

Do not use `--gpu-profile` to select models. It selects the reviewed hardware
layout (`dual_48G_ada`, `spark`, or `96G_blackwell`). The `--models` option
selects the deployment profile independently.

## Start from a shipped deployment profile

The shipped profiles are:

- `default`: local Parakeet STT, Pocket TTS, Nemotron 3.5 Lightning, Cosmos3-Nano
  Reasoner, and Nemotron embedding services.
- `vlm_llm_nim`: local STT, Pocket TTS, and embedding plus self-hosted
  Nemotron 3.5 Lightning and Cosmos3-Nano Reasoner NIM containers.

The shared profiles use Pocket TTS. Magpie remains available as a standalone
service, but is not integrated into the persistent model stack.

Copy the closest profile under a new name:

```bash
cp agent-samples/model-servers/yaml/models.default.json \
  agent-samples/model-servers/yaml/models.my-stack.json
```

Run it by filename stem:

```bash
uv run --project agent-samples/model-servers \
  model_servers --models my-stack
```

You can also pass an absolute JSON path or a path relative to the current
working directory to `--models`.
The profile must retain the wrapped JSON shape:

```json
{
  "models": {
    "vlm": {
      "adapter": {"preset": "cosmos3_nano_reasoner"},
      "endpoint": {
        "base_url": "http://localhost:8100",
        "readiness": "health"
      },
      "deployment": {
        "ownership": "managed",
        "service": "vlm"
      }
    }
  }
}
```

Within the shared profile, `managed` means `model-servers` owns that service.
Roles may share a service; for example, `llm` and `agent_llm` can both name
`lightning`. A service name must have a corresponding row in `_MODEL_SERVICES` in
`agent-samples/model-servers/main.py`.

## Customize a hardware-specific server

Each managed service resolves its YAML from the detected GPU-profile
directory. Change the relevant file there when customizing an existing
service. Common fields include:

```yaml
model: nvidia/Cosmos3-Nano
port: 8100
model_cache: ../../../../models
cuda_visible_devices: "0"
gpu_memory_utilization: 0.55
```

NIM services use `image`, `http_port`, `nim_cache`, and optional `NIM_*`
environment settings instead:

```yaml
image: nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0
http_port: 8100
nim_cache: ../../../../models/nim
cuda_visible_devices: "0"
env:
  NIM_MODEL_SIZE: "nano"
  NIM_MAX_MODEL_LEN: "16384"
```

The Cosmos 3 Reasoner container serves Nano as
`nvidia/cosmos3-nano-reasoner`. Keep the model ID in the deployment profile's
adapter synchronized with the NIM model size.

When one deployment profile needs a different launch configuration, add a
variant beside the base YAML:

```text
nim_vlm_server.yaml
nim_vlm_server_my-stack.yaml
```

The suffix is the deployment profile filename without `models.` or `.json`.
Only add a variant when the base configuration is not valid for that profile.
Review GPU placement and aggregate memory before running several services
together; the checked-in non-Ada NIM settings are estimates where their YAML
comments say so.

## Adapt a sample to the shared stack

Do not copy a complete model-server profile into a sample. It can omit roles
the sample needs, and its `managed` ownership belongs only in the shared stack.
Copy or update the relevant role entries in the sample's active models JSON,
omitting their `deployment` objects. Keep adapter settings and endpoint
credentials needed by the client; server-only credentials stay in the shared
stack's deployment profile.

For the shipped `vlm_llm_nim` stack:

1. Copy its `llm` entry when the sample needs language or tool calling.
2. Copy its `vlm` entry when the sample needs vision.
3. Remove the `deployment` object and endpoint `readiness` and `health_path`
   settings from the copied LLM and VLM entries. Keep the LLM endpoint on port
   8110 and the VLM endpoint on port 8100. The server profile keeps its settings.
4. If the sample uses `agent_llm`, duplicate the `llm` entry under that role.
5. Preserve the sample's STT, TTS, embedding, and other roles unless the shared
   stack provides intentional replacements.

The active files and NIM-relevant roles are:

| Sample | Active models file | Roles to replace for `vlm_llm_nim` |
|---|---|---|
| Simple VLM | `agent-samples/simple-vlm-example/yaml/models.json` | `vlm` |
| Lab instrument monitoring | `agent-samples/lab-instrument-monitoring/yaml/models.json` | `llm`, `vlm` |
| Tea making | `agent-samples/tea-making-sample/yaml/models.local.json` | `llm`, `vlm` |
| XR Render | `agent-samples/xr-render-demo/yaml/models.json` | `llm`, `agent_llm`, `vlm` |

For example, a sample reusing the Cosmos3-Nano Reasoner NIM uses:

```json
{
  "category": "vlm",
  "adapter": {
    "kind": "openai_compat",
    "model_name": "nvidia/cosmos3-nano-reasoner",
    "capabilities": {
      "streaming": true,
      "vision": true,
      "video": true
    }
  },
  "endpoint": {"base_url": "http://localhost:8100"}
}
```

Sample launchers declare only their application processes; model endpoints
are specified in the client profile. Starting or stopping the sample never
changes the shared NIM container. For startup ordering, refer to
{ref}`consumer-model-readiness`.

## Use an endpoint at another address

If an operator already runs a compatible service elsewhere, no model-server
change is required. Update the sample entry's `endpoint.base_url`. Both
operator-managed XR AI services and hosted APIs use client entries without
`deployment`. Worker model clients do not need health polling settings; the
RAG embedding exception is described below. A hosted endpoint may also need a
credential:

```json
{
  "endpoint": {
    "base_url": "https://integrate.api.nvidia.com",
    "api_key_env": "NGC_API_KEY"
  }
}
```

Do not put the credential value in JSON. Export it or use the credential store.
Refer to {doc}`/getting_started/credentials` for credential options.

(rag-embedding-health)=
## RAG embedding health

Tea making's RAG service reads the `embedding` role from the same
`yaml/models.local.json` profile as the worker. Unlike worker model clients,
it explicitly calls embedding `health()` before building its index and when
reporting RAG readiness. Preserve health settings for this role when changing
its endpoint.

For a hosted embedding provider without a health route, set `readiness: none`
in the embedding entry's `endpoint` object:

```json
{
  "endpoint": {
    "base_url": "https://integrate.api.nvidia.com",
    "api_key_env": "NGC_API_KEY",
    "readiness": "none"
  }
}
```

This skips the health request; actual embedding requests still use the configured
adapter and credentials. If the provider exposes a supported health route,
use `readiness: health` and set `health_path` to that route instead, for example
`/v1/health/ready`. Omitting these settings defaults to probing `/health`, which
can prevent RAG startup even when embedding inference works.

## Riva speech boundary

The generic model-server service table and NIM YAMLs still support custom Riva
STT and TTS profiles through the `stt-nim` and `tts-nim` service names. No
shipped model-server profile selects them, and the samples intentionally do not
install the optional Riva client. Adding Riva to a sample therefore requires an
explicit worker dependency and models configuration change; it is not an
endpoint-only customization.

## Validate and switch profiles

Before committing a custom profile:

```bash
jq empty agent-samples/model-servers/yaml/models.my-stack.json

uv run --project tests pytest -q \
  tests/test_model_servers.py \
  tests/test_launcher_config.py \
  tests/test_nim_docker.py
```

Model servers persist after the `model_servers` command reports readiness.
Starting another profile stops persisted services outside the new selection
before launching it. After changing an image, checkpoint, or launch setting,
stop the old stack once so it cannot continue serving stale configuration:

```bash
uv run --project agent-samples/model-servers model_servers --stop
uv run --project agent-samples/model-servers \
  model_servers --models my-stack
```

For adapter fields and model capabilities, refer to
{doc}`/reference/agent-sdk-models`. For server runtime behavior, persistence,
and NIM credentials, refer to {doc}`/components/ai-services`.
