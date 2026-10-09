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
| Deployment profile | `model-server-samples/model-servers/yaml/models.default.json` or a custom JSON path | Logical roles, client adapters, endpoints, credentials, and shared-service ownership |
| Hardware profile | `model-server-samples/model-servers/yaml/<gpu-profile>/` | Image or checkpoint, ports, GPU placement, cache paths, and runtime memory settings |
| Sample models JSON | `agent-samples/<sample>/yaml/models*.json` | The roles that worker consumes and the endpoints it reuses |
| Sample worker YAML | `agent-samples/<sample>/yaml/<worker>.yaml` | Application behavior such as prompts, voice gating, timeouts, and capability-service endpoints |

Do not use `--gpu-profile` to select models. It selects the reviewed hardware
layout (`dual_48G_ada`, `spark`, or `96G_blackwell`). The `--models` option
selects the deployment profile independently.

## Start from a shipped deployment profile

The default `yaml/models.default.json` profile starts local Parakeet STT,
Pocket TTS, Nemotron Omni, Cosmos3-Nano Reasoner, and Nemotron embedding
services. Run `model_servers` without `--models` to use it.

NIM launch support belongs to the separate {doc}`/reference/model-servers-nim`
sample, which provides Magpie TTS and compatible endpoints for existing agents.
The `model-servers` service catalog contains only local server wrappers.

Copy the default profile to customize it:

```bash
cp model-server-samples/model-servers/yaml/models.default.json \
  model-server-samples/model-servers/yaml/models.my-stack.json
```

Run it by JSON path:

```bash
uv run --project model-server-samples/model-servers \
  model_servers --models model-server-samples/model-servers/yaml/models.my-stack.json
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
Model entry names are application-defined. An application can configure multiple
LLMs under distinct names using the same LLM category and `make_llm(config, name)`
factory. Those entries may share a service or use different endpoints.
A service name must have a corresponding row in `_MODEL_SERVICES` in
`model-server-samples/model-servers/main.py`.

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

When one deployment profile needs a different launch configuration, add a
variant beside the base YAML:

```text
vlm_server.yaml
vlm_server_my-stack.yaml
```

The suffix is the deployment profile filename without `models.` or `.json`.
Only add a variant when the base configuration is not valid for that profile.
Review GPU placement and aggregate memory before running several services
together. A custom configuration needs validation on its target hardware.

## Adapt a sample to the shared stack

Do not copy a complete model-server profile into a sample. It can omit roles
the sample needs, and its `managed` ownership belongs only in the shared stack.
Copy or update the relevant role entries in the sample's active models JSON,
omitting their `deployment` objects. Keep adapter settings and endpoint
credentials needed by the client; server-only credentials stay in the shared
stack's deployment profile.

The standard local stack and the separate `model-servers-nim` stack work with
existing samples' checked-in models JSON. For custom deployments, change only
the roles whose model adapters or endpoints differ. Preserve the sample's other
roles and any required RAG embedding health settings described below.

Sample launchers declare only their application processes; model endpoints
are specified in the client profile. Starting or stopping a sample never
changes the shared model servers. For startup ordering, refer to
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

The `model-servers` launcher does not launch Riva NIMs. Refer to
{doc}`/reference/model-servers-nim` for the NIM stack and its HTTP compatibility
adapters. Direct Riva gRPC clients require the optional Riva SDK dependency and
an explicit models configuration change; that is not an endpoint-only
customization.

## Validate and switch profiles

Before committing a custom profile:

```bash
jq empty model-server-samples/model-servers/yaml/models.my-stack.json

uv run --project tests pytest -q \
  tests/test_model_servers.py \
  tests/test_launcher_config.py
```

Model servers persist after the `model_servers` command reports readiness.
Starting another profile stops persisted services outside the new selection
before launching it. After changing an image, checkpoint, or launch setting,
stop the old stack once so it cannot continue serving stale configuration:

```bash
uv run --project model-server-samples/model-servers model_servers --stop
uv run --project model-server-samples/model-servers \
  model_servers --models model-server-samples/model-servers/yaml/models.my-stack.json
```

For adapter fields and model capabilities, refer to
{doc}`/reference/agent-sdk-models`. For server runtime behavior, persistence,
and NIM credentials, refer to {doc}`/components/ai-services`.
