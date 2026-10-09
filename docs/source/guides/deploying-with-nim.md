<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Deploying samples with NIM endpoints

Develop samples with the local `model-servers` profile, then configure the same
workers to consume independently deployed NVIDIA NIM endpoints. Samples use
`xr_ai_models` factories and typed service interfaces in both environments.
They do not launch model containers or require local compatibility proxies.
The local server defaults and sample model profiles remain the development path.

## Ownership and protocol boundaries

| Responsibility | Owner |
|---|---|
| Sample logic and typed model calls | Application worker |
| Protocol serialization, credentials, response normalization, client cleanup | `xr_ai_models` |
| Model container images, optimized engines, GPU allocation, caching, scaling, upgrades | Deployment operator |
| Reachable endpoint addresses, TLS termination, inference authorization | Deployment operator |

A profile describes the client adapter and the endpoint separately. The adapter
selects the wire protocol; the endpoint describes connectivity. NIM is a serving
implementation, not a new model role. Local and deployment profiles use the same
`make_llm`, `make_vlm`, `make_stt`, `make_tts`, and `make_embedding` factories.

Chat and image inference use OpenAI-compatible HTTP. The speech NIMs used here
expose Riva gRPC, so speech entries select the existing `riva_grpc` adapter.
Embedding NIMs use OpenAI-compatible HTTP with a batch-wide `input_type` field.
That difference is handled inside the embedding client with existing request
defaults, rather than another server process. Refer to
{doc}`/reference/agent-sdk-models` for the profile contract.

## Deploy the required models

Deploy only the models the application uses. Endpoints can run on one machine,
separate GPU hosts, or a Kubernetes cluster. XR AI does not prescribe a shared
GPU layout or manage NIM container lifecycle. Refer to NVIDIA's
[NIM deployment reference implementations](https://github.com/NVIDIA/nim-deploy),
[Speech NIM deployment guide](https://docs.nvidia.com/nim/speech/latest/deployment/index.html),
and the deployment guide for each selected model and container version.

The example in `deployment/nim/models.yaml` selects Nemotron Omni for both LLM
roles, Cosmos3 for vision, Parakeet for recognition, Magpie for synthesis, and
Llama Nemotron Embed for retrieval. These are endpoint configuration examples,
not a claim of production or hardware qualification. Match model IDs, supported
languages, voices, input limits, and inference options to the deployed versions.
Changing the serving model can change sample behavior; validate the application's
prompts and evaluations as well as its transport.

Container-pull and engine-download credentials belong in the deployment
environment. The agent needs only credentials for inference, when the endpoint
requires them. The example uses `NIM_ENDPOINT_API_KEY`; each role can name a
different environment variable. Set the value in the agent's environment; never
put tokens in profiles. The standalone smoke script reads exported variables
and does not load the launcher's credential store.

## Configure a consumer profile

Run the following commands from the repository root. Copy the example to a
private path and replace the `example.com` addresses before making requests:

```bash
cp deployment/nim/models.yaml /tmp/models.nim.yaml
```

HTTP `base_url` is the service root, excluding `/v1`. Riva gRPC uses `host:port`,
without `http://`, `https://`, or a route. Use `use_ssl: true` for a TLS-enabled
gRPC endpoint. The speech gateway must carry gRPC traffic, not translate it into
OpenAI audio routes. `api_key_env` names a bearer-token variable for either
protocol; hosted NVCF speech additionally needs its model's `function_id`.

Each entry has `deployment.ownership: external`. The deployment remains owned
by its operator when the sample exits or restarts. This file is a consumer
profile; do not pass it to `model_servers --models`, which selects the local
services to start and stop.

The checked-in voice workers include the existing `xr-ai-models[riva]` extra.
Applications copied from an earlier sample must enable that extra in their
worker's dependency before selecting Riva speech. The SDK base installation
continues to defer the vendor dependency until a Riva factory is selected.

### Embedding inputs

For an asymmetric HTTP embedding endpoint, configure the existing adapter
request defaults:

```yaml
adapter:
  preset: nemotron_embedding
  model_name: nvidia/llama-nemotron-embed-1b-v2
  default_extras:
    input_type: passage
```

This labels unprefixed inputs as passages. Inputs beginning with `query: ` or
`passage: ` select their own type; the SDK strips that label once, groups mixed
batches by type, sends the native model ID and `input_type`, and restores input
order. The RAG service can keep its existing query and passage prefixes.
Local profiles omit `input_type`, preserving their original text payloads.
Refer to NVIDIA's
[embedding API reference](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/1.13.0/reference.html).

### Readiness and voice behavior

Set HTTP `health_path` to the deployed readiness route, normally
`/v1/health/ready` for these NIMs. A gateway that exposes inference but no health
route can use `readiness: none`; this skips the health probe, not inference.
Riva `health()` checks gRPC channel connectivity, which does not prove model
inference or authorization. Run the smoke check and application validation.
Worker startup follows the existing {ref}`consumer-model-readiness` policy;
tea making's RAG service separately checks embedding health.

Direct Magpie synthesis produces mono 16-bit PCM at the configured sample rate,
with streaming and cancellation handled by the SDK. Recognition timeout and
cancellation also cancel the native RPC and join its pending client read before
returning. It does not append the old
HTTP wrapper's 300 ms pause. Test sentence transitions, interruption, and the
listening chime in the consuming sample. Voice choice and speaking cadence
remain model-specific, even though the worker interface is the same.

## Validate the endpoints

From the repository root, run the smoke check in the model SDK environment.
The speech extra and Python audio compatibility package are its only additions:

```bash
uv --config-file uv.toml run --project agent-sdk/xr-ai-models --extra riva \
  --with 'audioop-lts; python_version >= "3.13"' python deployment/nim/smoke_test.py \
  --models /tmp/models.nim.yaml
```

The script starts no containers and owns only the clients it creates. It fails
on missing credentials, failed enabled health probes, and failed inference.
It exercises configured roles: synthesized WAV and PCM streaming, a speech round
trip when both speech roles exist, chat and image responses, text streams, LLM
function calls when declared, and query and passage embeddings. An STT-only
profile reports that recognition inference was skipped; a channel-ready result
alone does not qualify speech. Small embedding checks report actual dimensions;
configure RAG dimensions and rebuild its index when changing embedding models.

Remove unused role entries before validation. The smoke check is an inference
sanity check, not a load test or an application evaluation. Qualify cold start,
restart, concurrent traffic, GPU memory, model limits, gateway timeouts, and
application latency in the deployment environment.

## Switch an existing sample

Keep a copy of the sample's active local JSON. Replace only the roles it already
consumes with entries from the customized endpoint profile. This example adapts
simple VLM; run it from the repository root after validation:

```bash
cp agent-samples/simple-vlm-example/yaml/models.json /tmp/simple-vlm.models.local.json
uv --config-file uv.toml run --project agent-sdk/xr-ai-models python - <<'PYTHON'
import json
from pathlib import Path
import yaml
source = yaml.safe_load(Path('/tmp/models.nim.yaml').read_text())['models']
target = Path('agent-samples/simple-vlm-example/yaml/models.json')
active = json.loads(target.read_text())
active['models'] = {name: source[name] for name in active['models']}
target.write_text(json.dumps(active, indent=2) + '\n')
PYTHON
uv --config-file uv.toml run --project agent-sdk/xr-ai-models --extra riva \
  --with 'audioop-lts; python_version >= "3.13"' python deployment/nim/smoke_test.py \
  --models agent-samples/simple-vlm-example/yaml/models.json
uv run --project agent-samples/simple-vlm-example simple_vlm_example
```

| Sample | Active profile | Required roles |
|---|---|---|
| Simple VLM | `yaml/models.json` | `stt`, `vlm`, `tts` |
| Lab instrument monitoring | `yaml/models.json` | `llm`, `vlm`, `stt`, `tts` |
| XR render demo | `yaml/models.json` | `llm`, `agent_llm`, `vlm`, `stt`, `tts` |
| Tea making | `yaml/models.local.json` | `llm`, `vlm`, `stt`, `tts`, `embedding` |

Simple VLM, lab monitoring, and XR render resolve `models.json` beside their
worker YAML. Tea making's worker and RAG YAML each select `models_config`;
update both when choosing a different profile path. Paths are relative to the
respective YAML. Keep application-specific extra roles when adapting other
samples. No sample launcher option or model-serving process is needed.

To return simple VLM to local development, restore its saved profile, start
`model_servers`, and restart the sample. Restoring a profile does not stop remote
NIMs; the deployment operator controls them.

## Removed managed NIM stack

The former `model_servers_nim` command, hardware profiles, engine export cache,
and HTTP compatibility services are removed. They duplicated infrastructure
lifecycle and model adapter behavior. Deploy the native NIM services with the
operator's tooling, then replace local aliases such as `llm`, `vlm`, and `embed`
with native model IDs in a consumer profile. Refer to
{doc}`/reference/migrations` for the removed commands and services.
