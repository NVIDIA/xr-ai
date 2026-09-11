<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Shared model servers with NVIDIA NIM

This independent sample starts the same model families as `model-servers`,
with Magpie replacing Pocket TTS. It reuses `services/nim-server` for NIM
container lifecycle and `services/stt-server` for the Spark STT fallback.
The launcher waits for readiness, then exits with the model servers running.

| Role | Model | Endpoint |
|---|---|---|
| STT | Parakeet TDT 0.6B v3 | Riva gRPC `localhost:50051`; Spark HTTP `localhost:8103` |
| TTS | Magpie TTS Multilingual | Riva gRPC `localhost:50052` |
| LLM and agent LLM | Nemotron 3 Nano Omni 30B A3B Reasoning | HTTP `localhost:8108` |
| VLM | Cosmos3 Nano Reasoner | HTTP `localhost:8100` |
| Embedding | Llama Nemotron Embed 1B v2 | HTTP `localhost:8109` |

The embedding endpoint includes a sample-local adapter. It translates XR AI's
`query: ` and `passage: ` prefixes to NIM's input types, preserving input order
and avoiding duplicated prefixes. Unprefixed strings default to passages;
callers can also set `input_type`. Native embedding NIM access is on port 8119.
Speech NIM HTTP ports 9010 and 9011 are used for readiness; speech clients use
the gRPC ports. This sample does not provide a Pocket-compatible TTS endpoint.

## Run

Run all commands from `agent-samples/model-servers-nim/`:

```bash
uv sync
uv run model_servers_nim --gpu-profile 96G_blackwell --dry-run
uv run model_servers_nim
```

Docker with NVIDIA Container Toolkit and `NGC_API_KEY` are required. Spark also
uses `HF_TOKEN` for its local STT fallback. Credentials can be exported in the
environment or loaded from the repository's existing credential store.
Refer to the [credentials reference](https://nvidia.github.io/xr-ai/latest/getting_started/credentials.html).
First startup downloads container images and model engines and can take tens
of minutes. The caches persist under the repository's `models/` directory.

The default GPU detection is shared with `model-servers`. Explicit selections:

```bash
uv run model_servers_nim --gpu-profile 96G_blackwell
uv run model_servers_nim --gpu-profile dual_48G_ada
uv run model_servers_nim --gpu-profile spark
```

These are alternative stacks and share model ports and GPU capacity with
`model-servers`. Stop the active stack before switching. For example, when
switching from the original sample:

```bash
uv run --project ../model-servers model_servers --stop
uv run model_servers_nim
```

To stop this stack before editing its configuration or switching back:

```bash
uv run model_servers_nim --stop
```

Cleanup uses the ports declared in this sample's YAML, including the embedding
adapter and the Spark fallback. Run it before changing ports. It does not
remove downloaded images or model caches. Do not use it while another stack
owns those same ports.

## Use with an agent sample

Export the adapters, model IDs, and endpoints for the selected hardware:

```bash
uv run model_servers_nim --export-models yaml/models.reused.json
```

The export requires no credentials or running models. Add `--gpu-profile NAME`
to export for a different machine. Copy the needed entries from the export into
the consuming sample's `yaml/models.json`, preserving its other model roles.
The exported entries use `ownership: reused` and do not give the consumer
responsibility for starting these servers or holding NGC credentials. For
example, `simple-vlm-example` needs `stt`, `tts`, and `vlm`.

Riva clients require the existing `xr-ai-models[riva]` extra **in the consuming
worker's environment**. For a local simple-VLM trial, synchronize the worker
and add its optional Riva client before starting that sample:

```bash
uv sync --project ../simple-vlm-example/worker
uv pip install --python ../simple-vlm-example/worker/.venv/bin/python nvidia-riva-client
uv run --project ../simple-vlm-example simple_vlm_example
```

A later `uv sync` can remove the manually installed extra. For a maintained
consumer, declare `xr-ai-models[riva]` in its worker dependencies. The new sample
does not edit existing workers, SDK presets, or agent configurations.

## Configuration and memory

`yaml/<gpu-profile>/models.json` selects the deployment and client adapters.
The adjacent server YAML files own image versions and digests, model profiles, ports, GPU
placement, and runtime limits. `--models PATH` selects a custom deployment JSON
using the selected hardware directory. For example, reduce Omni concurrency in
`nim_llm_server.yaml` by changing its `NIM_PASSTHROUGH_ARGS` entry:

```yaml
    --max-num-seqs 2
```

| Hardware | GPU placement | Initial allocation plan |
|---|---|---|
| 96 GB Blackwell | All models on GPU 0 | Omni 35%, Cosmos 24%, plus speech and embedding engines |
| Two 48 GB Ada GPUs | Speech and Cosmos on GPU 0; Omni and embeddings on GPU 1 | Cosmos 36% on GPU 0; Omni 80% on GPU 1 |
| DGX Spark | All models on GPU 0 | Omni 25% with a fixed 2 GiB cache; Cosmos 22%; shared system memory |

These fractions budget **weights, runtime, and cache together**, relative to
total GPU memory. Magpie uses the pinned 1.10.0 `batch_size=8` engine
(approximately 13 GiB), and Parakeet NIM uses approximately 14 GB. The actual
Magpie manifest has no batch-size-one profile, despite older documentation
for this tag listing one. Embeddings use one FP8 engine on Blackwell and Ada
(approximately 3–4 GiB). The dual-Ada plan leaves roughly 4 GB on GPU 0 and
5 GB on GPU 1 beyond these estimates. It does not reserve capacity for a large
renderer or unrelated GPU workloads.

Omni is pinned to its generic TP=1 NVFP4 profile on Blackwell and Spark and
its FP8 profile on Ada. Cosmos selects Nano, with an 8,192-token context and
four image inputs. Ada explicitly selects the FP8 vLLM checkpoint profile
shipped under L40S, which shares SM 8.9 with RTX 6000 Ada; this profile contains
weights rather than a GPU-specific TensorRT engine. The generic BF16 fallback
would exceed the shared-GPU budget.
Omni has a 32,768-token context and at most four sequences. Graph capture is
disabled for these shared-GPU profiles to reduce startup and runtime memory.

Parakeet v3 NIM 1.3.0 has no ARM64 image, so Spark uses the original NeMo STT
service and configuration. The other selected images have ARM64 manifests.
Spark uses the generic BF16 Cosmos profile and generic FP16 ONNX embedding
profile; that embedding profile has a **4,096-token** input limit. ARM64 image
availability is distinct from validation on GB10: the complete Spark stack
still needs hardware qualification, including cold-cache startup. Cosmos NIM
uses fractional cache sizing and does not inherit the original sample's Spark
prefetch and fixed-cache safeguards.

Whole-stack startup, simultaneous load, and restart still need validation on
all three hardware profiles. The defaults are allocation estimates, not a
claim of measured full-stack support. Refer to the
[model-server customization guide](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html)
for the shared lifecycle and profiling workflow.

Container and profile references:

- [Cosmos3 Reasoner 1.7.0](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/cosmos3-reasoner/1.7.0)
- [Omni NIM support matrix](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/support-matrix.html)
- [Magpie engine memory](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html)
- [Parakeet v3 selection and Spark restrictions](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)
- [Embedding 1.13.0 engines and memory](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/1.13.0/support-matrix.html)

## Validate

The CPU checks load every profile with the real SDK, construct Docker commands,
check startup/export/stop behavior, and exercise prefix-aware embedding calls:

```bash
uv run --extra test python -m pytest tests -q
```

After startup, check every endpoint through the same SDK that agents use:

```bash
uv run model_servers_nim --export-models yaml/models.reused.json
uv run --extra test python smoke_test.py --models yaml/models.reused.json
```

This exercises a Magpie-to-Parakeet speech round trip, both LLM roles, a real
image request, and query/passage embeddings. Repeat after a cold start and a
warm restart on each target machine; monitor GPU memory under concurrent agent
traffic as well as during startup.

All implementation, configuration, and tests are inside this sample. The sole
repository-wide change is the generated `DEPENDENCIES.md` inventory required
when adding these Python projects.
