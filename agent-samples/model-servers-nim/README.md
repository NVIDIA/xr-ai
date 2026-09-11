<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Shared model servers with NVIDIA NIM

This independent sample launches shared NVIDIA NIM models with Magpie TTS,
using the existing NIM launcher and a sample-local embedding adapter. Spark
uses the original NeMo STT service because Parakeet NIM lacks an ARM64 image.
Refer to the [sample guide](https://nvidia.github.io/xr-ai/latest/reference/model-servers-nim.html)
for endpoints, consumer setup, memory allocation estimates, hardware validation
limits, and lifecycle tests. Complete stack qualification remains outstanding
on all three hardware profiles.

## Configure

`yaml/<gpu-profile>/models.json` selects model adapters and deployments.
The adjacent server YAML files own container digests, GPU placement, ports,
model profiles, and runtime limits. For example, reduce Cosmos concurrency in
`yaml/96G_blackwell/nim_vlm_server.yaml`:

```yaml
env:
  NIM_MAX_NUM_SEQS: "2"
```

Refer to the [sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/model-servers-nim.html#configure)
and generated [configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for the remaining settings. Stop and restart the stack after configuration changes.

## Run

Run all commands from `agent-samples/model-servers-nim/`:

```bash
uv sync
uv run model_servers_nim --gpu-profile 96G_blackwell --dry-run
uv run model_servers_nim
```

Docker with NVIDIA Container Toolkit and `NGC_API_KEY` are required; Spark also
uses `HF_TOKEN`. Refer to the
[credentials reference](https://nvidia.github.io/xr-ai/latest/getting_started/credentials.html).
First startup can take tens of minutes. After the launcher exits, start the
consuming sample from the same terminal.

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

GPU detection is automatic. Select a profile explicitly when needed:

```bash
uv run model_servers_nim --gpu-profile dual_48G_ada
uv run model_servers_nim --gpu-profile spark
```

When switching from the original stack, stop it first because the stacks share ports:

```bash
uv run --project ../model-servers model_servers --stop
uv run model_servers_nim
```

Export the client configuration for a consuming sample, or stop this stack:

```bash
uv run model_servers_nim --export-models yaml/models.reused.json
uv run model_servers_nim --stop
```
