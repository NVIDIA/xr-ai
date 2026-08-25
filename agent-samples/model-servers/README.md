<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Shared model servers

This sample starts the reusable inference services consumed by the agent
samples. The default deployment includes Parakeet STT, Piper TTS, Nemotron-3
Nano Omni, Cosmos3 Nano Reasoner, and the Nemotron embedding model.

The launcher waits for every selected service to become healthy. It then prints
the `All processes ready` banner and exits, while the services remain running
with their weights warm. You can start an agent sample from the same terminal
after the command returns.

## Configure

Model configuration has two layers:

| What to change | Where |
|---|---|
| Which services and endpoints belong to the stack | `yaml/models.<name>.json` |
| Model, port, GPU placement, memory budget, cache, and runtime options | `yaml/<gpu-profile>/*.yaml` |

The launcher detects the GPU profile and reads its YAML automatically. To tune
a server, edit the matching file before starting the stack. For example,
`yaml/96G_blackwell/vlm_server.yaml` owns the Cosmos VLM settings on a 96 GB
Blackwell GPU; changing `max_images_per_prompt` there changes the request limit
for that server:

```yaml
max_images_per_prompt: 2
```

Use `--models NAME` to select a deployment JSON and
`--gpu-profile NAME` only to select a reviewed hardware YAML directory.

Because model processes persist, stop and restart the stack after changing a
server YAML file. Refer to the
[customization guide](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html) for
instructions to copy a profile, size GPU budgets, change endpoints, and
validate the result. Refer to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html) for
the checked-in fields and comments.

## Run

Run all commands from `agent-samples/model-servers/`:

```bash
uv sync
uv run model_servers
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

The first cold start downloads model weights and can take tens of minutes.
Refer to the
[credentials reference](https://nvidia.github.io/xr-ai/latest/getting_started/credentials.html) and
configure the required credentials before starting the stack.

To use the self-hosted NIM profile instead of the default profile:

```bash
uv run model_servers --models vlm_llm_nim
```

The services persist across agent-sample restarts. Stop them explicitly when
they are no longer needed:

```bash
uv run model_servers --stop
```

Refer to [AI services](https://nvidia.github.io/xr-ai/latest/components/ai-services.html) for model
runtime behavior, persistence, health checks, and service-specific notes.
