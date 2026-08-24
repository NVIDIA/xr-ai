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
server YAML file. The
[customization guide](../../docs/source/guides/customizing-model-servers.md)
explains how to copy a profile, size GPU budgets, change endpoints, and validate
the result. The generated
[configuration reference](../../docs/source/reference/configuration.rst) lists
the checked-in fields and comments.

## Run

Run all commands from `agent-samples/model-servers/`:

```bash
uv sync
uv run model_servers
```

The first cold start downloads model weights and can take tens of minutes.
Configure the credentials described in the
[credentials reference](../../docs/source/getting_started/credentials.md)
before starting the stack.

To use the self-hosted NIM profile instead of the default profile:

```bash
uv run model_servers --models vlm_llm_nim
```

The services persist across agent-sample restarts. Stop them explicitly when
they are no longer needed:

```bash
uv run model_servers --stop
```

Refer to [AI services](../../docs/source/components/ai-services.md) for model
runtime behavior, persistence, health checks, and service-specific notes.
