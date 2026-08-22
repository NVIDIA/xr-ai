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

Refer to [Customizing model servers](../../docs/source/guides/customizing-model-servers.md)
for deployment and hardware profiles, GPU placement, credentials, persistence,
and custom configurations.
