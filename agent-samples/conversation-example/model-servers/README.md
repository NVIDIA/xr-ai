<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# conversation-example model servers

This directory contains the convenience launcher and GPU tuning for this sample's
local inference stack, including speaker STT, TTS, one Omni instance for
conversation and vision, and the sample's Clef router. Refer to the
[model-server workflow](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html)
for startup, shutdown, switching samples, and configuration composition.

When `../yaml/conversation_example_worker.yaml` selects a Gemma 4 model
profile, the launcher starts one Gemma 4 server for both conversation and
vision, D3 routing, speaker STT, and TTS. Refer to the
[model profiles](https://nvidia.github.io/xr-ai/latest/reference/conversation-example.html#model-profiles)
for the shared worker/server selection and Docker requirements.

## Configure

`yaml/<gpu-profile>.yaml` contains sample-specific placement and workload budgets
for the original Omni stack; `yaml/gemma4.yaml` holds portable Gemma 4/D3 settings.
For example, edit `cuda_visible_devices` for a server in `yaml/dual_48G_ada.yaml`
to change its GPU placement. Service defaults live in each server's project under
`services/`. Use `--gpu-profile NAME` for an explicit profile on both start and
stop for Omni. Gemma uses one portable configuration and selects one GPU by
free memory. Refer to the model profile guide above for placement and restart
requirements.

Refer to the
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for checked-in settings.

## Run

Run all commands from `agent-samples/conversation-example/model-servers/`:

```bash
uv sync
uv run conversation_example_model_servers
```

After synchronization, the source entry point also works:

```bash
uv run main.py
```

Stop the configured servers before switching samples or editing server settings:

```bash
uv run conversation_example_model_servers --stop
```
