<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-render-demo model servers

This directory contains the convenience launcher and GPU tuning for this sample's
local inference stack. Refer to the
[model-server workflow](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html)
for startup, shutdown, switching samples, and configuration composition.

## Configure

`yaml/<gpu-profile>.yaml` contains sample-specific placement and workload budgets.
For example, edit `cuda_visible_devices` for a server in `yaml/dual_48G_ada.yaml`
to change its GPU placement. Ports, models, and images
come from each service's own YAML under `services/`. Use `--gpu-profile NAME` for an explicit
profile on both start and stop.

Refer to the
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for checked-in settings.

## Run

Run all commands from `agent-samples/xr-render-demo/model-servers/`:

```bash
uv sync
uv run xr_render_demo_model_servers
```

After synchronization, the source entry point also works:

```bash
uv run main.py
```

Stop the configured servers before switching samples or editing server settings:

```bash
uv run xr_render_demo_model_servers --stop
```
