<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Task capture model servers

This directory contains the convenience launcher and GPU profiles for task
capture's batch STT, TTS, Nemotron Omni LLM, and Cosmos VLM. Refer to the
[model-server workflow](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html)
for startup, shutdown, and configuration composition.

## Configure

`yaml/<gpu-profile>.yaml` overrides shared server defaults. For example, edit
`stt.cuda_visible_devices` in `yaml/dual_48G_ada.yaml` to change speech-model
placement. Refer to the
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for the checked-in settings.

## Run

Run all commands from `agent-samples/task-capture/model-servers/`:

```bash
uv sync
uv run task_capture_model_servers
```

After synchronization, the source entry point also works:

```bash
uv run main.py
```

Stop the configured servers before switching samples or editing server settings:

```bash
uv run task_capture_model_servers --stop
```
