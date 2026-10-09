<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Clef-Flash model server

Standalone launcher for the persistent local Clef SystemOne decision service.
Refer to the
[Clef-Flash guide](https://nvidia.github.io/xr-ai/latest/reference/clef-flash.html) for request
behavior and deployment details.

## Configure

`yaml/clef_server.yaml` selects the pinned model revision, cache directory,
GPU device, context limit, and HTTP port. For example, raise the context limit
after checking available GPU memory:

```yaml
max_length: 8192
```

Set `model_path` only when you have an extracted copy of the pinned snapshot.
Refer to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for the remaining checked-in fields and comments.

## Run

Run all commands from `model-server-samples/clef-flash/`:

```bash
export UV_CONFIG_FILE=../../uv.toml
uv sync
uv run clef_flash_model
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

Stop only this model server with:

```bash
uv run clef_flash_model --stop
```

The stop command fails closed unless the configured port belongs to the
launcher-owned Clef process; it does not stop Docker containers or unrelated
listeners on the same port.
