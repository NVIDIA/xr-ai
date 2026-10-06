<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Clef-Flash model server

Standalone launcher for the persistent local Clef SystemOne decision service.
Refer to the
[Clef-Flash guide](../../docs/source/reference/clef-flash.md) for request
behavior and deployment details.

Run these commands from `model-server-samples/clef-flash/`:

```bash
uv --config-file ../../uv.toml sync
uv --config-file ../../uv.toml run clef_flash_model
```

Stop only this model server with:

```bash
uv --config-file ../../uv.toml run clef_flash_model --stop
```

`yaml/clef_server.yaml` selects the pinned model revision, cache directory,
GPU device, context limit, and HTTP port. To raise the context limit after
checking available GPU memory, edit `max_length` in that file. Set `model_path`
there only when you have an extracted copy of the pinned snapshot.
