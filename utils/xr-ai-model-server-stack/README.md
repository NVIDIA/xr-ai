<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Sample model-server stack helpers

Public convenience API for sample-local model-server launchers. Import
`run_model_stack` from `xr_ai_model_server_stack`; configuration composition and
process-building helpers remain private.

This helper is used from a source checkout with editable dependencies, as
configured by each sample's `model-servers/pyproject.toml`. A standalone wheel
installation is not supported: defaults and service projects are repository
resources, not a bundled deployment.

Shared server YAML lives in `yaml/`; sample-specific GPU profiles live beside each
launcher in `agent-samples/<sample>/model-servers/yaml/`. Refer to the
[Python API reference](https://nvidia.github.io/xr-ai/latest/reference/python/xr_ai_model_server_stack/index.html)
and the
[model-server workflow](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html).
