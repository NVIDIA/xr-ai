<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Sample model-server stack helpers

Internal configuration composition for sample-local convenience launchers. Shared
server YAML lives in `yaml/`; sample-specific GPU profiles live beside each
launcher in `agent-samples/<sample>/model-servers/yaml/`. This utility has no
public Python API or standalone command. Refer to the
[model-server workflow](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html).
