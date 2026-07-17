<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI functions for NeMo Agent Toolkit

`xr-ai-nat` provides typed, in-process XR functions for NVIDIA NeMo Agent
Toolkit (NAT). Applications compose these functions directly; process-backed
or MCP compatibility adapters remain separate boundaries.

## Spatial math

The `xr_spatial_math` function group contains deterministic coordinate
operations. Callers supply an explicit `SpatialFrame`, so the functions do not
depend on OpenXR, a tracking service, or an MCP server.

```yaml
functions:
  spatial_math:
    _type: xr_spatial_math
```

The group exposes:

- `spatial_math__position_in_gaze`
- `spatial_math__place_user_relative`
- `spatial_math__place_object_relative`
- `spatial_math__displace_object`
- `spatial_math__move_relative_to`
- `spatial_math__midpoint`
- `spatial_math__place_in_container`

Install the package in the NAT environment so its `nat.plugins` entry point is
discovered automatically.
