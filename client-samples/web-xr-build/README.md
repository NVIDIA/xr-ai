<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Web-XR vendor bundles

Builds the pinned CloudXR and LiveKit ESM bundles into `../web-xr/vendor/` for
same-origin XR and offline-LAN use.

```bash
./build.sh
```

The xr-render orchestrator runs this automatically when needed. See
[Connecting clients](../../docs/source/getting_started/clients.md#web-xr-xr-render-demo)
for cache behavior and dependency updates.
