<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Deployment inventory

`catalog.py` is a read-only helper for deployment tooling. It discovers samples
from their `pyproject.toml` metadata and checks the host and available GPU profiles.
It does not deploy samples, download models, or change host settings.

From the repository root, using the checkout's launcher and the same video
codec package required by DeviceIOHub:

```bash
uv run --project utils/xr-ai-launcher --with 'PyNvVideoCodec>=2.2' python deployment/brev/catalog.py "$PWD"
uv run --project utils/xr-ai-launcher --with 'PyNvVideoCodec>=2.2' python deployment/brev/catalog.py "$PWD" --resolve simple-vlm-example
```

The first command prints JSON with GPU facts and per-sample blockers. The second
resolves an explicit `project.name` to its ID, project directory, and console
command, separated by tabs. Resolution exits nonzero when the sample is unknown
or blocked. Invalid sample metadata blocks only that sample; other samples
remain resolvable. Descriptions come from each sample's `project.description`.

H.264 NVENC availability is queried through PyNvVideoCodec in a short-lived
subprocess per GPU, selected by UUID. This briefly initializes encoder hardware;
it does not load models or record media. NVML statistics are not used.
An explicit unsupported-device response reports `nvenc: false`; missing packages,
libraries, timeouts, and other failures report `null`. Details are retained in
`nvenc_error`. Qualification is blocked unless a GPU reports success. These
checks do not verify sample-specific dependencies, credentials, or inference readiness.
