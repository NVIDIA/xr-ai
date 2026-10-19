<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Deployment inventory

`catalog.py` is a read-only helper for deployment tooling. It discovers samples
from their `pyproject.toml` metadata and checks the host and available GPU profiles.
It does not deploy samples, download models, or change host settings.

From the repository root, using the checkout's stdlib-only launcher project:

```bash
uv run --project utils/xr-ai-launcher python deployment/brev/catalog.py "$PWD"
uv run --project utils/xr-ai-launcher python deployment/brev/catalog.py "$PWD" --resolve simple-vlm-example
```

The first command prints JSON with GPU facts and per-sample blockers. The second
resolves an explicit `project.name` to its ID, project directory, and console
command, separated by tabs. Resolution exits nonzero when the sample is unknown
or blocked. Invalid sample metadata blocks only that sample; other samples
remain resolvable. Descriptions come from each sample's `project.description`.

NVENC support uses NVML's H.264 encoder-capacity query, not a session count.
An unavailable capability query is reported as `null` and blocks qualification
unless another visible GPU reports support. Qualification does not verify
sample-specific dependencies, credentials, or inference readiness.
