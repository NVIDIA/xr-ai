<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai tests

Cross-package coverage for SDKs, services, samples, documentation invariants,
and multi-client/multi-agent routing.

```bash
cd tests
uv sync
uv run pytest -v
```

GPU-, Docker-, and NVENC-dependent tests are excluded from CI and can be run
from the repository root:

```bash
bash tests/run_local_gpu_tests.sh
```

See [Testing](../docs/source/guides/testing.md) for suite scope, markers, CI,
routing guarantees, and guidance for adding tests.
