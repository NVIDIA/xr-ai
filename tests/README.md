<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai tests

Cross-package coverage for SDKs, services, samples, documentation invariants,
and multi-client and multi-agent routing.

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

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="xr-ai-integration-tests"></a><a id="layout"></a>
<a id="running"></a><a id="gpu-docker-nvenc-tests"></a>
<a id="test-taxonomy"></a><a id="no-cross-talk-guarantee"></a>

Refer to [Testing](https://nvidia.github.io/xr-ai/latest/guides/testing.html) for suite scope, markers, CI,
routing guarantees, and guidance for adding tests.
