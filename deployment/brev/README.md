<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Deployment inventory

`catalog.py` is a read-only helper for deployment tooling. It discovers samples
from their `pyproject.toml` metadata and checks the host and available GPU profiles.
It does not deploy samples, download models, or change host settings.

From the repository root, with Python 3.11 or newer:

```bash
python3 deployment/brev/catalog.py "$PWD"
python3 deployment/brev/catalog.py "$PWD" --resolve auto
```

The first command prints JSON with GPU facts and per-sample blockers. The second
prints the first qualifying sample's ID, project directory, and console command,
separated by tabs; substitute a sample's `project.name` for `auto` to select it.
Resolution exits nonzero when the sample is unknown or blocked. Qualification
does not verify sample-specific dependencies, credentials, or inference readiness.
