<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Contributing and conventions

The root [`CONTRIBUTING.md`](https://github.com/NVIDIA/xr-ai/blob/main/CONTRIBUTING.md)
explains setup, testing, pull requests, and DCO sign-off.
[`AGENTS.md`](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md) contains the
architecture and change constraints used by both humans and coding agents.
[`DEPENDENCIES.md`](https://github.com/NVIDIA/xr-ai/blob/main/DEPENDENCIES.md)
is the package dependency map.

For a Python change:

```bash
cd <affected-project>
uv sync
uv run pytest
uv tool run ruff check <changed-python-files>
```

Update the relevant README and this documentation with user-visible changes.
A `pyproject.toml` change also requires a `DEPENDENCIES.md` update; regenerate
the affected project's gitignored `uv.lock` locally. New source files require an
SPDX header; see
[SPDX headers](spdx-headers.md).
