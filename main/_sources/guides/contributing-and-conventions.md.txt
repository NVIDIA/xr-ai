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

Public Python references are generated from each enrolled module's literal
`__all__`, declarations, annotations, defaults, and docstrings. An API-only
change therefore updates the code, its co-located documentation, and tests; the
strict documentation build rejects unresolved or undocumented exports. Update
a README or narrative page when concepts, workflows, operations, or
architecture change, and add a migration entry for a breaking change.

The user-facing command catalog is generated from top-level sample
`[project.scripts]` entries and literal `argparse` declarations. Keep option
descriptions in `help=` and do not repeat flag tables in narrative pages.

Sample configuration examples and field-level guidance live in checked-in
YAML/JSON and adjacent YAML comments. The generated catalog enrolls files under
a top-level sample's `yaml/` tree and files beside a direct capability
subproject, then renders them verbatim. Narrative docs should cover only
workflows, operational decisions, credentials, and process relationships.

A `pyproject.toml` change also requires a `DEPENDENCIES.md` update; regenerate
the affected project's gitignored `uv.lock` locally. New source files require
an SPDX header; see [SPDX headers](spdx-headers.md).
