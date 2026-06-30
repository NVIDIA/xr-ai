<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai — Claude Working Instructions

Read `AGENTS.md` before making any changes. It is the authoritative source for
architecture, process model, conventions, and documentation rules.

Read `DEPENDENCIES.md` before and after any change that touches a `pyproject.toml`.
Update it in the same commit — not as a follow-up.

Sub-repos may have their own `CLAUDE.md` with module-specific context — read
those before working inside them.

Before committing any Python changes, run:

```
uv tool run ruff check --fix <changed_files>
```

Every commit must have a `Signed-off-by` trailer — use `git commit -s`.
See the DCO section in `AGENTS.md`.

After pushing to a PR branch, a PR is not done until CI is green:

```
gh pr checks <number>
```

Run this after every push. Fix any failures before reporting the PR as ready.
Only Python files need ruff. Only commits need DCO. CI catches both.
