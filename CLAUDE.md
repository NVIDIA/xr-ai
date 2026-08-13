<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Working instructions

Follow [`AGENTS.md`](AGENTS.md) and the nearest package or sample README.
Update `DEPENDENCIES.md` with any `pyproject.toml` change and regenerate the
affected project's gitignored `uv.lock` locally. Run relevant tests and Ruff
before committing, and sign commits with `git commit -s`. After every push,
check `gh pr checks`; the change is not complete until required CI is green.
