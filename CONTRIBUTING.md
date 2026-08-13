<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing

Read [`AGENTS.md`](AGENTS.md) for repository constraints,
[`DEPENDENCIES.md`](DEPENDENCIES.md) before changing dependencies, and the
nearest package or sample README for local context.

## External contributors

Before opening a pull request:

1. Open a Bug or Enhancement issue and wait for maintainer approval.
2. Work on a feature branch in your fork.
3. Open the pull request against `main` and add the `contribution` label.
4. Ask a maintainer to trigger CI for a forked pull request.
5. Follow the [Code of Conduct](CODE_OF_CONDUCT.md) and sign off every commit.

## Development setup

Each Python subproject owns its environment:

```bash
cd tests
uv sync
uv run pytest -v
```

Install the repository hooks once per clone:

```bash
uv tool install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

The hooks run Ruff, SPDX checks, and DCO validation. Run the checks relevant to
your change before submitting it; integration tests require a live hub and are
skipped when one is unavailable.

Language-specific toolchains are pinned in each client project. Python supports
3.11 and 3.12, uses type annotations, and is formatted and linted with Ruff.

## Change requirements

- Keep code, tests, dependency metadata, and user-facing docs in the same
  change.
- Update `DEPENDENCIES.md` and the affected `uv.lock` whenever a
  `pyproject.toml` changes.
- Add the repository SPDX header to new source files. See
  [SPDX headers](docs/source/guides/spdx-headers.md).
- Describe the motivation and validation in the pull request and link related
  issues.
- Keep commits focused and sign them with `git commit -s`.

## Developer Certificate of Origin

A signed-off commit certifies that you have the right to submit the work under
this repository's Apache-2.0 license. Commits without a `Signed-off-by` trailer
cannot be accepted. Use:

```bash
git commit -s -m "Describe the change"
```

The sign-off is governed by the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
