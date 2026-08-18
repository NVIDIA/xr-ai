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
4. Ask a maintainer to comment `/build-ci` to trigger CI for a forked pull
   request. Verify the relevant checks locally before opening the PR.
5. Follow the [Code of Conduct](CODE_OF_CONDUCT.md) and sign off every commit.

Maintainers aim to provide an initial response within five business days.
External review cycles may take longer; respond promptly to requested changes.

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

Language-specific toolchains are pinned in each client project:

- Python supports 3.11 and 3.12, uses type annotations, and is formatted and
  linted with Ruff.
- Swift uses the `swift-tools-version` pinned by each `Package.swift` and
  Xcode's default formatting.
- Kotlin uses the versions in `gradle/libs.versions.toml` and the official
  Kotlin style.
- Web clients use plain JavaScript ES modules and keep dependencies minimal.

## Change requirements

- Keep code, tests, dependency metadata, and user-facing docs in the same
  change.
- After changing a `pyproject.toml`, run
  `python3 .github/scripts/generate_dependency_map.py` and regenerate the
  affected project's gitignored `uv.lock` locally. Do not hand-edit the
  generated dependency inventory; pre-commit updates it and CI rejects drift.
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
The required text is reproduced verbatim:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license
document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right
to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my
knowledge, is covered under an appropriate open source license and I have the
right under that license to submit that work with modifications, whether
created in whole or in part by me, under the same open source license (unless I
am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who
certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public
and that a record of the contribution (including all personal information I
submit with it, including my sign-off) is maintained indefinitely and may be
redistributed consistent with this project or the open source license(s)
involved.
```
