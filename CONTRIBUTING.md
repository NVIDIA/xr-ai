<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing

Thanks for your interest in contributing! This document outlines how to build,
test, and submit changes.

Before making any change, read the authoritative working docs at the repo root:

- [`AGENTS.md`](AGENTS.md) — despite the name (which follows the
  [agents.md](https://agents.md) convention), this is the working-conventions
  doc for **both human developers and AI assistants**. It covers architecture,
  the process model, sample layout, license-header rules, and the change log.
- [`DEPENDENCIES.md`](DEPENDENCIES.md) — the authoritative dependency map. Any
  change to a `pyproject.toml` must update `DEPENDENCIES.md` in the same commit.

Sub-projects may have their own `README.md` with module-specific context — read
those before working inside them.

## External Contributors

If you are not an NVIDIA employee, please follow this workflow before opening a
pull request:

1. **Open an issue first.** File a Bug or Enhancement issue describing your
   intended change. NVIDIA maintainers will triage it and confirm whether a PR
   would be accepted before you invest time writing code. Any contribution that
   has not been pre-approved via an issue may be closed without review.

2. **Fork the repository.** Work on a feature branch in your personal fork; do
   not request direct branch access to this repository.

3. **Open a PR from your fork** targeting `main`. In the PR sidebar, add the
   **`contribution`** label — this flags the PR as an external contribution so
   maintainers can route it correctly.

4. **CI requires a maintainer trigger.** GitHub Actions will not run
   automatically on PRs from forks. An NVIDIA team member must comment
   `/build-ci` (or equivalent) to start the test suite. Do not open a PR until
   your changes pass locally (see [CI](#ci) below).

5. **Follow all rules of engagement** — code style, license headers, DCO
   sign-off (see [Signing Your Work](#signing-your-work)), and the
   [Code of Conduct](CODE_OF_CONDUCT.md). Contributions that violate these
   rules, or which are submitted without a signed-off issue approval, will be
   closed.

> **Response SLA.** Maintainers aim to give an initial response within 5
> business days. Review cycles for external PRs may be longer than for internal
> work; please be patient and respond promptly to any requested changes.

## Code Style

- Use meaningful, descriptive names for variables, functions, and types in all languages.
- Write short docstrings for public modules, classes, and functions.
- Write clear, easy-to-read, and maintainable code.
- Keep code warnings and linter errors to a minimum.
- In general, prefer clarity over clever tricks, and keep the codebase friendly for contributors.

**Python** (`server-runtime/`, `agent-sdk/`, `utils/`, `ai-services/`,
`agent-mcp-servers/`, `agent-samples/`, `cloudxr-runtime/`, `tests/`)
- Target Python 3.11+ (CI matrix runs 3.11 and 3.12).
- Follow PEP 8 for style.
- Use type annotations (PEP 484) and prefer formatted string literals (f-strings).
- Use `uv` for environment and dependency management — every Python sub-project
  is its own uv project (`uv sync` in the project directory).

**Swift** (`client-samples/ios-visionos/`)
- Use the Swift toolchain pinned by `// swift-tools-version:` in `Package.swift`.
- Stick to Xcode's default formatting.

**Kotlin** (`client-samples/android/`)
- Use the Kotlin / Android Gradle Plugin versions pinned in
  `gradle/libs.versions.toml`.
- Follow the Kotlin official style.

**JavaScript** (`client-samples/web/`)
- Plain ES modules, no build step. Keep dependencies minimal.

## License headers

This repository uses [REUSE](https://reuse.software/) / SPDX headers on every
source file we own. Every new source file must start with:

```
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Use the comment syntax for the file's language and place the header at the top,
after any required first-line directive (`#!/...` shebangs, `<?xml ...?>`,
`<!DOCTYPE ...>`, or Swift's `// swift-tools-version:`):

| Style | Used for |
|---|---|
| `# ...` | `.py`, `.yaml`/`.yml`, `.toml`, `.properties`, `.sh`, `.pro`, `.gitignore`, `.gitattributes`, `requirements.txt` |
| `// ...` | `.swift`, `.kt`/`.kts`, `.js`, `.ts`/`.tsx` |
| `<!-- ... -->` | `.xml`, `.html`, `.plist`, `.entitlements`, `.md` |

The Apache-2.0 license text lives in [`LICENSE`](LICENSE).

To add or fix headers, install the [reuse tool](https://github.com/fsfe/reuse-tool):

```bash
uv tool install reuse
```

and run for example:

```bash
reuse annotate -t compact -l Apache-2.0 --skip-unrecognised -r path/to/file
```

REUSE does not auto-update the copyright year when you touch a file; include the
current year when adding or editing headers.

Skip files that can't carry comments or aren't ours to license: `LICENSE`,
`*.json`, `*.resolved`, binary assets (e.g. `*.gif`), `.gitkeep` markers,
Xcode-managed files (`*.pbxproj`, `*.xcworkspacedata`), and third-party Gradle
wrapper files (`gradlew`, `gradlew.bat`, `gradle/wrapper/gradle-wrapper.properties`).

## CI

GitHub Actions runs the IPC + multi-client / multi-agent suite under `tests/`
on every push and pull request, across Python 3.11 and 3.12. The suite is
hub-on-loopback only — no Docker, LiveKit, or NVENC required.

Run it locally:

```bash
cd tests
uv sync
uv run pytest -v
```

Integration tests marked `-m integration` are auto-skipped in CI because no hub
is started there; run them locally against a live hub when relevant.

## Pull Requests

1. Create a feature branch.
2. Update `AGENTS.md`, the relevant `README.md`, and `DEPENDENCIES.md` in the
   same commit as the code change — a change is not done until the docs reflect it.
3. Ensure builds and tests pass locally and in CI.
4. Describe motivation, changes, and testing in the PR.
5. Link related issues.

## License

- Your contributions are under the repository's license (Apache-2.0,
  see [`LICENSE`](LICENSE)) unless stated otherwise.

### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
