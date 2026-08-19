<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Contributing and conventions

The conventions every change to xr-ai must satisfy. The authoritative
sources live at the repository root and override anything summarized here:

- [`CONTRIBUTING.md`](https://github.com/NVIDIA/xr-ai/blob/main/CONTRIBUTING.md)
  — how to build, test, and submit changes, plus the PR process and the
  Developer Certificate of Origin (DCO) sign-off requirement.
- [`AGENTS.md`](https://github.com/NVIDIA/xr-ai/blob/main/AGENTS.md) — despite
  the name (which follows the [agents.md](https://agents.md) convention), this
  is the working-conventions doc for **both human developers and AI
  assistants**: architecture, the process model, sample layout, license-header
  rules, and the change log.

## Code style

- Use meaningful, descriptive names for variables, functions, and types in all
  languages.
- Write short docstrings for public modules, classes, and functions.
- Prefer clarity over clever tricks; keep code warnings and linter errors to a
  minimum.

Per-language specifics:

| Language | Locations | Conventions |
|---|---|---|
| Python | `server-runtime/`, `agent-sdk/`, `utils/`, `ai-services/`, `agent-mcp-servers/`, `agent-samples/`, `cloudxr-runtime/`, `tests/` | Target Python 3.11+ (CI runs 3.11 and 3.12); follow PEP 8; use type annotations and f-strings; manage environments with `uv` (each sub-project is its own uv project — run `uv sync` in its directory). |
| Swift | `client-samples/ios-visionos/` | Use the toolchain pinned by `// swift-tools-version:` in `Package.swift`; stick to Xcode's default formatting. |
| Kotlin | `client-samples/android/` | Use the Kotlin and Android Gradle Plugin versions pinned in `gradle/libs.versions.toml`; follow the Kotlin official style. |
| JavaScript | `client-samples/web/` | Plain ES modules, no build step; keep dependencies minimal. |

## Documentation and changelog discipline

A change is not done until the docs reflect it. Update `README.md` (and any
relevant sub-repository docs), `AGENTS.md`, and `DEPENDENCIES.md` **in the same
commit** as the code change. This applies to new packages, changed entry
points, new quickstart flows, renamed commands, and new configuration files.

Record significant new decisions in
[`docs/changelog.md`](https://github.com/NVIDIA/xr-ai/blob/main/docs/changelog.md)
(reverse chronological). Architectural rationale and historical context belong
in the changelog, not in source comments.

## Dependency discipline

[`DEPENDENCIES.md`](https://github.com/NVIDIA/xr-ai/blob/main/DEPENDENCIES.md)
at the repository root is the authoritative dependency map. **Any change to a
`pyproject.toml` must update `DEPENDENCIES.md` in the same commit** — a change
is not complete until `DEPENDENCIES.md` reflects it. Several hard rules follow
from this map (for example, `utils/xr-ai-launcher/` is stdlib-only and
`agent-sdk/xr-ai-agent` depends only on `pyzmq` + `msgpack`); refer to
`DEPENDENCIES.md` for the full set.

## SPDX license headers

This repository uses [REUSE](https://reuse.software/) and SPDX headers on every
source file we own. The hard rule lives in `AGENTS.md`: every new source file
gets the SPDX header at the top. This section documents the comment-style
choices and edge cases.

### The header

```
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

The Apache-2.0 license text lives in
[`LICENSE`](https://github.com/NVIDIA/xr-ai/blob/main/LICENSE). REUSE does not
auto-update the copyright year when you touch a file; include the current year
when adding or editing headers.

### Comment style by file type

Use the comment syntax for the file's language and place the header before any
other content, with one blank line separating it from the body:

| Style | Used for |
|---|---|
| `# …` | `.py`, `.yaml`/`.yml`, `.toml`, `.properties`, `.sh`, `.pro`, `.gitignore`, `.gitattributes`, `requirements.txt` |
| `// …` | `.swift`, `.kt`/`.kts`, `.js`, `.ts`/`.tsx` |
| `<!-- … -->` | `.xml`, `.html`, `.plist`, `.entitlements`, `.md` |

Insert the header **after** these required first-line directives when present:
`#!/...` shebangs, `<?xml …?>` declarations, `<!DOCTYPE …>`, and Swift's
`// swift-tools-version:` directive.

To add or fix headers, install the
[reuse tool](https://github.com/fsfe/reuse-tool) (`uv tool install reuse`) and
run, for example:

```bash
reuse annotate -t compact -l Apache-2.0 --skip-unrecognised -r path/to/file
```

### Files to skip

Skip files that can't carry comments or aren't ours to license: `LICENSE`,
`*.json`, `*.resolved`, binary assets (e.g. `*.gif`), `.gitkeep` markers,
Xcode-managed files (`*.pbxproj`, `*.xcworkspacedata`), and third-party Gradle
wrapper files (`gradlew`, `gradlew.bat`,
`gradle/wrapper/gradle-wrapper.properties`).

### Enforcement

Headers are enforced locally by
[`.github/scripts/check_spdx_headers.py`](https://github.com/NVIDIA/xr-ai/blob/main/.github/scripts/check_spdx_headers.py),
wired into
[`.pre-commit-config.yaml`](https://github.com/NVIDIA/xr-ai/blob/main/.pre-commit-config.yaml).
Run `pre-commit install` once after cloning to enable it;
`python3 .github/scripts/check_spdx_headers.py` audits the whole tree at any
time. The same check runs in CI as a backstop:
[`.github/workflows/spdx.yml`](https://github.com/NVIDIA/xr-ai/blob/main/.github/workflows/spdx.yml).

## Pull requests

Create a feature branch, keep code and docs in the same commit, ensure builds
and tests pass locally and in CI, and describe motivation, changes, and testing
in the PR. All commits must be signed off (`git commit -s`) per the DCO. Refer to
[`CONTRIBUTING.md`](https://github.com/NVIDIA/xr-ai/blob/main/CONTRIBUTING.md)
for the full PR process and the DCO text.
