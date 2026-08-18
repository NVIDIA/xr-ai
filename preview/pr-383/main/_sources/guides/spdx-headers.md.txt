<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SPDX license headers

The hard rule lives in `AGENTS.md`: every new source file gets the SPDX
header at the top. This file documents comment-style choices and edge cases.

## The header

```
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

## Comment style by file type

Use the comment syntax for the file's language and place the header before
any other content, with one blank line separating it from the body:

| Style | Used for |
|---|---|
| `# …` | `.py`, `.yaml`/`.yml`, `.toml`, `.properties`, `.sh`, `.pro`, `.gitignore`, `.gitattributes`, `requirements.txt` |
| `// …` | `.swift`, `.kt`/`.kts`, `.js`, `.ts`/`.tsx` |
| `<!-- … -->` | `.xml`, `.html`, `.plist`, `.entitlements`, `.md` |

Insert the header **after** these required first-line directives when present:
`#!/...` shebangs, `<?xml …?>` declarations, `<!DOCTYPE …>`, and Swift's
`// swift-tools-version:` directive.

## Files to skip

Skip files that can't carry comments or aren't ours to license: `LICENSE`,
`*.json`, `*.resolved`, binary assets (e.g. `*.gif`), `.gitkeep` markers,
Xcode-managed files (`*.pbxproj`, `*.xcworkspacedata`), and third-party Gradle
wrapper files (`gradlew`, `gradlew.bat`, `gradle/wrapper/gradle-wrapper.properties`).

## Enforcement

Enforced locally by `.github/scripts/check_spdx_headers.py`, wired into
`.pre-commit-config.yaml`. Run `pre-commit install` once after cloning to
enable it; `python3 .github/scripts/check_spdx_headers.py` audits the whole
tree at any time. The same check runs in CI as a backstop:
`.github/workflows/spdx.yml`.
