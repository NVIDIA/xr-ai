<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Software bill of materials

The `sbom` GitHub Actions workflow generates a software bill of materials (SBOM)
from the committed `dependency-manifest/` snapshots. It runs on pushes to
`main`, version tags, pull requests that change its inputs or tooling, and manual
dispatch. Download `xr-ai-sbom-<commit>` from the workflow run's artifacts:

- `xr-ai.spdx.json`: SPDX 2.3 JSON for dependency analysis tools.
- `xr-ai.cdx.json`: CycloneDX JSON for dependency analysis tools.
- `xr-ai.syft.json`: Syft's native inventory and discovery evidence.
- `provenance.json`: source revision, qualification timestamp, input and output
  SHA-256 digests, generator version, and package counts by ecosystem.

Artifacts are retained for 90 days. Download and retain them alongside release
materials when longer retention is required.

## Generate locally

Install [Syft 1.55.0](https://github.com/anchore/syft/releases/tag/v1.55.0)
and Python 3.11 or later. From the repository root, run:

```bash
python3 .github/scripts/generate_sbom.py
```

Use `--syft /path/to/syft` to select the executable and `--output-dir <directory>`
to change the default `dist/sbom/` destination. Generation reads lockfiles
without installing their packages or resolving newer versions. It includes web
development dependencies and both Android application and buildscript locks.
The generator checks every locked package name and version against the native
inventory and fails if a package is missing or unexpected.
CI also validates the SPDX document with `spdx-tools`. npm's local SDK download
reference is preserved as source information and represented as `NOASSERTION`
in SPDX's absolute download-location field.

## Coverage

This SBOM describes the **qualified dependency snapshot**, which changes during
qualification refreshes and targeted security updates. Ordinary project changes
can precede the next snapshot refresh. The Python lock includes mutually
exclusive environments and platform alternatives; their union is an inventory,
not a claim that all packages can be installed together. Refer to
[`DEPENDENCIES.md`](https://github.com/NVIDIA/xr-ai/blob/main/DEPENDENCIES.md)
for snapshot refresh rules and dependency boundaries.

The inventory covers Python runtime dependencies, web runtime and development
dependencies, and Android application and buildscript dependencies recorded in
those locks. License fields are limited to metadata in the inputs; generation
does not fetch license texts or establish license compliance. The SBOM does not
replace `THIRD_PARTY_NOTICES.md` or `third_party_licenses/`.

Installed environments, container operating-system packages, native libraries
bundled inside wheels or SDK archives, Apple client dependencies, and Python
build-system requirements outside the lock are outside this inventory's scope.
Use an SBOM generated from the final wheel, application, or container to describe
redistributed contents.
