# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and check the qualified dependency snapshot SBOM; run from any directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

SYFT_VERSION = "1.55.0"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "dependency-manifest"


def _expected_packages() -> dict[str, set[tuple[str, str]]]:
    lock = tomllib.loads((MANIFEST / "uv.lock").read_text())
    python = {(re.sub(r"[-_.]+", "-", p["name"]).lower(), p["version"]) for p in lock["package"]}
    lock = json.loads((MANIFEST / "web-xr-build/package-lock.json").read_text())
    npm = {(p.get("name", path.rsplit("node_modules/", 1)[-1]), p["version"]) for path, p in lock["packages"].items()}
    java = set()
    for path in sorted((MANIFEST / "android").glob("*.lockfile")):
        for line in path.read_text().splitlines():
            if line and not line.startswith(("#", "empty=")):
                group, name, version = line.split("=", 1)[0].split(":")
                java.add((f"{group}:{name}", version))
    return {"python": python, "npm": npm, "java-archive": java}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/sbom")
    parser.add_argument("--syft", default="syft", help=f"Path to Syft {SYFT_VERSION}")
    args = parser.parse_args()
    version = json.loads(subprocess.check_output([args.syft, "version", "-o", "json"], text=True))
    if version["version"].removeprefix("v") != SYFT_VERSION:
        raise SystemExit(f"Install Syft {SYFT_VERSION}; found {version['version']}")

    expected = _expected_packages()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    inputs = ["pyproject.toml", "uv.lock", "web-xr-build/package.json", "web-xr-build/package-lock.json"]
    inputs += [str(p.relative_to(MANIFEST)) for p in sorted((MANIFEST / "android").glob("*.lockfile"))]
    with tempfile.TemporaryDirectory(prefix="xr-ai-sbom-") as temporary:
        staged = Path(temporary)
        for name in inputs:
            target = staged / name
            # Syft discovers Gradle locks by basename. The buildscript lock is
            # stored under a repository-specific name, so give it its own root.
            if name == "android/buildscript-gradle.lockfile":
                target = staged / "android/buildscript/gradle.lockfile"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MANIFEST / name, target)

        config = staged / "syft.yaml"
        config.write_text("javascript:\n  include-dev-dependencies: true\n")
        subprocess.run(
            [
                args.syft,
                "scan",
                f"dir:{staged}",
                "--config",
                str(config),
                "--source-name",
                "xr-ai-qualified-dependencies",
                "--source-version",
                revision,
                "-o",
                f"syft-json={staged / 'xr-ai.syft.json'}",
                "-o",
                f"spdx-json={staged / 'xr-ai.spdx.json'}",
                "-o",
                f"cyclonedx-json={staged / 'xr-ai.cdx.json'}",
            ],
            check=True,
            env={**os.environ, "SYFT_CHECK_FOR_APP_UPDATE": "false"},
        )
        document = json.loads((staged / "xr-ai.syft.json").read_text())
        for ecosystem, packages in expected.items():
            found = set()
            for package in document["artifacts"]:
                if package["type"] != ecosystem:
                    continue
                name = package["name"]
                if ecosystem == "java-archive":
                    name = f"{package['metadata']['pomProject']['groupId']}:{name}"
                found.add((name, package["version"]))
            if not packages or packages != found:
                raise SystemExit(
                    f"Incomplete {ecosystem} inventory: missing={sorted(packages - found)}, "
                    f"unexpected={sorted(found - packages)}"
                )

        spdx_path = staged / "xr-ai.spdx.json"
        spdx = json.loads(spdx_path.read_text())
        for package in spdx["packages"]:
            location = package.get("downloadLocation", "")
            # SPDX requires an absolute download location. npm's relative file
            # reference identifies a local input, not a retrievable URL.
            if location.startswith("file:"):
                package["downloadLocation"] = "NOASSERTION"
                package["sourceInfo"] = (
                    package.get("sourceInfo", "") + f"; lockfile local download reference: {location}"
                )
        spdx_path.write_text(json.dumps(spdx, indent=2) + "\n")

        for name in ("xr-ai.syft.json", "xr-ai.spdx.json", "xr-ai.cdx.json"):
            shutil.copyfile(staged / name, output / name)

    evidence = {
        "scope": "Committed qualified dependency snapshots; not installed or redistributed contents",
        "revision": revision,
        "syft_version": SYFT_VERSION,
        "qualification": tomllib.loads((ROOT / "uv.toml").read_text())["exclude-newer"],
        "packages": {kind: len(packages) for kind, packages in expected.items()},
        "inputs": {
            f"dependency-manifest/{name}": hashlib.sha256((MANIFEST / name).read_bytes()).hexdigest() for name in inputs
        },
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("xr-ai.*.json"))},
    }
    (output / "provenance.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    print(f"Validated dependency SBOMs: {evidence['packages']} -> {output}")


if __name__ == "__main__":
    main()
