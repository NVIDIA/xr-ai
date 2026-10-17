# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover version tags without a GitHub API dependency or a maintained release map."""

import json
import re
import subprocess
import sys


def version(tag):
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)(?:-(alpha|beta|rc)(?:[.-]?(\d+))?)?", tag)
    if not match:
        return None
    major, minor, patch, stage, number = match.groups()
    return (int(major), int(minor), int(patch), {"alpha": 0, "beta": 1, "rc": 2, None: 3}[stage], int(number or 0))


def releases():
    output = subprocess.check_output(
        ["git", "ls-remote", "--tags", "--refs", "https://github.com/NVIDIA/xr-ai.git", "v*"],
        text=True, stderr=subprocess.DEVNULL, timeout=10,
    )
    # Earlier layouts are not covered by this deployment's release adapter.
    minimum = version("v0.2.0-beta.2")
    tags = {line.split("refs/tags/", 1)[1] for line in output.splitlines() if "refs/tags/" in line}
    tags = sorted((tag for tag in tags if version(tag) and version(tag) >= minimum), key=version, reverse=True)
    if not tags:
        raise ValueError("no supported XR AI release tags found")
    return {"tags": tags, "latest": tags[0]}


if __name__ == "__main__":
    try:
        result = releases()
        print(result["latest"] if sys.argv[1:] == ["--latest"] else json.dumps(result))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SystemExit(f"could not discover XR AI releases: {exc}; select current or an explicit tag or SHA") from exc
