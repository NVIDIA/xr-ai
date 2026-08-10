# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the documentation release tag according to repository policy."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

_SEMVER = re.compile(
    r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


@dataclass(frozen=True)
class _Version:
    tag: str
    major: int
    minor: int
    patch: int
    prerelease: tuple[tuple[int, int | str], ...] | None

    @property
    def precedence(self) -> tuple[object, ...]:
        return (
            self.major,
            self.minor,
            self.patch,
            self.prerelease is None,
            self.prerelease or (),
            self.tag,
        )


def _parse(tag: str) -> _Version | None:
    match = _SEMVER.fullmatch(tag)
    if not match:
        return None
    prerelease_text = match.group(4)
    prerelease: list[tuple[int, int | str]] = []
    if prerelease_text:
        for identifier in prerelease_text.split("."):
            if identifier.isdigit():
                if len(identifier) > 1 and identifier.startswith("0"):
                    return None
                prerelease.append((0, int(identifier)))
            else:
                prerelease.append((1, identifier))
    return _Version(
        tag=tag,
        major=int(match.group(1)),
        minor=int(match.group(2)),
        patch=int(match.group(3)),
        prerelease=tuple(prerelease) if prerelease_text else None,
    )


def select_latest(tags: list[str]) -> str | None:
    versions = [version for tag in tags if (version := _parse(tag))]
    stable = [version for version in versions if version.prerelease is None]
    candidates = stable or versions
    if not candidates:
        return None
    return max(candidates, key=lambda version: version.precedence).tag


def main() -> None:
    if latest := select_latest([line.strip() for line in sys.stdin if line.strip()]):
        print(latest)


if __name__ == "__main__":
    main()
