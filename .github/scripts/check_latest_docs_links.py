# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that repository entry-point links resolve in rendered ``/latest/`` docs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

_LATEST_URL = re.compile(
    r"https://nvidia\.github\.io/xr-ai/latest/[^\s<>'\"\])}]+"
)


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del tag
        for name, value in attrs:
            if name == "id" and value is not None:
                self.ids.add(value)

    handle_startendtag = handle_starttag


def _tracked_entry_points(repository_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "*README.md", "CONTRIBUTING.md"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(repository_root / line for line in result.stdout.splitlines())


def check_latest_docs_links(
    repository_root: Path,
    rendered_docs: Path,
    entry_points: tuple[Path, ...] | None = None,
) -> tuple[str, ...]:
    """Return errors for missing pages or fragments linked by entry points."""

    repository_root = repository_root.resolve()
    rendered_docs = rendered_docs.resolve()
    paths = (
        entry_points
        if entry_points is not None
        else _tracked_entry_points(repository_root)
    )
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    checked: set[tuple[Path, str]] = set()

    for readme in paths:
        source = readme.read_text(encoding="utf-8")
        for match in _LATEST_URL.finditer(source):
            url = match.group().rstrip(".,;:!?")
            parsed = urlsplit(url)
            relative_url = unquote(parsed.path.removeprefix("/xr-ai/latest/"))
            relative_path = PurePosixPath(relative_url)
            if (
                not relative_url
                or relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                errors.append(f"{readme}: invalid latest documentation URL {url}")
                continue

            page = rendered_docs.joinpath(*relative_path.parts)
            fragment = unquote(parsed.fragment)
            key = (page, fragment)
            if key in checked:
                continue
            checked.add(key)

            if not page.is_file():
                errors.append(f"{readme}: rendered page does not exist for {url}")
                continue
            if not fragment:
                continue

            if page not in anchor_cache:
                parser = _AnchorParser()
                parser.feed(page.read_text(encoding="utf-8"))
                anchor_cache[page] = parser.ids
            if fragment not in anchor_cache[page]:
                errors.append(f"{readme}: rendered fragment does not exist for {url}")

    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rendered_docs", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    errors = check_latest_docs_links(args.repository_root, args.rendered_docs)
    if errors and not args.quiet:
        print("\n".join(errors), file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
