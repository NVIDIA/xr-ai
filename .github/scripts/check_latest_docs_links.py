# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify repository entry-point links against a rendered documentation build."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


def iter_doc_urls(source: str, ref: str = "latest") -> Iterator[str]:
    """Yield published documentation URLs for the selected ref."""
    prefix = re.escape(f"https://nvidia.github.io/xr-ai/{ref}/")
    for match in re.finditer(prefix + r"[^\s<>'\"`\])}]+", source):
        yield match.group().rstrip(".,;:!?")


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
        [
            "git", "ls-files", "-z", "*README.md", "CONTRIBUTING.md", "CLAUDE.md",
            "AGENTS.md", ":(glob)**/AGENTS.md", ":(glob)skills/**/SKILL.md",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(repository_root / path for path in result.stdout.split("\0") if path)


def check_latest_docs_links(
    repository_root: Path,
    rendered_docs: Path,
    entry_points: tuple[Path, ...] | None = None,
    *,
    ref: str = "latest",
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

    for entry_point in paths:
        source = entry_point.read_text(encoding="utf-8")
        for url in iter_doc_urls(source, ref):
            parsed = urlsplit(url)
            relative_url = unquote(parsed.path.removeprefix(f"/xr-ai/{ref}/"))
            relative_path = PurePosixPath(relative_url)
            if (
                not relative_url
                or relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                errors.append(f"{entry_point}: invalid {ref} documentation URL {url}")
                continue

            page = rendered_docs.joinpath(*relative_path.parts)
            fragment = unquote(parsed.fragment)
            key = (page, fragment)
            if key in checked:
                continue
            checked.add(key)

            if not page.is_file():
                errors.append(f"{entry_point}: rendered page does not exist for {url}")
                continue
            if not fragment:
                continue

            if page not in anchor_cache:
                parser = _AnchorParser()
                parser.feed(page.read_text(encoding="utf-8"))
                anchor_cache[page] = parser.ids
            if fragment not in anchor_cache[page]:
                errors.append(f"{entry_point}: rendered fragment does not exist for {url}")

    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rendered_docs", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--ref", default="latest", help="Documentation URL ref to check (default: latest)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    errors = check_latest_docs_links(args.repository_root, args.rendered_docs, ref=args.ref)
    if errors and not args.quiet:
        print("\n".join(errors), file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
