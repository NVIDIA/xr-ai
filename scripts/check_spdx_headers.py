# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Enforce the SPDX header convention from ``AGENTS.md`` § License headers.

Every source file we license must start with::

    SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

wrapped in the comment syntax for that file's language. Required first-line
directives (``#!`` shebang, ``<?xml ?>``, ``<!DOCTYPE>``, Swift's
``// swift-tools-version:``) are skipped before the header is inspected.

Usage
-----
    python3 scripts/check_spdx_headers.py [paths...]

With no paths, walks the repo. Designed for ``pre-commit`` with
``pass_filenames: true``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ── Header pattern ──────────────────────────────────────────────────────────
_COPYRIGHT_RE = re.compile(
    r"SPDX-FileCopyrightText:\s+Copyright\s+\(c\)\s+\d{4}\s+"
    r"NVIDIA CORPORATION & AFFILIATES\.\s+All rights reserved\."
)
_LICENSE_LINE = "SPDX-License-Identifier: Apache-2.0"

# ── Comment-style mapping (mirrors AGENTS.md § License headers) ─────────────
_HASH_EXTS = {".py", ".yaml", ".yml", ".toml", ".properties", ".sh", ".pro"}
_HASH_NAMES = {".gitignore", ".gitattributes", "requirements.txt"}
_SLASH_EXTS = {".swift", ".kt", ".kts", ".js", ".ts", ".tsx"}
_HTML_EXTS = {".xml", ".html", ".plist", ".entitlements", ".md"}

# ── Files to skip (not ours to license, can't carry comments, or third party) ──
_SKIP_NAMES = {
    "LICENSE",
    "gradlew",
    "gradlew.bat",
    ".gitkeep",
}
_SKIP_EXTS = {
    ".json",
    ".resolved",
    ".gif",
    ".pbxproj",
    ".xcworkspacedata",
}
_SKIP_PATH_SUFFIXES = (
    "gradle/wrapper/gradle-wrapper.properties",
)

# ── Walk pruning ────────────────────────────────────────────────────────────
_PRUNE_DIRS = {".git", ".venv", "node_modules", "models", "__pycache__", ".cache"}

_REPO_ROOT = Path(__file__).resolve().parents[1]


def comment_style(path: Path) -> str | None:
    """Return ``"hash"`` / ``"slash"`` / ``"html"`` for ``path``, or ``None`` to skip."""
    name = path.name
    suffix = path.suffix
    s = str(path).replace("\\", "/")

    if name in _SKIP_NAMES or suffix in _SKIP_EXTS:
        return None
    if any(s.endswith(suf) for suf in _SKIP_PATH_SUFFIXES):
        return None

    if suffix in _HASH_EXTS or name in _HASH_NAMES:
        return "hash"
    if suffix in _SLASH_EXTS:
        return "slash"
    if suffix in _HTML_EXTS:
        return "html"
    return None


def _strip_directives(lines: list[str], style: str) -> list[str]:
    """Drop required first-line directives so the header check starts after them."""
    if not lines:
        return lines
    first = lines[0]
    if first.startswith("#!"):
        return lines[1:]
    stripped = first.lstrip()
    if stripped.startswith("<?xml") or stripped.startswith("<!DOCTYPE"):
        return lines[1:]
    if style == "slash" and stripped.startswith("// swift-tools-version"):
        return lines[1:]
    return lines


def _read_head(path: Path, max_bytes: int = 4096) -> str | None:
    try:
        with path.open("rb") as f:
            raw = f.read(max_bytes)
    except OSError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def check(path: Path) -> tuple[bool, str]:
    """Validate ``path``'s SPDX header. Returns ``(ok, reason)``."""
    style = comment_style(path)
    if style is None:
        return True, "skipped"

    text = _read_head(path)
    if text is None:
        return False, "could not read as UTF-8"
    if not text.strip():
        return False, "empty file (header required)"

    lines = _strip_directives(text.splitlines(), style)

    # Examine a small window after any directives.
    window_lines = lines[:10]
    window = "\n".join(window_lines)

    copyright_match = _COPYRIGHT_RE.search(window)
    if not copyright_match:
        return False, (
            "missing 'SPDX-FileCopyrightText: Copyright (c) <year> "
            "NVIDIA CORPORATION & AFFILIATES. All rights reserved.' near top of file"
        )
    if _LICENSE_LINE not in window:
        return False, f"missing '{_LICENSE_LINE}' near top of file"

    # Verify the SPDX line uses the right comment marker for this file type.
    spdx_line = next((ln for ln in window_lines if "SPDX-FileCopyrightText" in ln), "")
    lstripped = spdx_line.lstrip()
    if style == "hash":
        if not lstripped.startswith("#"):
            return False, f"SPDX line must start with '#' (got: {spdx_line!r})"
    elif style == "slash":
        if not lstripped.startswith("//"):
            return False, f"SPDX line must start with '//' (got: {spdx_line!r})"
    elif style == "html":
        # The header must sit inside an HTML comment block somewhere in the window.
        if "<!--" not in window or "-->" not in window:
            return False, "SPDX header must be wrapped in '<!-- ... -->'"

    return True, "ok"


def discover(root: Path) -> list[Path]:
    found: list[Path] = []

    def _walk(d: Path) -> None:
        for entry in sorted(d.iterdir()):
            if entry.is_dir():
                if entry.name in _PRUNE_DIRS or entry.name.startswith("."):
                    continue
                _walk(entry)
            elif entry.is_file():
                if comment_style(entry) is not None:
                    found.append(entry)

    _walk(root)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="files to check (default: walk repo)")
    args = ap.parse_args(argv)

    if args.paths:
        paths = [p for p in args.paths if p.is_file()]
    else:
        paths = discover(_REPO_ROOT)

    failures: list[tuple[Path, str]] = []
    checked = 0
    for p in paths:
        if comment_style(p) is None:
            continue
        checked += 1
        ok, reason = check(p)
        if not ok:
            failures.append((p, reason))

    if failures:
        print("SPDX header check failed:", file=sys.stderr)
        for p, reason in failures:
            try:
                rel = p.relative_to(_REPO_ROOT)
            except ValueError:
                rel = p
            print(f"  {rel}: {reason}", file=sys.stderr)
        print(
            "\nSee AGENTS.md § License headers for the exact text and "
            "comment-syntax mapping.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {checked} file(s) carry a valid SPDX header")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
