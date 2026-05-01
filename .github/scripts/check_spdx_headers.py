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
    python3 .github/scripts/check_spdx_headers.py [paths...]            # check
    python3 .github/scripts/check_spdx_headers.py --fix [paths...]      # insert missing headers

With no paths, walks the repo. Designed for ``pre-commit`` with
``pass_filenames: true``.

In ``--fix`` mode, missing headers are inserted in the right comment
style for the file's language, after any required first-line directive
(shebang / xml decl / DOCTYPE / Swift tools-version). If the script
modifies any file, pre-commit will detect the change, abort the commit,
and the operator re-stages + re-commits with the fix applied.
"""
from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

# ── Header pattern ──────────────────────────────────────────────────────────
# Accept a single year or a `first-current` range (e.g. ``2024-2026``).
_COPYRIGHT_RE = re.compile(
    r"SPDX-FileCopyrightText:\s+Copyright\s+\(c\)\s+\d{4}(?:-\d{4})?\s+"
    r"NVIDIA CORPORATION & AFFILIATES\.\s+All rights reserved\."
)
_LICENSE_LINE = "SPDX-License-Identifier: Apache-2.0"


def _first_commit_year(path: Path) -> int | None:
    """
    Return the year of the commit that introduced ``path`` to the repo,
    or ``None`` for untracked / brand-new files.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow",
             "--format=%aI", "--", str(path)],
            capture_output=True, text=True,
            cwd=_REPO_ROOT, check=False, timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # The last line is the earliest commit (git log is reverse-chronological).
    earliest = result.stdout.strip().splitlines()[-1]
    if len(earliest) >= 4 and earliest[:4].isdigit():
        return int(earliest[:4])
    return None


def _copyright_line(path: Path | None = None) -> str:
    """
    Build the SPDX copyright line for ``--fix`` insertion.

    * Untracked / brand-new file → current year only.
    * File whose first commit was earlier → ``firstYear-currentYear`` range.
    * File whose first commit was the current year → current year only.

    Existing valid headers (any 4-digit year or range) still pass the check
    via :data:`_COPYRIGHT_RE`, so this only governs *insertion*; we never
    rewrite a year that's already there.
    """
    current = datetime.date.today().year
    first = _first_commit_year(path) if path is not None else None
    years = f"{first}-{current}" if first and first < current else str(current)
    return (
        f"SPDX-FileCopyrightText: Copyright (c) {years} "
        f"NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    )

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
# Explicit set rather than "any dotted directory" — we want to scan `.github/`
# (workflows + this very script live there).
_PRUNE_DIRS = {
    ".git", ".venv", ".cache",
    ".idea", ".vscode",
    ".mypy_cache", ".pytest_cache",
    ".tox", ".nox",
    "node_modules", "models", "__pycache__",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _build_header(path: Path, style: str) -> str:
    """Return the SPDX header text (with trailing blank line) for ``path``."""
    cline = _copyright_line(path)
    if style == "hash":
        return f"# {cline}\n# {_LICENSE_LINE}\n\n"
    if style == "slash":
        return f"// {cline}\n// {_LICENSE_LINE}\n\n"
    if style == "html":
        return (
            "<!--\n"
            f"  {cline}\n"
            f"  {_LICENSE_LINE}\n"
            "-->\n\n"
        )
    raise ValueError(f"unknown comment style: {style}")


def _directive_index(lines: list[str], style: str) -> int:
    """Return the index after any required first-line directive (0 if none)."""
    if not lines:
        return 0
    first = lines[0]
    if first.startswith("#!"):
        return 1
    stripped = first.lstrip()
    if stripped.startswith("<?xml") or stripped.startswith("<!DOCTYPE"):
        return 1
    if style == "slash" and stripped.startswith("// swift-tools-version"):
        return 1
    return 0


def insert_header(path: Path) -> bool:
    """
    Insert the SPDX header into ``path`` (after any directive line).

    Returns True on success, False if the file's style is unknown or it
    cannot be read as UTF-8.
    """
    style = comment_style(path)
    if style is None:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    lines = text.splitlines(keepends=True)
    insert_at = _directive_index([ln.rstrip("\r\n") for ln in lines], style)
    header = _build_header(path, style)

    new_text = "".join(lines[:insert_at]) + header + "".join(lines[insert_at:])
    if not new_text.endswith("\n"):
        new_text += "\n"

    path.write_text(new_text, encoding="utf-8")
    return True


def discover(root: Path) -> list[Path]:
    found: list[Path] = []

    def _walk(d: Path) -> None:
        for entry in sorted(d.iterdir()):
            if entry.is_dir():
                if entry.name in _PRUNE_DIRS:
                    continue
                # Skip virtualenvs regardless of directory name — the
                # `.venv` convention is common but not universal, so look
                # for the marker file the venv module always writes.
                if (entry / "pyvenv.cfg").is_file():
                    continue
                _walk(entry)
            elif entry.is_file():
                if comment_style(entry) is not None:
                    found.append(entry)

    _walk(root)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fix",
        action="store_true",
        help="insert missing headers in place (in the right comment style)",
    )
    ap.add_argument("paths", nargs="*", type=Path, help="files to check (default: walk repo)")
    args = ap.parse_args(argv)

    if args.paths:
        paths = [p for p in args.paths if p.is_file()]
    else:
        paths = discover(_REPO_ROOT)

    failures: list[tuple[Path, str]] = []
    fixed: list[Path] = []
    checked = 0
    for p in paths:
        if comment_style(p) is None:
            continue
        checked += 1
        ok, reason = check(p)
        if ok:
            continue
        if args.fix:
            if insert_header(p):
                fixed.append(p)
                continue
        failures.append((p, reason))

    if fixed:
        print(f"inserted SPDX header into {len(fixed)} file(s):", file=sys.stderr)
        for p in fixed:
            try:
                rel = p.relative_to(_REPO_ROOT)
            except ValueError:
                rel = p
            print(f"  {rel}", file=sys.stderr)

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

    if fixed:
        # Pre-commit re-stages on its own when files change; signaling failure
        # here makes the commit abort so the operator reviews + recommits.
        print(
            "\nSPDX headers were inserted. Review the diff, re-stage the files, "
            "and re-commit.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {checked} file(s) carry a valid SPDX header")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
