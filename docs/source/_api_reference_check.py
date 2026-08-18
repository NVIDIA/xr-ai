# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for implementation details in the generated public API reference."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_API_REFERENCE = Path("reference/python")
_PRIVATE_MODULE_REFERENCE = re.compile(
    r"\bxr_ai_[a-z0-9_]+\._[a-z0-9_.]+",
    flags=re.IGNORECASE,
)
_PRIVATE_TERMS = {
    "pipecat": "Pipecat",
    "xrmediahubinputtransport": "XRMediaHubInputTransport",
    "xrmediahuboutputtransport": "XRMediaHubOutputTransport",
}


def validate_generated_api(build_dir: Path) -> list[str]:
    """Return private implementation details present in generated HTML."""

    reference = build_dir / _API_REFERENCE
    pages = sorted(reference.rglob("*.html")) if reference.is_dir() else []
    if not pages:
        return [f"generated API reference is missing: {reference}"]

    private_references: set[str] = set()
    private_terms: set[str] = set()
    for page in pages:
        html = page.read_text(encoding="utf-8")
        private_references.update(
            match.group(0) for match in _PRIVATE_MODULE_REFERENCE.finditer(html)
        )
        folded = html.casefold()
        private_terms.update(
            label for term, label in _PRIVATE_TERMS.items() if term in folded
        )
    return [
        *(f"generated API exposes private module path: {name}" for name in sorted(private_references)),
        *(
            f"generated API exposes private implementation detail: {label}"
            for label in sorted(private_terms)
        ),
    ]


def main() -> int:
    """Validate a Sphinx HTML build directory."""

    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} BUILD_DIR")
        return 2
    failures = validate_generated_api(Path(sys.argv[1]))
    if failures:
        print("Generated public API reference check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Generated public API reference check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
