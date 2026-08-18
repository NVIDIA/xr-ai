# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for implementation details in the generated public API reference."""

from __future__ import annotations

import sys
from pathlib import Path

_VOICE_REFERENCE = Path("reference/python/xr_ai_voice/index.html")
_PRIVATE_VOICE_TERMS = {
    "pipecat": "Pipecat",
    "xrmediahubinputtransport": "XRMediaHubInputTransport",
    "xrmediahuboutputtransport": "XRMediaHubOutputTransport",
}


def validate_generated_api(build_dir: Path) -> list[str]:
    """Return private implementation details present in generated HTML."""

    reference = build_dir / _VOICE_REFERENCE
    if not reference.is_file():
        return [f"generated voice API page is missing: {reference}"]

    html = reference.read_text(encoding="utf-8").casefold()
    return [
        f"generated voice API exposes private implementation detail: {label}"
        for term, label in _PRIVATE_VOICE_TERMS.items()
        if term in html
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
