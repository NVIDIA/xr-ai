# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m xr_latency.report RUN_DIR... --out DIR``: the same as ``xr-latency report``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build import make_report


def main() -> None:
    """Fit ``model.json`` from the given run directories and render the report next to it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--findings", type=Path, help="HTML fragment with the findings section")
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()
    print(make_report(args.runs, args.out, args.findings, args.subtitle))


if __name__ == "__main__":
    main()
