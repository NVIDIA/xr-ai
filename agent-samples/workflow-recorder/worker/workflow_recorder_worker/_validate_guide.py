# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line validation for generated SOP guide files."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._workflow_spec import load_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate workflow-recorder SOP YAML")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        workflow = load_workflow(path)
        print(f"valid: {path} ({workflow.id} v{workflow.version}, {workflow.status}, {len(workflow.steps)} steps)")


if __name__ == "__main__":
    main()
