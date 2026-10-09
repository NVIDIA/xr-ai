# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only schema validation of model-authored SOP guides; never generates YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ._workflow_spec import load_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SOP guide YAML without modifying or approving it")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for path in args.paths:
        try:
            workflow = load_workflow(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f"invalid: {path}: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"valid: {path} ({workflow.id} v{workflow.version}, {workflow.status}, {len(workflow.steps)} steps)")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
