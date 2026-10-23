# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select an approved local procedure guide for the guide worker.

Select a file directly or from a directory containing one YAML file. Validate it
through _workflow_spec.py and require approved status. Retain its content hash so
editing the source file does not change a running worker's instructions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from ._workflow_spec import Workflow, parse_workflow


@dataclass(frozen=True, slots=True)
class SelectedGuide:
    path: Path
    sha256: str
    workflow: Workflow


def load_guide(path: Path) -> SelectedGuide:
    """Resolve an unambiguous guide; pin validation and hashing to the same bytes."""
    try:
        if path.is_dir():
            candidates = sorted(
                child for child in path.iterdir()
                if child.suffix.lower() in {".yaml", ".yml"} and child.is_file()
            )
            if not candidates:
                raise ValueError(
                    "no YAML files at this directory level; provide the path to a specific guide YAML file"
                )
            if len(candidates) > 1:
                listing = "\n".join(f"  {candidate.name}" for candidate in candidates)
                raise ValueError(
                    f"multiple YAML files at this directory level:\n{listing}\n"
                    "Provide the path to a specific guide YAML file."
                )
            path = candidates[0]
        path = path.resolve()
        with path.open("rb") as stream:
            content = stream.read(1_000_001)
        if len(content) > 1_000_000:
            raise ValueError("guide exceeds 1000000 bytes")
        workflow = parse_workflow(content)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load guide {path}: {exc}") from exc
    guide = SelectedGuide(path, hashlib.sha256(content).hexdigest(), workflow)
    if not guide.workflow.runnable:
        raise ValueError(
            f"Guide {path} is a draft; review it and set task.status to approved before using guide mode."
        )
    return guide
