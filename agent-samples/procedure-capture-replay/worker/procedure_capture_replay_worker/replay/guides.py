# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select and pin one locally reviewed guide for a replay worker."""

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


def select_guide(directory: Path, name: str) -> SelectedGuide:
    """Match an exact case-insensitive task name or ID; never approve implicitly."""
    selector = name.strip().casefold()
    if not selector:
        raise ValueError("guide name must not be empty")
    directory = directory.resolve()
    matches: list[SelectedGuide] = []
    invalid: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.name.endswith((".guide.yaml", ".guide.yml")):
            continue
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError("symbolic-link guides are not allowed")
            with path.open("rb") as stream:
                content = stream.read(1_000_001)
            if len(content) > 1_000_000:
                raise ValueError("guide exceeds 1000000 bytes")
            workflow = parse_workflow(content)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            invalid.append(f"{path.name}: {exc}")
            continue
        if selector in {workflow.name.casefold(), workflow.id.casefold()}:
            matches.append(SelectedGuide(path, hashlib.sha256(content).hexdigest(), workflow))
    if not matches:
        detail = f" Invalid guides: {'; '.join(invalid)}" if invalid else ""
        raise ValueError(f"No guide named {name!r} in {directory}.{detail}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous guide name {name!r}: {[str(item.path) for item in matches]}")
    guide = matches[0]
    if not guide.workflow.runnable:
        raise ValueError(f"Guide {name!r} is a draft; review it and set task.status to approved before replay.")
    return guide
