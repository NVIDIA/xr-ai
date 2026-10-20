# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency boundaries for shared utility ownership."""
import ast
import sys
import tomllib
from pathlib import Path

from test_dependency_map import dependency_map

_ROOT = Path(__file__).resolve().parents[1]


def test_shared_helper_imports_only_standard_library():
    directory = _ROOT / "utils" / "xr-ai-gpu" / "xr_ai_gpu"
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module.split(".")[0]]
            else:
                continue
            assert set(names) <= sys.stdlib_module_names, (path, names)
    metadata = tomllib.loads((directory.parent / "pyproject.toml").read_text())
    assert metadata["project"]["dependencies"] == []


def test_repository_local_dependency_graph_is_acyclic():
    projects = dependency_map.discover_projects(_ROOT)
    graph = {project.name: {
        dependency_map.dependency_name(dep)
        for dep in (*project.dependencies, *(
            dep for values in project.optional_dependencies.values() for dep in values
        ))
    } for project in projects}
    visited = set()

    def visit(name, ancestors):
        assert name not in ancestors, f"dependency cycle: {ancestors + (name,)}"
        if name in visited or name not in graph:
            return
        for dependency in graph[name]:
            visit(dependency, ancestors + (name,))
        visited.add(name)

    for name in graph:
        visit(name, ())
