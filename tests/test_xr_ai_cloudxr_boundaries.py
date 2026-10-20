# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency boundaries for shared utility ownership."""
import ast
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_shared_helper_imports_only_standard_library():
    directory = _ROOT / "utils" / "xr-ai-cloudxr" / "xr_ai_cloudxr"
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


def test_cloudxr_consumers_do_not_import_helpers_from_launcher():
    consumers = [
        "services/cloudxr-runtime/cloudxr_runtime/__main__.py",
        "services/openxr-service/openxr_service/__main__.py",
        "agent-samples/xr-render-demo/scene/xr_render_scene/engine.py",
        "agent-samples/xr-render-demo/main.py",
    ]
    helpers = {"is_native_profile", "read_device_profile", "load_cloudxr_env"}
    for relative in consumers:
        tree = ast.parse((_ROOT / relative).read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert any(node.module == "xr_ai_cloudxr" for node in imports), relative
        for node in imports:
            if node.module == "xr_ai_launcher":
                assert helpers.isdisjoint(alias.name for alias in node.names), relative
    for service in ("cloudxr-runtime", "openxr-service"):
        metadata = tomllib.loads((_ROOT / "services" / service / "pyproject.toml").read_text())
        assert "xr-ai-launcher" not in metadata["project"]["dependencies"]
        assert "xr-ai-cloudxr" in metadata["project"]["dependencies"]
        if service == "cloudxr-runtime":
            assert "xr-ai-gpu" in metadata["project"]["dependencies"]


def test_launcher_depends_only_on_shared_helpers():
    metadata = tomllib.loads((_ROOT / "utils/xr-ai-launcher/pyproject.toml").read_text())
    assert set(metadata["project"]["dependencies"]) == {"xr-ai-gpu", "xr-ai-cloudxr"}
