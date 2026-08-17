# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static public-API documentation checks shared by Sphinx and CI."""

from __future__ import annotations

import ast
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
API_PACKAGE_DIRS = (
    _REPOSITORY_ROOT / "agent-sdk" / "xr-ai-runtime" / "xr_ai_runtime",
    _REPOSITORY_ROOT / "agent-sdk" / "xr-ai-voice" / "xr_ai_voice",
)


def _literal_exports(tree: ast.Module, path: Path) -> tuple[str, ...]:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: __all__ must be a literal sequence") from error
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(name, str) for name in value
        ):
            raise ValueError(f"{path}: __all__ must contain only strings")
        return tuple(value)
    raise ValueError(f"{path}: public package must define __all__")


def _assignment_has_docstring(tree: ast.Module, node: ast.AST) -> bool:
    try:
        index = tree.body.index(node)
    except ValueError:
        return False
    if index + 1 >= len(tree.body):
        return False
    following = tree.body[index + 1]
    return (
        isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
        and bool(following.value.value.strip())
    )


def _documented_definitions(
    package_dir: Path,
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    definitions: dict[str, bool] = {}
    missing_methods: dict[str, list[str]] = {}
    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[node.name] = bool(ast.get_docstring(node))
                if isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        if member.name.startswith("_") or ast.get_docstring(member):
                            continue
                        missing_methods.setdefault(node.name, []).append(member.name)
                continue
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = _assignment_has_docstring(tree, node)
    return definitions, missing_methods


def validate_public_api() -> list[str]:
    """Return documentation contract violations for the enrolled packages."""

    failures: list[str] = []
    for package_dir in API_PACKAGE_DIRS:
        facade = package_dir / "__init__.py"
        tree = ast.parse(facade.read_text(encoding="utf-8"), filename=str(facade))
        exports = _literal_exports(tree, facade)
        definitions, missing_methods = _documented_definitions(package_dir)
        for name in exports:
            if name not in definitions:
                failures.append(f"{package_dir.name}: exported {name} does not resolve")
            elif not definitions[name]:
                failures.append(f"{package_dir.name}: exported {name} has no docstring")
            failures.extend(
                f"{package_dir.name}: public method {name}.{method} has no docstring"
                for method in missing_methods.get(name, ())
            )
    return failures


def main() -> int:
    """Print public API documentation failures and return a process status."""

    failures = validate_public_api()
    if failures:
        print("Public API documentation check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Public API documentation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
