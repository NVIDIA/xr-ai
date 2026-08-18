# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static public-API documentation checks shared by Sphinx and CI."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
API_PACKAGE_DIRS = (
    _REPOSITORY_ROOT / "agent-sdk" / "xr-ai-runtime" / "xr_ai_runtime",
    _REPOSITORY_ROOT / "agent-sdk" / "xr-ai-voice" / "xr_ai_voice",
)


@dataclass(frozen=True)
class _Module:
    path: Path
    parts: tuple[str, ...]
    is_package: bool
    tree: ast.Module


@dataclass(frozen=True)
class _Definition:
    module: _Module
    node: ast.AST


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
        if not isinstance(value, (list, tuple)) or not all(isinstance(name, str) for name in value):
            raise ValueError(f"{path}: __all__ must contain only strings")
        return tuple(value)
    raise ValueError(f"{path}: public package must define __all__")


def _following_docstring(body: list[ast.stmt], node: ast.AST) -> bool:
    try:
        index = body.index(node)
    except ValueError:
        return False
    if index + 1 >= len(body):
        return False
    following = body[index + 1]
    return (
        isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
        and bool(following.value.value.strip())
    )


def _load_module(
    package_dir: Path,
    parts: tuple[str, ...],
    modules: dict[tuple[str, ...], _Module],
) -> _Module | None:
    if parts in modules:
        return modules[parts]

    candidate = package_dir.joinpath(*parts)
    if candidate.is_dir():
        path = candidate / "__init__.py"
        is_package = True
    else:
        path = candidate.with_suffix(".py")
        is_package = False
    if not path.is_file():
        return None

    module = _Module(
        path=path,
        parts=parts,
        is_package=is_package,
        tree=ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
    )
    modules[parts] = module
    return module


def _imported_module_parts(
    package_dir: Path,
    module: _Module,
    node: ast.ImportFrom,
) -> tuple[str, ...] | None:
    imported = tuple(node.module.split(".")) if node.module else ()
    if node.level == 0:
        if not imported or imported[0] != package_dir.name:
            return None
        return imported[1:]

    package = module.parts if module.is_package else module.parts[:-1]
    parent_count = node.level - 1
    if parent_count > len(package):
        return None
    return (*package[: len(package) - parent_count], *imported)


def _assignment_value(node: ast.Assign | ast.AnnAssign) -> ast.expr | None:
    return node.value


def _resolve_name(
    package_dir: Path,
    module: _Module,
    name: str,
    modules: dict[tuple[str, ...], _Module],
    resolving: frozenset[tuple[tuple[str, ...], str]] = frozenset(),
) -> _Definition | None:
    key = (module.parts, name)
    if key in resolving:
        return None
    resolving |= {key}

    binding: ast.AST | tuple[tuple[str, ...], str] | None = None
    for node in module.tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                binding = node
            continue
        if isinstance(node, ast.ImportFrom):
            imported_parts = _imported_module_parts(package_dir, module, node)
            for alias in node.names:
                if alias.name == "*" or (alias.asname or alias.name) != name:
                    continue
                binding = (imported_parts, alias.name) if imported_parts is not None else None
            continue
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            binding = node

    if isinstance(binding, tuple):
        imported_module = _load_module(package_dir, binding[0], modules)
        if imported_module is None:
            return None
        return _resolve_name(
            package_dir,
            imported_module,
            binding[1],
            modules,
            resolving,
        )
    if isinstance(binding, (ast.Assign, ast.AnnAssign)):
        value = _assignment_value(binding)
        if isinstance(value, ast.Name) and value.id != name:
            target = _resolve_name(package_dir, module, value.id, modules, resolving)
            if target is not None:
                return target
    if binding is None:
        return None
    return _Definition(module=module, node=binding)


def _definition_has_docstring(definition: _Definition) -> bool:
    node = definition.node
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return bool(ast.get_docstring(node))
    return _following_docstring(definition.module.tree.body, node)


def _public_methods_without_docstrings(node: ast.ClassDef) -> list[str]:
    return [
        member.name
        for member in node.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not member.name.startswith("_")
        and not ast.get_docstring(member)
    ]


def _name_tail(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name_tail(node.func)
    return None


def _is_documented_field_container(node: ast.ClassDef) -> bool:
    return any(_name_tail(decorator) == "dataclass" for decorator in node.decorator_list) or any(
        _name_tail(base) == "BaseModel" for base in node.bases
    )


def _is_class_var(annotation: ast.expr) -> bool:
    target = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    return _name_tail(target) == "ClassVar"


def _public_fields_without_docstrings(node: ast.ClassDef) -> list[str]:
    if not _is_documented_field_container(node):
        return []
    return [
        member.target.id
        for member in node.body
        if isinstance(member, ast.AnnAssign)
        and isinstance(member.target, ast.Name)
        and not member.target.id.startswith("_")
        and not _is_class_var(member.annotation)
        and not _following_docstring(node.body, member)
    ]


def validate_public_api() -> list[str]:
    """Return documentation contract violations for the enrolled packages."""

    failures: list[str] = []
    for package_dir in API_PACKAGE_DIRS:
        modules: dict[tuple[str, ...], _Module] = {}
        facade = _load_module(package_dir, (), modules)
        if facade is None:
            failures.append(f"{package_dir.name}: package facade does not resolve")
            continue
        exports = _literal_exports(facade.tree, facade.path)
        for name in exports:
            definition = _resolve_name(package_dir, facade, name, modules)
            if definition is None:
                failures.append(f"{package_dir.name}: exported {name} does not resolve")
                continue
            if not _definition_has_docstring(definition):
                failures.append(f"{package_dir.name}: exported {name} has no docstring")
            if not isinstance(definition.node, ast.ClassDef):
                continue
            failures.extend(
                f"{package_dir.name}: public method {name}.{method} has no docstring"
                for method in _public_methods_without_docstrings(definition.node)
            )
            failures.extend(
                f"{package_dir.name}: public field {name}.{field} has no docstring"
                for field in _public_fields_without_docstrings(definition.node)
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
