# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static sample CLI discovery and the Sphinx directive that renders it."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class CliArgument:
    """One statically declared ``ArgumentParser.add_argument`` call."""

    flags: tuple[str, ...]
    help: str
    action: str | None = None
    choices: tuple[Any, ...] = ()
    metavar: str | None = None
    required: bool = False

    @property
    def label(self) -> str:
        """Return the option spelling shown in the generated reference."""

        value = ", ".join(self.flags)
        if self.action in {"store_true", "store_false", "count", "help"}:
            return value
        metavar = self.metavar
        if self.choices:
            metavar = "{" + ",".join(str(choice) for choice in self.choices) + "}"
        if metavar is None:
            metavar = self.flags[-1].lstrip("-").replace("-", "_").upper()
        return f"{value} {metavar}"

    @property
    def usage(self) -> str:
        """Return this argument's fragment of the command synopsis."""

        label = self.label
        return label if self.required or not self.flags[0].startswith("-") else f"[{label}]"


@dataclass(frozen=True)
class CliCommand:
    """An installed top-level sample command and its static argument metadata."""

    program: str
    project_dir: Path
    description: str
    arguments: tuple[CliArgument, ...]

    @property
    def invocation(self) -> str:
        """Return the canonical repository-root invocation."""

        base = f"uv run --project {self.project_dir.as_posix()} {self.program}"
        usage = " ".join(argument.usage for argument in self.arguments)
        return f"{base} {usage}" if usage else base


def _literal(node: ast.AST | None, *, path: Path, label: str) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {label} must be a literal for CLI documentation") from error


def _extract_arguments(path: Path) -> tuple[CliArgument, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    arguments: list[tuple[int, CliArgument]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        flags = tuple(_literal(argument, path=path, label="argument name") for argument in node.args)
        if not flags or not all(isinstance(flag, str) for flag in flags):
            raise ValueError(f"{path}: add_argument names must be literal strings")
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        help_text = _literal(keywords.get("help"), path=path, label="argument help")
        if not isinstance(help_text, str) or not help_text.strip():
            raise ValueError(f"{path}: {flags[-1]} must have literal help text")
        action = _literal(keywords.get("action"), path=path, label="argument action")
        choices = _literal(keywords.get("choices"), path=path, label="argument choices")
        metavar = _literal(keywords.get("metavar"), path=path, label="argument metavar")
        required = _literal(keywords.get("required"), path=path, label="argument required")
        arguments.append(
            (
                node.lineno,
                CliArgument(
                    flags=flags,
                    help=" ".join(help_text.split()),
                    action=action,
                    choices=tuple(choices or ()),
                    metavar=metavar,
                    required=bool(required),
                ),
            )
        )
    return tuple(argument for _line, argument in sorted(arguments))


def load_cli_catalog(repository_root: Path) -> tuple[CliCommand, ...]:
    """Discover installed commands from top-level sample projects without imports."""

    commands: list[CliCommand] = []
    samples_dir = repository_root / "agent-samples"
    for project_path in sorted(samples_dir.glob("*/pyproject.toml")):
        project = tomllib.loads(project_path.read_text(encoding="utf-8"))
        scripts = project.get("project", {}).get("scripts", {})
        for program, target in sorted(scripts.items()):
            if not isinstance(target, str) or ":" not in target:
                raise ValueError(f"{project_path}: script {program!r} has an invalid target")
            module_name, _callable = target.split(":", maxsplit=1)
            module_path = project_path.parent.joinpath(*module_name.split(".")).with_suffix(".py")
            tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
            module_doc = ast.get_docstring(tree, clean=True) or ""
            description = module_doc.splitlines()[0] if module_doc else program
            commands.append(
                CliCommand(
                    program=program,
                    project_dir=project_path.parent.relative_to(repository_root),
                    description=description,
                    arguments=_extract_arguments(module_path),
                )
            )
    return tuple(commands)


def _directive_type():
    from docutils import nodes
    from docutils.parsers.rst import Directive

    class CliReferenceDirective(Directive):
        has_content = False

        def run(self):
            repository_root = Path(self.state.document.settings.env.srcdir).parents[1]
            container = nodes.container(classes=["xr-ai-cli-reference"])
            for command in load_cli_catalog(repository_root):
                section = nodes.section(ids=[nodes.make_id(command.program)])
                section += nodes.title(text=command.program)
                section += nodes.paragraph(text=command.description)
                invocation = f"$ {command.invocation}"
                literal = nodes.literal_block(invocation, invocation)
                literal["language"] = "console"
                section += literal
                if command.arguments:
                    section += nodes.rubric(text="Options")
                    definitions = nodes.definition_list()
                    for argument in command.arguments:
                        term = nodes.term()
                        term += nodes.literal(text=argument.label)
                        definition = nodes.definition()
                        definition += nodes.paragraph(text=argument.help)
                        item = nodes.definition_list_item()
                        item += term
                        item += definition
                        definitions += item
                    section += definitions
                else:
                    section += nodes.paragraph(text="This command has no sample-specific options.")
                container += section
            return [container]

    return CliReferenceDirective


def setup(app):
    """Register the static CLI reference directive with Sphinx."""

    app.add_directive("xr-ai-cli-reference", _directive_type())
    return {"parallel_read_safe": True, "parallel_write_safe": True}
