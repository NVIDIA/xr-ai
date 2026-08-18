# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static sample CLI discovery and the Sphinx directive that renders it."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

_VALUE_ACTIONS = {None, "append", "extend", "store"}
_NO_VALUE_ACTIONS = {
    "append_const",
    "count",
    "help",
    "store_const",
    "store_false",
    "store_true",
    "version",
}
_SUPPORTED_NARGS = {"?", "*", "+"}


@dataclass(frozen=True)
class CliArgument:
    """One statically declared ``ArgumentParser.add_argument`` call."""

    flags: tuple[str, ...]
    help: str
    action: str | None = None
    choices: tuple[Any, ...] | None = None
    metavar: str | None = None
    nargs: str | int | None = None
    required: bool = False

    @property
    def is_positional(self) -> bool:
        """Return whether this declaration is a positional argument."""

        return not self.flags[0].startswith("-")

    @property
    def _metavar(self) -> str:
        if self.choices is not None:
            return "{" + ",".join(str(choice) for choice in self.choices) + "}"
        if self.metavar is not None:
            return self.metavar
        if self.is_positional:
            return self.flags[0]
        return self.flags[-1].lstrip("-").replace("-", "_").upper()

    @property
    def _value_usage(self) -> str:
        if self.action in _NO_VALUE_ACTIONS:
            return ""

        metavar = self._metavar
        if self.nargs is None or self.nargs == 1:
            return metavar
        if isinstance(self.nargs, int):
            return " ".join(metavar for _ in range(self.nargs))
        if self.nargs == "?":
            return f"[{metavar}]"
        if self.nargs == "*":
            return f"[{metavar} ...]"
        return f"{metavar} [{metavar} ...]"

    @property
    def label(self) -> str:
        """Return the spelling shown in the generated definition list."""

        if self.is_positional:
            return self._metavar

        value = ", ".join(self.flags)
        return f"{value} {self._value_usage}" if self._value_usage else value

    @property
    def usage(self) -> str:
        """Return this argument's fragment of the command synopsis."""

        if self.is_positional:
            return self._value_usage
        return self.label if self.required else f"[{self.label}]"


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
        positional = not flags[0].startswith("-")
        if positional and (len(flags) != 1 or any(flag.startswith("-") for flag in flags)):
            raise ValueError(f"{path}: positional arguments must have exactly one name")
        if not positional and not all(flag.startswith("-") for flag in flags):
            raise ValueError(f"{path}: option declarations cannot mix names and flags")
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        help_text = _literal(keywords.get("help"), path=path, label="argument help")
        if not isinstance(help_text, str) or not help_text.strip():
            raise ValueError(f"{path}: {flags[-1]} must have literal help text")
        action = _literal(keywords.get("action"), path=path, label="argument action")
        if not isinstance(action, (str, type(None))) or action not in (
            _VALUE_ACTIONS | _NO_VALUE_ACTIONS
        ):
            raise ValueError(f"{path}: {flags[-1]} uses unsupported action {action!r}")
        choices_value = _literal(keywords.get("choices"), path=path, label="argument choices")
        if choices_value is not None and not isinstance(choices_value, (list, tuple)):
            raise ValueError(f"{path}: {flags[-1]} choices must be a literal list or tuple")
        choices = tuple(choices_value) if choices_value is not None else None
        metavar = _literal(keywords.get("metavar"), path=path, label="argument metavar")
        if metavar is not None and not isinstance(metavar, str):
            raise ValueError(f"{path}: {flags[-1]} metavar must be a literal string")
        nargs = _literal(keywords.get("nargs"), path=path, label="argument nargs")
        valid_nargs = (
            nargs is None
            or (isinstance(nargs, str) and nargs in _SUPPORTED_NARGS)
            or (isinstance(nargs, int) and not isinstance(nargs, bool) and nargs >= 1)
        )
        if not valid_nargs:
            raise ValueError(f"{path}: {flags[-1]} uses unsupported nargs {nargs!r}")
        if action in _NO_VALUE_ACTIONS and (
            positional or nargs is not None or choices is not None or metavar is not None
        ):
            raise ValueError(
                f"{path}: {flags[-1]} no-value action cannot use positional, nargs, choices, or metavar"
            )
        required_value = _literal(
            keywords.get("required"), path=path, label="argument required"
        )
        if required_value is not None and not isinstance(required_value, bool):
            raise ValueError(f"{path}: {flags[-1]} required must be a literal boolean")
        if positional and "required" in keywords:
            raise ValueError(f"{path}: {flags[-1]} positional cannot set required")
        arguments.append(
            (
                node.lineno,
                CliArgument(
                    flags=flags,
                    help=" ".join(help_text.split()),
                    action=action,
                    choices=choices,
                    metavar=metavar,
                    nargs=nargs,
                    required=bool(required_value),
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
                    groups = (
                        (
                            "Positional arguments",
                            tuple(
                                argument
                                for argument in command.arguments
                                if argument.is_positional
                            ),
                        ),
                        (
                            "Options",
                            tuple(
                                argument
                                for argument in command.arguments
                                if not argument.is_positional
                            ),
                        ),
                    )
                    for title, arguments in groups:
                        if not arguments:
                            continue
                        section += nodes.rubric(text=title)
                        definitions = nodes.definition_list()
                        for argument in arguments:
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
