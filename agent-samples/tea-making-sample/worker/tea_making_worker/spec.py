# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict task-neutral schema for the declarative guidance workflow."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_TYPES = {"boolean", "integer", "number", "string"}


@dataclass(frozen=True, slots=True)
class StateField:
    """One typed field in participant-local workflow state."""

    type: str
    description: str
    initial: Any = None
    initialized: bool = False


@dataclass(frozen=True, slots=True)
class Trigger:
    """Periodic source used to observe one workflow step."""

    function: str
    interval_s: float
    arguments: dict[str, Any]
    result_field: str | None = None


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    """Prompt and tool names available to one model turn."""

    prompt: str
    tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Evidence:
    """Deterministic gate applied before a completion commit."""

    pattern: str
    consecutive: int


@dataclass(frozen=True, slots=True)
class Step:
    """One homogeneous observe-and-answer workflow step."""

    id: str
    title: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    trigger: Trigger
    agent: AgentPolicy
    voice: AgentPolicy
    evidence: Evidence | None
    complete_when: dict[str, Any]
    next_step: str | None
    complete_on_skip: bool
    state_on_skip: dict[str, Any]
    enter_message: str
    complete_message: str
    skip_message: str

    @property
    def context_fields(self) -> tuple[str, ...]:
        """Return the stable sparse state projection visible at this step."""

        return tuple(dict.fromkeys((*self.reads, *self.writes)))

    def is_complete(self, state: dict[str, Any]) -> bool:
        """Evaluate the declarative completion predicate."""

        return bool(self.complete_when) and all(
            state.get(name) == value
            for name, value in self.complete_when.items()
        )


@dataclass(frozen=True, slots=True)
class Workflow:
    """Validated guidance workflow definition."""

    name: str
    start_step: str
    foreground_prompt: str
    complete_message: str
    state_fields: dict[str, StateField]
    steps: dict[str, Step]

    def step(self, step_id: str) -> Step:
        """Return one validated step by identifier."""

        return self.steps[step_id]

    def initial_state(self) -> dict[str, Any]:
        """Build a fresh deep-copied initial state."""

        return {
            name: copy.deepcopy(field.initial)
            for name, field in self.state_fields.items()
            if field.initialized
        }

    def project(self, step: Step, state: dict[str, Any]) -> dict[str, Any]:
        """Return only state explicitly visible to the supplied step."""

        return {
            name: copy.deepcopy(state[name])
            for name in step.context_fields
            if name in state
        }


def load_workflow(path: Path) -> Workflow:
    """Load and validate a YAML workflow definition."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"workflow must be a mapping: {path}")
    task = _mapping(raw.get("task"), "task")
    fields = _fields(_mapping(raw.get("state"), "state"))
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("steps must be a non-empty list")
    steps = [_step(item, fields) for item in raw_steps]
    by_id = {step.id: step for step in steps}
    if len(by_id) != len(steps):
        raise ValueError("step ids must be unique")
    start = str(task.get("start_step", ""))
    if start not in by_id:
        raise ValueError(f"unknown start step: {start!r}")
    for step in steps:
        if step.next_step is not None and step.next_step not in by_id:
            raise ValueError(
                f"step {step.id!r} has unknown next step {step.next_step!r}"
            )
    return Workflow(
        name=str(task.get("name", "guided workflow")),
        start_step=start,
        foreground_prompt=str(task.get("foreground_prompt", "")).strip(),
        complete_message=str(
            task.get("complete_message", "Workflow complete.")
        ).strip(),
        state_fields=fields,
        steps=by_id,
    )


def _fields(raw: dict[str, Any]) -> dict[str, StateField]:
    fields: dict[str, StateField] = {}
    for name, value in raw.items():
        if not _ID.fullmatch(str(name)):
            raise ValueError(f"invalid state field: {name!r}")
        item = _mapping(value, f"state.{name}")
        kind = str(item.get("type", "string"))
        if kind not in _TYPES:
            raise ValueError(
                f"state.{name}.type must be one of {sorted(_TYPES)}"
            )
        fields[str(name)] = StateField(
            type=kind,
            description=str(item.get("description", "")).strip(),
            initial=copy.deepcopy(item.get("initial")),
            initialized="initial" in item,
        )
    return fields


def _step(value: Any, fields: dict[str, StateField]) -> Step:
    raw = _mapping(value, "step")
    step_id = str(raw.get("id", ""))
    if not _ID.fullmatch(step_id):
        raise ValueError(f"invalid step id: {step_id!r}")
    if "auto_advance" in raw:
        raise ValueError(
            f"step {step_id!r} cannot auto-advance; advancement is user-controlled"
        )
    reads = _names(raw.get("reads", []), f"steps.{step_id}.reads")
    writes = _names(raw.get("writes", []), f"steps.{step_id}.writes")
    completion = _mapping(
        raw.get("complete_when"),
        f"steps.{step_id}.complete_when",
    )
    state_on_skip = _mapping(
        raw.get("state_on_skip", {}),
        f"steps.{step_id}.state_on_skip",
    )
    unknown = (
        set(reads)
        | set(writes)
        | completion.keys()
        | state_on_skip.keys()
    ) - fields.keys()
    if unknown:
        raise ValueError(
            f"step {step_id!r} references unknown state: {sorted(unknown)}"
        )
    trigger = _mapping(raw.get("trigger"), f"steps.{step_id}.trigger")
    agent = _mapping(raw.get("agent"), f"steps.{step_id}.agent")
    voice = _mapping(raw.get("voice"), f"steps.{step_id}.voice")
    messages = _mapping(
        raw.get("messages", {}),
        f"steps.{step_id}.messages",
    )
    interval = float(trigger.get("interval_s", 0))
    if interval <= 0:
        raise ValueError(
            f"step {step_id!r} trigger interval must be positive"
        )
    return Step(
        id=step_id,
        title=str(raw.get("title", step_id)).strip(),
        reads=reads,
        writes=writes,
        trigger=Trigger(
            function=str(trigger.get("function", "")).strip(),
            interval_s=interval,
            arguments=_mapping(
                trigger.get("arguments", {}),
                f"steps.{step_id}.trigger.arguments",
            ),
            result_field=(
                str(trigger["result_field"])
                if trigger.get("result_field")
                else None
            ),
        ),
        agent=AgentPolicy(
            prompt=str(agent.get("prompt", "")).strip(),
            tools=_names(
                agent.get("tools", []),
                f"steps.{step_id}.agent.tools",
            ),
        ),
        voice=AgentPolicy(
            prompt=str(voice.get("prompt", "")).strip(),
            tools=_names(
                voice.get("tools", []),
                f"steps.{step_id}.voice.tools",
            ),
        ),
        evidence=_evidence(raw.get("evidence"), step_id),
        complete_when=dict(completion),
        next_step=(
            str(raw["next"])
            if raw.get("next") is not None
            else None
        ),
        complete_on_skip=bool(raw.get("complete_on_skip", False)),
        state_on_skip=dict(state_on_skip),
        enter_message=str(messages.get("enter", "")).strip(),
        complete_message=str(messages.get("complete", "")).strip(),
        skip_message=str(messages.get("skip", "")).strip(),
    )


def _evidence(value: Any, step_id: str) -> Evidence | None:
    if value is None:
        return None
    raw = _mapping(value, f"steps.{step_id}.evidence")
    pattern = str(raw.get("pattern", ""))
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"step {step_id!r} has invalid evidence pattern: {exc}"
        ) from exc
    consecutive = int(raw.get("consecutive", 1))
    if not pattern or consecutive < 1:
        raise ValueError(
            f"step {step_id!r} evidence needs a pattern and positive consecutive count"
        )
    return Evidence(pattern=pattern, consecutive=consecutive)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item
        for item in value
    ):
        raise ValueError(f"{label} must be a list of names")
    return tuple(value)


__all__ = [
    "AgentPolicy",
    "Evidence",
    "StateField",
    "Step",
    "Trigger",
    "Workflow",
    "load_workflow",
]
