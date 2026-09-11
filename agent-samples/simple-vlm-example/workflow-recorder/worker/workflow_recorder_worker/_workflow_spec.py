# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, task-neutral schema for locally discovered SOP guides."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_TYPES = frozenset({"boolean", "integer", "number", "string"})
_STATUSES = frozenset({"draft", "approved"})
_TRIGGERS = frozenset({"current_view", "clock__timer"})
_TOOLS = frozenset({"current_view", "clock__now", "clock__timer"})
_ROOT_KEYS = frozenset({"schema_version", "task", "state", "steps"})
_TASK_KEYS = frozenset(
    {
        "id",
        "name",
        "version",
        "status",
        "source_session",
        "start_step",
        "foreground_prompt",
        "complete_message",
    }
)
_STATE_KEYS = frozenset({"type", "description", "initial"})
_STEP_KEYS = frozenset(
    {
        "id",
        "title",
        "reads",
        "writes",
        "trigger",
        "agent",
        "voice",
        "evidence",
        "complete_when",
        "next",
        "complete_on_skip",
        "state_on_skip",
        "messages",
    }
)
_TRIGGER_KEYS = frozenset({"function", "interval_s", "arguments", "result_field"})
_POLICY_KEYS = frozenset({"prompt", "tools"})
_EVIDENCE_KEYS = frozenset({"pattern", "consecutive", "commit"})
_MESSAGE_KEYS = frozenset({"enter", "complete", "skip"})


@dataclass(frozen=True, slots=True)
class StateField:
    type: str
    description: str
    initial: Any = None
    initialized: bool = False

    def accepts(self, value: Any) -> bool:
        if self.type == "boolean":
            return isinstance(value, bool)
        if self.type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if self.type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return isinstance(value, str)


@dataclass(frozen=True, slots=True)
class Trigger:
    function: str
    interval_s: float
    arguments: dict[str, Any]
    result_field: str | None


@dataclass(frozen=True, slots=True)
class Policy:
    prompt: str
    tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Evidence:
    pattern: str
    consecutive: int
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Step:
    id: str
    title: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    trigger: Trigger
    agent: Policy
    voice: Policy
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
        return tuple(dict.fromkeys((*self.reads, *self.writes)))

    def is_complete(self, state: dict[str, Any]) -> bool:
        return all(state.get(name) == value for name, value in self.complete_when.items())


@dataclass(frozen=True, slots=True)
class Workflow:
    id: str
    name: str
    version: int
    status: str
    source_session: str
    start_step: str
    foreground_prompt: str
    complete_message: str
    state_fields: dict[str, StateField]
    steps: dict[str, Step]

    @property
    def runnable(self) -> bool:
        return self.status == "approved"

    def initial_state(self) -> dict[str, Any]:
        return {name: copy.deepcopy(field.initial) for name, field in self.state_fields.items() if field.initialized}

    def project(self, step: Step, state: dict[str, Any]) -> dict[str, Any]:
        return {name: copy.deepcopy(state[name]) for name in step.context_fields if name in state}


def load_workflow(path: Path) -> Workflow:
    root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, "workflow")
    _only(root, _ROOT_KEYS, "workflow")
    if root.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    task = _mapping(root.get("task"), "task")
    _only(task, _TASK_KEYS, "task")
    fields = _fields(_mapping(root.get("state"), "state"))
    raw_steps = root.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("steps must be a non-empty list")
    steps = tuple(_step(value, fields) for value in raw_steps)
    by_id = {step.id: step for step in steps}
    if len(by_id) != len(steps):
        raise ValueError("step ids must be unique")
    workflow_id = _identifier(task.get("id"), "task.id")
    name = _required_text(task.get("name"), "task.name")
    version = task.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("task.version must be a positive integer")
    status = str(task.get("status", ""))
    if status not in _STATUSES:
        raise ValueError(f"task.status must be one of {sorted(_STATUSES)}")
    start = str(task.get("start_step", ""))
    if start not in by_id:
        raise ValueError(f"unknown start step: {start!r}")
    for step in steps:
        if step.next_step is not None and step.next_step not in by_id:
            raise ValueError(f"step {step.id!r} has unknown next step {step.next_step!r}")
    _validate_linear_graph(start, by_id)
    return Workflow(
        id=workflow_id,
        name=name,
        version=version,
        status=status,
        source_session=_required_text(task.get("source_session"), "task.source_session"),
        start_step=start,
        foreground_prompt=_required_text(task.get("foreground_prompt"), "task.foreground_prompt"),
        complete_message=_required_text(task.get("complete_message"), "task.complete_message"),
        state_fields=fields,
        steps=by_id,
    )


def _fields(raw: dict[str, Any]) -> dict[str, StateField]:
    fields: dict[str, StateField] = {}
    for raw_name, value in raw.items():
        name = _identifier(raw_name, f"state field {raw_name!r}")
        item = _mapping(value, f"state.{name}")
        _only(item, _STATE_KEYS, f"state.{name}")
        kind = str(item.get("type", ""))
        if kind not in _TYPES:
            raise ValueError(f"state.{name}.type must be one of {sorted(_TYPES)}")
        field = StateField(
            type=kind,
            description=_required_text(item.get("description"), f"state.{name}.description"),
            initial=copy.deepcopy(item.get("initial")),
            initialized="initial" in item,
        )
        if field.initialized and not field.accepts(field.initial):
            raise ValueError(f"state.{name}.initial must be {kind}")
        fields[name] = field
    if not fields:
        raise ValueError("state must declare at least one field")
    return fields


def _step(value: Any, fields: dict[str, StateField]) -> Step:
    raw = _mapping(value, "step")
    _only(raw, _STEP_KEYS, "step")
    step_id = _identifier(raw.get("id"), "step.id")
    reads = _names(raw.get("reads", []), f"steps.{step_id}.reads")
    writes = _names(raw.get("writes", []), f"steps.{step_id}.writes")
    if not writes:
        raise ValueError(f"step {step_id!r} must declare at least one writable field")
    unknown = (set(reads) | set(writes)) - fields.keys()
    if unknown:
        raise ValueError(f"step {step_id!r} references unknown state: {sorted(unknown)}")
    completion = _mapping(raw.get("complete_when"), f"steps.{step_id}.complete_when")
    if not completion or not completion.keys() <= set(writes):
        raise ValueError(f"step {step_id!r} complete_when must be non-empty and writable")
    skip = _mapping(raw.get("state_on_skip", {}), f"steps.{step_id}.state_on_skip")
    if not skip.keys() <= set(writes):
        raise ValueError(f"step {step_id!r} state_on_skip must only use writable fields")
    _state_values(completion, fields, f"steps.{step_id}.complete_when")
    _state_values(skip, fields, f"steps.{step_id}.state_on_skip")

    trigger_raw = _mapping(raw.get("trigger"), f"steps.{step_id}.trigger")
    _only(trigger_raw, _TRIGGER_KEYS, f"steps.{step_id}.trigger")
    function = str(trigger_raw.get("function", ""))
    if function not in _TRIGGERS:
        raise ValueError(f"step {step_id!r} uses unsupported trigger {function!r}")
    interval = _positive_number(trigger_raw.get("interval_s"), f"steps.{step_id}.trigger.interval_s")
    arguments = _mapping(trigger_raw.get("arguments", {}), f"steps.{step_id}.trigger.arguments")
    _validate_state_refs(arguments, set(reads) | set(writes), fields, step_id)
    result_field = trigger_raw.get("result_field")
    if result_field is not None and (not isinstance(result_field, str) or not result_field):
        raise ValueError(f"steps.{step_id}.trigger.result_field must be a non-empty string")

    agent = _policy(raw.get("agent"), f"steps.{step_id}.agent")
    voice = _policy(raw.get("voice"), f"steps.{step_id}.voice")
    evidence = _evidence(raw.get("evidence"), step_id, writes, fields)
    messages = _mapping(raw.get("messages"), f"steps.{step_id}.messages")
    _only(messages, _MESSAGE_KEYS, f"steps.{step_id}.messages")
    complete_on_skip = raw.get("complete_on_skip", False)
    if not isinstance(complete_on_skip, bool):
        raise ValueError(f"steps.{step_id}.complete_on_skip must be boolean")
    return Step(
        id=step_id,
        title=_required_text(raw.get("title"), f"steps.{step_id}.title"),
        reads=reads,
        writes=writes,
        trigger=Trigger(function, interval, arguments, result_field),
        agent=agent,
        voice=voice,
        evidence=evidence,
        complete_when=dict(completion),
        next_step=str(raw["next"]) if raw.get("next") is not None else None,
        complete_on_skip=complete_on_skip,
        state_on_skip=dict(skip),
        enter_message=_required_text(messages.get("enter"), f"steps.{step_id}.messages.enter"),
        complete_message=_required_text(messages.get("complete"), f"steps.{step_id}.messages.complete"),
        skip_message=_required_text(messages.get("skip"), f"steps.{step_id}.messages.skip"),
    )


def _policy(value: Any, label: str) -> Policy:
    raw = _mapping(value, label)
    _only(raw, _POLICY_KEYS, label)
    tools = _names(raw.get("tools", []), f"{label}.tools")
    unknown = set(tools) - _TOOLS
    if unknown:
        raise ValueError(f"{label} references unsupported tools: {sorted(unknown)}")
    return Policy(_required_text(raw.get("prompt"), f"{label}.prompt"), tools)


def _evidence(value: Any, step_id: str, writes: tuple[str, ...], fields: dict[str, StateField]) -> Evidence | None:
    if value is None:
        return None
    raw = _mapping(value, f"steps.{step_id}.evidence")
    _only(raw, _EVIDENCE_KEYS, f"steps.{step_id}.evidence")
    pattern = _required_text(raw.get("pattern"), f"steps.{step_id}.evidence.pattern")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"step {step_id!r} has invalid evidence regex: {exc}") from exc
    consecutive = raw.get("consecutive")
    if not isinstance(consecutive, int) or isinstance(consecutive, bool) or consecutive < 1:
        raise ValueError(f"steps.{step_id}.evidence.consecutive must be a positive integer")
    commit = _mapping(raw.get("commit", {}), f"steps.{step_id}.evidence.commit")
    if commit and not commit.keys() <= set(writes):
        raise ValueError(f"step {step_id!r} evidence.commit must only use writable fields")
    _state_values(commit, fields, f"steps.{step_id}.evidence.commit")
    return Evidence(pattern, consecutive, dict(commit))


def _validate_linear_graph(start: str, steps: dict[str, Step]) -> None:
    visited: set[str] = set()
    current: str | None = start
    while current is not None:
        if current in visited:
            raise ValueError(f"workflow contains a cycle at step {current!r}")
        visited.add(current)
        current = steps[current].next_step
    unreachable = steps.keys() - visited
    if unreachable:
        raise ValueError(f"workflow has unreachable steps: {sorted(unreachable)}")


def _validate_state_refs(value: Any, visible: set[str], fields: dict[str, StateField], step_id: str) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_state_refs(item, visible, fields, step_id)
    elif isinstance(value, list):
        for item in value:
            _validate_state_refs(item, visible, fields, step_id)
    elif isinstance(value, str) and value.startswith("$state."):
        name = value.removeprefix("$state.")
        if name not in fields:
            raise ValueError(f"step {step_id!r} trigger references unknown state {name!r}")
        if name not in visible:
            raise ValueError(f"step {step_id!r} trigger references undeclared state {name!r}")


def _state_values(values: dict[str, Any], fields: dict[str, StateField], label: str) -> None:
    unknown = values.keys() - fields.keys()
    if unknown:
        raise ValueError(f"{label} references unknown state: {sorted(unknown)}")
    for name, value in values.items():
        if not fields[name].accepts(value):
            raise ValueError(f"{label}.{name} must be {fields[name].type}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _only(value: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")


def _identifier(value: Any, label: str) -> str:
    result = str(value or "")
    if not _ID.fullmatch(result):
        raise ValueError(f"{label} must match {_ID.pattern!r}")
    return result


def _required_text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of names")
    names = tuple(value)
    if len(names) != len(set(names)):
        raise ValueError(f"{label} must not contain duplicates")
    return names


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result
