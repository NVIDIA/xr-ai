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

from ..conversation._policy import ConversationPolicy
from ..flows import ObservationPolicy

_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_TYPES = {"boolean", "integer", "number", "string"}
_TRUE_BOOL_STRINGS = {"1", "true", "yes", "on"}
_FALSE_BOOL_STRINGS = {"0", "false", "no", "off"}


def _parse_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOL_STRINGS:
            return True
        if normalized in _FALSE_BOOL_STRINGS:
            return False
    accepted = sorted(_TRUE_BOOL_STRINGS | _FALSE_BOOL_STRINGS)
    raise ValueError(f"{label} must be a boolean or one of {accepted} (got {value!r})")


@dataclass(frozen=True, slots=True)
class StateField:
    """One typed field in participant-local workflow state."""

    type: str
    description: str
    initial: Any = None
    initialized: bool = False

    def accepts(self, value: Any) -> bool:
        """Return whether a value satisfies this field's declared type."""

        if self.type == "boolean":
            return isinstance(value, bool)
        if self.type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if self.type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return isinstance(value, str)


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
    """Deterministic gate over consecutive conclusive observations."""

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
    observation: ObservationPolicy
    voice: AgentPolicy
    evidence: Evidence | None
    complete_when: dict[str, Any]
    next_step: str | None
    complete_on_skip: bool
    state_on_skip: dict[str, Any]
    enter_message: str
    complete_message: str
    skip_message: str
    context: dict[str, Trigger]

    @property
    def context_fields(self) -> tuple[str, ...]:
        """Return the stable sparse state projection visible at this step."""

        return tuple(dict.fromkeys((*self.reads, *self.writes)))

    def is_complete(self, state: dict[str, Any]) -> bool:
        """Evaluate the declarative completion predicate."""

        return bool(self.complete_when) and all(state.get(name) == value for name, value in self.complete_when.items())


@dataclass(frozen=True, slots=True)
class Workflow:
    """Validated guidance workflow definition."""

    name: str
    start_step: str
    foreground_prompt: str
    complete_message: str
    state_fields: dict[str, StateField]
    foreground: ConversationPolicy | None
    steps: dict[str, Step]
    advancement: str = "automatic"
    manual_advance_message: str = "Say next when ready."

    def step(self, step_id: str) -> Step:
        """Return one validated step by identifier."""

        return self.steps[step_id]

    def initial_state(self) -> dict[str, Any]:
        """Build a fresh deep-copied initial state."""

        return {name: copy.deepcopy(field.initial) for name, field in self.state_fields.items() if field.initialized}

    def project(self, step: Step, state: dict[str, Any]) -> dict[str, Any]:
        """Return only state explicitly visible to the supplied step."""

        return {name: copy.deepcopy(state[name]) for name in step.context_fields if name in state}


def load_workflow(path: Path) -> Workflow:
    """Load and validate a YAML workflow definition."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"workflow must be a mapping: {path}")
    task = _mapping(raw.get("task"), "task")
    advancement = task.get("advancement", "automatic")
    if advancement not in ("automatic", "manual"):
        raise ValueError("task.advancement must be automatic or manual")
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
            raise ValueError(f"step {step.id!r} has unknown next step {step.next_step!r}")
    return Workflow(
        name=str(task.get("name", "guided workflow")),
        start_step=start,
        foreground_prompt=str(task.get("foreground_prompt", "")).strip(),
        complete_message=str(task.get("complete_message", "Workflow complete.")).strip(),
        state_fields=fields,
        foreground=(ConversationPolicy.model_validate(task["foreground"]) if "foreground" in task else None),
        steps=by_id,
        advancement=advancement,
        manual_advance_message=str(task.get("manual_advance_message", "Say next when ready.")).strip(),
    )


def _fields(raw: dict[str, Any]) -> dict[str, StateField]:
    fields: dict[str, StateField] = {}
    for name, value in raw.items():
        if not _ID.fullmatch(str(name)):
            raise ValueError(f"invalid state field: {name!r}")
        item = _mapping(value, f"state.{name}")
        kind = str(item.get("type", "string"))
        if kind not in _TYPES:
            raise ValueError(f"state.{name}.type must be one of {sorted(_TYPES)}")
        field = StateField(
            type=kind,
            description=str(item.get("description", "")).strip(),
            initial=copy.deepcopy(item.get("initial")),
            initialized="initial" in item,
        )
        if field.initialized and not field.accepts(field.initial):
            raise ValueError(f"state.{name}.initial must be {kind}")
        fields[str(name)] = field
    return fields


def _step(value: Any, fields: dict[str, StateField]) -> Step:
    raw = _mapping(value, "step")
    step_id = str(raw.get("id", ""))
    if not _ID.fullmatch(step_id):
        raise ValueError(f"invalid step id: {step_id!r}")
    if "auto_advance" in raw:
        raise ValueError(f"step {step_id!r}: use task.advancement instead of step.auto_advance")
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
    unknown = (set(reads) | set(writes) | completion.keys() | state_on_skip.keys()) - fields.keys()
    if unknown:
        raise ValueError(f"step {step_id!r} references unknown state: {sorted(unknown)}")
    _validate_state_values(
        completion,
        fields,
        f"steps.{step_id}.complete_when",
    )
    _validate_state_values(
        state_on_skip,
        fields,
        f"steps.{step_id}.state_on_skip",
    )
    trigger = _mapping(raw.get("trigger"), f"steps.{step_id}.trigger")
    observation = ObservationPolicy.model_validate(raw.get("observation"))
    patch_fields = set(observation.updates) | (set(observation.lookup.updates) if observation.lookup else set())
    patch_fields |= set(observation.retain)
    if observation.lookup:
        patch_fields |= set(observation.lookup.source_fields.values())
    if not patch_fields <= set(writes):
        raise ValueError(f"step {step_id!r} observation updates must be writable")
    if observation.lookup:
        for name in observation.lookup.source_fields.values():
            if fields[name].type != "string":
                raise ValueError("provenance bindings require string state fields")
    for patch in (observation.updates, observation.lookup.updates if observation.lookup else {}):
        for name, value in patch.items():
            if isinstance(value, str) and value.startswith("$"):
                _validate_binding(value, set(reads) | set(writes), set(observation.fields))
                if value == "$clock.now_us" and fields[name].type != "integer":
                    raise ValueError("clock binding requires an integer state field")
            elif not fields[name].accepts(value):
                raise ValueError(f"step {step_id!r} update {name!r} has wrong type")
    if observation.evaluation:
        for value in observation.evaluation.arguments.values():
            if isinstance(value, str) and value.startswith("$"):
                _validate_binding(value, set(reads) | set(writes), set(observation.fields))
    voice = _mapping(raw.get("voice"), f"steps.{step_id}.voice")
    if "choices" in voice:
        raise ValueError("step voice choices are replaced by conversation tools and context projections")
    messages = _mapping(
        raw.get("messages", {}),
        f"steps.{step_id}.messages",
    )
    interval = float(trigger.get("interval_s", 0))
    if interval <= 0:
        raise ValueError(f"step {step_id!r} trigger interval must be positive")
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
            result_field=(str(trigger["result_field"]) if trigger.get("result_field") else None),
        ),
        observation=observation,
        context=_context_projections(raw.get("context", {}), set(reads) | set(writes)),
        voice=AgentPolicy(
            prompt=str(voice.get("prompt", "")).strip(),
            tools=_names(
                voice.get("tools", []),
                f"steps.{step_id}.voice.tools",
            ),
        ),
        evidence=_evidence(raw.get("evidence"), step_id),
        complete_when=dict(completion),
        next_step=(str(raw["next"]) if raw.get("next") is not None else None),
        complete_on_skip=_parse_bool(
            raw.get("complete_on_skip", False),
            f"steps.{step_id}.complete_on_skip",
        ),
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
        raise ValueError(f"step {step_id!r} has invalid evidence pattern: {exc}") from exc
    consecutive = int(raw.get("consecutive", 1))
    if not pattern or consecutive < 1:
        raise ValueError(f"step {step_id!r} evidence needs a pattern and positive consecutive count")
    return Evidence(pattern=pattern, consecutive=consecutive)


def _validate_state_values(
    values: dict[str, Any],
    fields: dict[str, StateField],
    label: str,
) -> None:
    for name, value in values.items():
        field = fields[name]
        if not field.accepts(value):
            raise ValueError(f"{label}.{name} must be {field.type}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
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


def _validate_binding(value: str, state: set[str], observed: set[str]) -> None:
    if value == "$clock.now_us":
        return
    if value.startswith("$state.") and value.removeprefix("$state.") in state:
        return
    if value.startswith("$observation.") and value.removeprefix("$observation.") in observed:
        return
    raise ValueError(f"invalid observation binding: {value}")


def _context_projections(raw: Any, state: set[str]) -> dict[str, Trigger]:
    projections = {}
    for name, value in _mapping(raw, "step.context").items():
        item = _mapping(value, f"step.context.{name}")
        if item.get("function") not in {"clock__now", "clock__timer", "temperature__threshold", "temperature__verify"}:
            raise ValueError("context projections require native read-only tools")
        arguments = _mapping(item.get("arguments", {}), f"step.context.{name}.arguments")
        for binding in arguments.values():
            if isinstance(binding, str) and binding.startswith("$"):
                _validate_binding(binding, state, set())
        projections[str(name)] = Trigger(str(item["function"]), 0, arguments)
    return projections
