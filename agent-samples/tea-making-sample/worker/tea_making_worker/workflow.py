# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and evaluate YAML-defined guided workflow steps."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_TEMPLATE_RE = re.compile(r"{{\s*([a-zA-Z0-9_.-]+)\s*}}")
_OBSERVATION_FIELD_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9_ ]+)\s*:\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_STEP_STATES = frozenset({"started", "needs_input", "complete"})
_INVALID_CONTEXT_VALUE = object()


@dataclass(frozen=True, slots=True)
class ContextField:
    """One field the step agent should maintain in workflow context."""

    name: str
    label: str
    type: str = "string"
    description: str = ""
    default: Any = ""
    required: bool = False

    def schema(self) -> dict[str, Any]:
        schema_type = self.type if self.type in {
            "string",
            "number",
            "integer",
            "boolean",
            "array",
            "object",
        } else "string"
        out: dict[str, Any] = {"type": schema_type}
        if self.description:
            out["description"] = self.description
        if self.default not in (None, ""):
            out["default"] = self.default
        return out


@dataclass(frozen=True, slots=True)
class TimerSpec:
    """YAML-defined wall-clock timer for a non-visual workflow step."""

    label: str
    started_at_us_field: str
    duration_seconds_field: str
    completion_field: str


@dataclass(frozen=True, slots=True)
class TimerStatus:
    """Current timer values derived from workflow context."""

    label: str
    started_at_us: int
    duration_seconds: int
    elapsed_seconds: int
    remaining_seconds: int
    expired: bool


@dataclass(frozen=True, slots=True)
class StateUpdateSpec:
    """Declarative mapping from a structured VLM field to workflow context."""

    context_field: str
    states: tuple[str, ...]
    observation_key: str = ""
    value_map: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One YAML-configured step in the guided workflow."""

    id: int
    name: str
    description: str
    vlm_prompt: str
    agent_prompt: str
    context_fields: tuple[ContextField, ...]
    advance_when: dict[str, Any] = field(default_factory=dict)
    skip_defaults: dict[str, Any] = field(default_factory=dict)
    agent_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    timer: TimerSpec | None = None
    state_updates: tuple[StateUpdateSpec, ...] = ()
    on_enter_message: str = ""
    on_reminder_message: str = ""
    on_complete_message: str = ""
    on_skip_message: str = ""

    @property
    def is_idle(self) -> bool:
        return self.id == 0

    def context_schema(self) -> dict[str, Any]:
        required = [field.name for field in self.context_fields if field.required]
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                field.name: field.schema()
                for field in self.context_fields
            },
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    def state_update_fields(self, state: str) -> set[str]:
        return {
            update.context_field
            for update in self.state_updates
            if state in update.states
        }

    def observation_context_patch(
        self,
        observation: str,
        *,
        state: str,
    ) -> dict[str, Any]:
        observed = {
            _normalize_observation_key(match.group(1)): match.group(2).strip()
            for match in _OBSERVATION_FIELD_RE.finditer(observation)
        }
        field_types = {item.name: item.type for item in self.context_fields}
        patch: dict[str, Any] = {}
        for update in self.state_updates:
            if state not in update.states or not update.observation_key:
                continue
            raw_value = observed.get(
                _normalize_observation_key(update.observation_key)
            )
            if not raw_value:
                continue
            mapped = update.value_map.get(
                _normalize_observation_value(raw_value),
                raw_value,
            )
            value = _coerce_context_value(
                mapped,
                field_types[update.context_field],
            )
            if value is not _INVALID_CONTEXT_VALUE:
                patch[update.context_field] = value
        return patch


@dataclass(slots=True)
class WorkflowSession:
    """Participant-local workflow state."""

    participant_id: str
    step_id: int
    context: dict[str, Any]
    active: bool = True
    connected: bool = True
    last_frame_pts_us: int = 0
    last_notice_us: int = 0
    last_notice_text: str = ""
    ready_step_id: int | None = None
    step_state: str = "started"
    step_started_us: int = 0
    last_reminder_us: int = 0
    reminder_count: int = 0
    pending_instruction: str = ""
    evaluation_active: bool = False
    user_turn_active: bool = False
    deferred_notice: str = ""
    observation_log: list[dict[str, Any]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """Task, step, and runtime settings loaded from YAML."""

    task: dict[str, Any]
    runtime: dict[str, Any]
    steps: tuple[WorkflowStep, ...]

    @classmethod
    def load(cls, path: Path) -> "WorkflowDefinition":
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"workflow YAML must be a mapping: {path}")
        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"workflow YAML must define a non-empty steps list: {path}")
        steps = tuple(sorted((_parse_step(item) for item in raw_steps), key=lambda item: item.id))
        ids = [step.id for step in steps]
        if len(set(ids)) != len(ids):
            raise ValueError(f"workflow step IDs must be unique: {path}")
        if 0 not in ids:
            raise ValueError(f"workflow must include idle step id 0: {path}")
        definition = cls(
            task=dict(raw.get("task") or {}),
            runtime=dict(raw.get("runtime") or {}),
            steps=steps,
        )
        _validate_definition(definition, path)
        return definition

    @property
    def monitor_interval_s(self) -> float:
        return float(self.runtime.get("monitor_interval_s", 2.0))

    @property
    def min_notice_interval_s(self) -> float:
        return float(self.runtime.get("min_notice_interval_s", 8.0))

    @property
    def reminder_interval_s(self) -> float:
        return float(self.runtime.get("reminder_interval_s", 30.0))

    @property
    def max_reminders_per_step(self) -> int:
        return int(self.runtime.get("max_reminders_per_step", 1))

    @property
    def navigation_timeout_s(self) -> float:
        return float(self.runtime.get("navigation_timeout_s", 4.0))

    @property
    def max_agent_iterations(self) -> int:
        return int(self.runtime.get("max_agent_iterations", 6))

    @property
    def start_triggers(self) -> tuple[str, ...]:
        return tuple(str(item).casefold() for item in self.task.get("start_triggers", ()))

    @property
    def stop_triggers(self) -> tuple[str, ...]:
        return tuple(str(item).casefold() for item in self.task.get("stop_triggers", ()))

    @property
    def status_triggers(self) -> tuple[str, ...]:
        return tuple(str(item).casefold() for item in self.task.get("status_triggers", ()))

    def step_by_id(self, step_id: int) -> WorkflowStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"unknown workflow step id: {step_id}")

    def first_active_step(self) -> WorkflowStep:
        for step in self.steps:
            if not step.is_idle:
                return step
        raise ValueError("workflow has no non-idle steps")

    def next_step(self, step_id: int) -> WorkflowStep | None:
        active = [step for step in self.steps if not step.is_idle]
        for index, step in enumerate(active):
            if step.id == step_id:
                return active[index + 1] if index + 1 < len(active) else None
        return None

    def initial_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {}
        for step in self.steps:
            for item in step.context_fields:
                context.setdefault(item.name, copy.deepcopy(item.default))
        return context

    def apply_skip_defaults(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply YAML-authored defaults for fields still missing on manual advance."""

        now = datetime.now().astimezone()
        values = {
            "now_us": int(now.timestamp() * 1_000_000),
            "now_iso": now.isoformat(timespec="seconds"),
        }
        applied: dict[str, Any] = {}
        fields = {item.name: item for item in step.context_fields}
        for name, value in step.skip_defaults.items():
            field_name = str(name)
            if _unset_for_skip(_lookup(context, field_name), fields.get(field_name)):
                rendered = _render_default(copy.deepcopy(value), values)
                context[field_name] = rendered
                applied[field_name] = rendered
        return applied

    def advance_when_met(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        *,
        ready_to_advance: bool = False,
    ) -> bool:
        rule = step.advance_when or {}
        return _rule_met(rule, context) if rule else ready_to_advance

    def timer_status(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        *,
        now_us: int,
    ) -> TimerStatus | None:
        spec = step.timer
        if spec is None:
            return None
        started_at_us = _positive_int(_lookup(context, spec.started_at_us_field))
        duration_seconds = _positive_int(_lookup(context, spec.duration_seconds_field))
        if started_at_us <= 0 or duration_seconds <= 0:
            return None
        elapsed_us = max(0, now_us - started_at_us)
        duration_us = duration_seconds * 1_000_000
        return TimerStatus(
            label=spec.label,
            started_at_us=started_at_us,
            duration_seconds=duration_seconds,
            elapsed_seconds=elapsed_us // 1_000_000,
            remaining_seconds=max(
                0,
                math.ceil((duration_us - elapsed_us) / 1_000_000),
            ),
            expired=elapsed_us >= duration_us,
        )

    def find_timer_status(
        self,
        context: dict[str, Any],
        *,
        current_step_id: int,
        now_us: int,
    ) -> TimerStatus | None:
        timer_steps = (step for step in self.steps if step.timer is not None)
        ordered = sorted(timer_steps, key=lambda step: step.id != current_step_id)
        for step in ordered:
            status = self.timer_status(step, context, now_us=now_us)
            if status is not None:
                return status
        return None

    def mark_timer_complete(self, step: WorkflowStep, context: dict[str, Any]) -> None:
        if step.timer is not None:
            context[step.timer.completion_field] = True


def _rule_met(rule: dict[str, Any], context: dict[str, Any]) -> bool:
    all_rules = rule.get("all")
    if isinstance(all_rules, list):
        return bool(all_rules) and all(
            isinstance(item, dict) and _rule_met(item, context)
            for item in all_rules
        )

    any_rules = rule.get("any")
    if isinstance(any_rules, list):
        return bool(any_rules) and any(
            isinstance(item, dict) and _rule_met(item, context)
            for item in any_rules
        )

    field_name = rule.get("field")
    if field_name:
        value = _lookup(context, str(field_name))
        if rule.get("exists") is True:
            return _present(value)
        if "equals" in rule:
            return value == rule["equals"]
        if "not_equals" in rule:
            return value != rule["not_equals"]
        if "gt" in rule:
            return _compare_number(value, rule["gt"], lambda left, right: left > right)
        if "gte" in rule:
            return _compare_number(value, rule["gte"], lambda left, right: left >= right)
        if "lt" in rule:
            return _compare_number(value, rule["lt"], lambda left, right: left < right)
        if "lte" in rule:
            return _compare_number(value, rule["lte"], lambda left, right: left <= right)

    all_present = rule.get("all_present") or ()
    if all_present:
        return all(_present(_lookup(context, str(name))) for name in all_present)
    any_present = rule.get("any_present") or ()
    if any_present:
        return any(_present(_lookup(context, str(name))) for name in any_present)
    return False


def render_template(
    template: str,
    *,
    context: dict[str, Any],
    step: WorkflowStep | None = None,
    task: dict[str, Any] | None = None,
) -> str:
    """Render simple ``{{field}}`` placeholders from context, step, and task."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key.startswith("step.") and step is not None:
            return str(getattr(step, key.removeprefix("step."), ""))
        if key.startswith("task.") and task is not None:
            return str(_lookup(task, key.removeprefix("task.")) or "")
        value = _lookup(context, key)
        if isinstance(value, dict | list):
            return json.dumps(value, ensure_ascii=True)
        return "" if value is None else str(value)

    return _TEMPLATE_RE.sub(replace, template)


def context_for_prompt(context: dict[str, Any]) -> str:
    """Stable JSON rendering used in model prompts."""

    return json.dumps(context, indent=2, sort_keys=True, ensure_ascii=True)


def _parse_step(raw: Any) -> WorkflowStep:
    if not isinstance(raw, dict):
        raise ValueError(f"workflow step must be a mapping: {raw!r}")
    step_id = int(raw["id"])
    return WorkflowStep(
        id=step_id,
        name=str(raw.get("name") or f"Step {step_id}"),
        description=str(raw.get("description") or ""),
        vlm_prompt=str(raw.get("vlm_prompt") or ""),
        agent_prompt=str(raw.get("agent_prompt") or ""),
        context_fields=tuple(_parse_context_fields(raw.get("context_output") or {})),
        advance_when=dict(raw.get("advance_when") or {}),
        skip_defaults=dict(raw.get("skip_defaults") or {}),
        agent_tools=_parse_agent_tools(raw.get("agent_tools") or {}),
        timer=_parse_timer(raw.get("timer")),
        state_updates=_parse_state_updates(raw.get("state_updates")),
        on_enter_message=str(raw.get("on_enter_message") or ""),
        on_reminder_message=str(raw.get("on_reminder_message") or ""),
        on_complete_message=str(raw.get("on_complete_message") or ""),
        on_skip_message=str(raw.get("on_skip_message") or ""),
    )


def _parse_agent_tools(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, list):
        return {str(name): {} for name in raw}
    if not isinstance(raw, dict):
        raise ValueError("agent_tools must be a mapping or list")
    tools: dict[str, dict[str, Any]] = {}
    for name, policy in raw.items():
        if policy is not None and not isinstance(policy, dict):
            raise ValueError(f"agent tool policy must be a mapping: {name}")
        tools[str(name)] = dict(policy or {})
    return tools


def _parse_state_updates(raw: Any) -> tuple[StateUpdateSpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("state_updates must be a list")
    updates: list[StateUpdateSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each state_updates entry must be a mapping")
        context_field = str(item.get("context_field") or "").strip()
        if not context_field:
            raise ValueError("state_updates entry must define context_field")
        raw_states = item.get("states") or ["started", "needs_input"]
        if not isinstance(raw_states, list) or not raw_states:
            raise ValueError("state_updates states must be a non-empty list")
        value_map = item.get("value_map") or {}
        if not isinstance(value_map, dict):
            raise ValueError("state_updates value_map must be a mapping")
        updates.append(
            StateUpdateSpec(
                context_field=context_field,
                states=tuple(str(state).casefold().strip() for state in raw_states),
                observation_key=str(item.get("observation_key") or "").strip(),
                value_map={
                    _normalize_observation_value(key): value
                    for key, value in value_map.items()
                },
            )
        )
    return tuple(updates)


def _parse_timer(raw: Any) -> TimerSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("timer must be a mapping")
    required = (
        "started_at_us_field",
        "duration_seconds_field",
        "completion_field",
    )
    missing = [name for name in required if not str(raw.get(name) or "").strip()]
    if missing:
        raise ValueError(f"timer is missing required fields: {', '.join(missing)}")
    return TimerSpec(
        label=str(raw.get("label") or "timer").strip(),
        started_at_us_field=str(raw["started_at_us_field"]),
        duration_seconds_field=str(raw["duration_seconds_field"]),
        completion_field=str(raw["completion_field"]),
    )


def _parse_context_fields(raw: Any) -> list[ContextField]:
    if isinstance(raw, dict) and "fields" in raw:
        raw = raw["fields"]
    if not isinstance(raw, dict):
        return []
    fields: list[ContextField] = []
    for name, spec in raw.items():
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            spec = {"description": str(spec)}
        field_type = str(spec.get("type", "string"))
        default = spec.get("default", _default_for_type(field_type))
        fields.append(
            ContextField(
                name=str(name),
                label=str(spec.get("label") or name),
                type=field_type,
                description=str(spec.get("description") or ""),
                default=default,
                required=bool(spec.get("required", False)),
            )
        )
    return fields


def _validate_definition(definition: WorkflowDefinition, path: Path) -> None:
    for key in (
        "monitor_interval_s",
        "min_notice_interval_s",
        "reminder_interval_s",
        "navigation_timeout_s",
        "max_agent_iterations",
    ):
        if float(definition.runtime.get(key, 1)) <= 0:
            raise ValueError(f"runtime.{key} must be positive: {path}")
    if definition.max_reminders_per_step < 0:
        raise ValueError(f"runtime.max_reminders_per_step cannot be negative: {path}")

    for step in definition.steps:
        if step.is_idle:
            continue
        if not step.description.strip():
            raise ValueError(f"step {step.id} must define description: {path}")
        if not step.agent_prompt.strip():
            raise ValueError(f"step {step.id} must define agent_prompt: {path}")
        if step.timer is None and not step.vlm_prompt.strip():
            raise ValueError(f"step {step.id} must define vlm_prompt or timer: {path}")
        if not step.advance_when:
            raise ValueError(f"step {step.id} must define advance_when: {path}")

        fields = {field.name for field in step.context_fields}
        unknown_defaults = set(step.skip_defaults) - fields
        if unknown_defaults:
            names = ", ".join(sorted(unknown_defaults))
            raise ValueError(f"step {step.id} skip_defaults use unknown fields {names}: {path}")
        unknown_update_fields = {
            update.context_field for update in step.state_updates
        } - fields
        if unknown_update_fields:
            names = ", ".join(sorted(unknown_update_fields))
            raise ValueError(
                f"step {step.id} updates unknown context fields {names}: {path}"
            )
        invalid_states = {
            state
            for update in step.state_updates
            for state in update.states
            if state not in _STEP_STATES
        }
        if invalid_states:
            names = ", ".join(sorted(invalid_states))
            raise ValueError(
                f"step {step.id} state_updates use invalid states {names}: {path}"
            )
        if step.state_update_fields("complete") and not step.vlm_prompt.strip():
            raise ValueError(
                f"step {step.id} cannot update complete state without a "
                f"vlm_prompt: {path}"
            )
        for tool_name, policy in step.agent_tools.items():
            if not policy.get("auto_invoke"):
                continue
            outputs = policy.get("context_outputs")
            if not isinstance(outputs, dict) or not outputs:
                raise ValueError(
                    f"step {step.id} automatic tool {tool_name} must define "
                    f"context_outputs: {path}"
                )
            unknown_outputs = {str(name) for name in outputs.values()} - fields
            if unknown_outputs:
                names = ", ".join(sorted(unknown_outputs))
                raise ValueError(
                    f"step {step.id} automatic tool {tool_name} uses unknown context "
                    f"fields {names}: {path}"
                )
            empty_field = str(policy.get("when_context_empty") or "").strip()
            if empty_field and empty_field not in fields:
                raise ValueError(
                    f"step {step.id} automatic tool {tool_name} checks unknown context "
                    f"field {empty_field}: {path}"
                )
        if step.timer is not None:
            timer_fields = {
                step.timer.started_at_us_field,
                step.timer.duration_seconds_field,
                step.timer.completion_field,
            }
            unknown_timer_fields = timer_fields - fields
            if unknown_timer_fields:
                names = ", ".join(sorted(unknown_timer_fields))
                raise ValueError(f"step {step.id} timer uses unknown fields {names}: {path}")


def _default_for_type(field_type: str) -> Any:
    return {
        "boolean": False,
        "array": [],
        "object": {},
        "integer": 0,
        "number": 0.0,
    }.get(field_type, "")


def _normalize_observation_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def _normalize_observation_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).casefold().strip()


def _coerce_context_value(value: Any, field_type: str) -> Any:
    if field_type == "string":
        return str(value)
    if field_type == "boolean":
        if isinstance(value, bool):
            return value
        normalized = str(value).casefold().strip()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return _INVALID_CONTEXT_VALUE
    if field_type == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return _INVALID_CONTEXT_VALUE
    if field_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return _INVALID_CONTEXT_VALUE
    if field_type in {"array", "object"}:
        decoded = value
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return _INVALID_CONTEXT_VALUE
        expected_type = list if field_type == "array" else dict
        return decoded if isinstance(decoded, expected_type) else _INVALID_CONTEXT_VALUE
    return value


def _render_default(value: Any, values: dict[str, Any]) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "{{now_us}}":
            return values["now_us"]
        if stripped == "{{now_iso}}":
            return values["now_iso"]
        return render_template(value, context=values)
    if isinstance(value, list):
        return [_render_default(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _render_default(item, values) for key, item in value.items()}
    return value


def _lookup(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, list | dict) and not value:
        return False
    return True


def _unset_for_skip(value: Any, field: ContextField | None) -> bool:
    if not _present(value):
        return True
    if field is None or value != field.default:
        return False
    return field.default in (None, "", False, 0, 0.0)


def _positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _compare_number(value: Any, expected: Any, comparator: Any) -> bool:
    try:
        left = float(value)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    return bool(comparator(left, right))


__all__ = [
    "ContextField",
    "StateUpdateSpec",
    "TimerSpec",
    "TimerStatus",
    "WorkflowDefinition",
    "WorkflowSession",
    "WorkflowStep",
    "context_for_prompt",
    "render_template",
]
