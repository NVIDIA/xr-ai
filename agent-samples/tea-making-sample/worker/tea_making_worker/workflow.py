# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and evaluate YAML-defined guided workflow steps."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_TEMPLATE_RE = re.compile(r"{{\s*([a-zA-Z0-9_.-]+)(?:\s*\|\s*([a-zA-Z0-9_-]+))?\s*}}")
_TEMPERATURE_RE = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([CF])\b",
    re.IGNORECASE,
)
_ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:\d{2})?\b"
)


@dataclass(frozen=True, slots=True)
class ContextField:
    """One field the step agent should maintain in workflow context."""

    name: str
    label: str
    type: str = "string"
    description: str = ""
    default: Any = ""
    required: bool = False
    initialize: bool = True

    def schema(self) -> dict[str, Any]:
        schema_type = (
            self.type
            if self.type
            in {
                "string",
                "number",
                "integer",
                "boolean",
                "array",
                "object",
            }
            else "string"
        )
        out: dict[str, Any] = {"type": schema_type}
        if self.description:
            out["description"] = self.description
        if self.initialize and self.default not in (None, ""):
            out["default"] = self.default
        return out


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One YAML-configured step in the guided workflow."""

    id: int
    name: str
    description: str
    vlm_prompt: str
    agent_prompt: str
    context_fields: tuple[ContextField, ...]
    mechanism: str = "caption_agent"
    read_fields: tuple[str, ...] = ()
    partial_context: bool = False
    advance_when: dict[str, Any] = field(default_factory=dict)
    skip_defaults: dict[str, Any] = field(default_factory=dict)
    agent_tools: tuple[str, ...] = ()
    on_enter_message: str = ""
    on_reminder_message: str = ""
    on_complete_message: str = ""
    on_skip_message: str = ""

    @property
    def is_idle(self) -> bool:
        return self.id == 0

    def context_schema(self) -> dict[str, Any]:
        required = [field.name for field in self.context_fields if field.required and not self.partial_context]
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {field.name: field.schema() for field in self.context_fields},
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    @property
    def writable_fields(self) -> set[str]:
        """Context fields this step is allowed to change."""

        return {item.name for item in self.context_fields}

    @property
    def prompt_fields(self) -> tuple[str, ...]:
        """Ordered context projection supplied to this step's state agent."""

        writes = tuple(item.name for item in self.context_fields)
        return tuple(dict.fromkeys((*self.read_fields, *writes)))


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
    context_fields: tuple[ContextField, ...] = ()
    sparse_context: bool = False

    @classmethod
    def load(cls, path: Path) -> "WorkflowDefinition":
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"workflow YAML must be a mapping: {path}")
        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"workflow YAML must define a non-empty steps list: {path}")
        raw_context = raw.get("context")
        has_context_registry = isinstance(raw_context, dict) and "fields" in raw_context
        registered_fields = _parse_context_fields(raw_context or {}, lazy=True)
        if has_context_registry:
            registered_fields = _extend_registry_from_step_writes(
                registered_fields,
                raw_steps,
            )
        registry = {item.name: item for item in registered_fields}
        steps = tuple(
            sorted(
                (
                    _parse_step(
                        item,
                        registry=registry,
                        registry_enabled=has_context_registry,
                    )
                    for item in raw_steps
                ),
                key=lambda item: item.id,
            )
        )
        ids = [step.id for step in steps]
        if len(set(ids)) != len(ids):
            raise ValueError(f"workflow step IDs must be unique: {path}")
        if 0 not in ids:
            raise ValueError(f"workflow must include idle step id 0: {path}")
        if not has_context_registry:
            registered_fields = list({item.name: item for step in steps for item in step.context_fields}.values())
        definition = cls(
            task=dict(raw.get("task") or {}),
            runtime=dict(raw.get("runtime") or {}),
            steps=steps,
            context_fields=tuple(registered_fields),
            sparse_context=has_context_registry,
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
        return {item.name: copy.deepcopy(item.default) for item in self.context_fields if item.initialize}

    def context_for_step(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Return only populated fields that the current step reads or writes."""

        if not self.sparse_context:
            return dict(context)
        return {name: copy.deepcopy(context[name]) for name in step.prompt_fields if name in context}

    def field_map(self) -> dict[str, ContextField]:
        return {item.name: item for item in self.context_fields}

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
        fields = self.field_map()
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

def _rule_met(rule: dict[str, Any], context: dict[str, Any]) -> bool:
    all_rules = rule.get("all")
    if isinstance(all_rules, list):
        return bool(all_rules) and all(isinstance(item, dict) and _rule_met(item, context) for item in all_rules)

    any_rules = rule.get("any")
    if isinstance(any_rules, list):
        return bool(any_rules) and any(isinstance(item, dict) and _rule_met(item, context) for item in any_rules)

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
    """Render context placeholders and optional speech-oriented filters."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        formatter = (match.group(2) or "").casefold()
        if key.startswith("step.") and step is not None:
            value: Any = getattr(step, key.removeprefix("step."), "")
        elif key.startswith("task.") and task is not None:
            value = _lookup(task, key.removeprefix("task."))
        else:
            value = _lookup(context, key)
        return _format_template_value(value, formatter)

    return _TEMPLATE_RE.sub(replace, template)


def speech_text(text: str) -> str:
    """Normalize common machine-oriented values for text-to-speech output."""

    def temperature(match: re.Match[str]) -> str:
        unit = "Celsius" if match.group(2).casefold() == "c" else "Fahrenheit"
        return f"{match.group(1)} degrees {unit}"

    def timestamp(match: re.Match[str]) -> str:
        return _spoken_local_time(match.group(0))

    normalized = _TEMPERATURE_RE.sub(temperature, text)
    normalized = _ISO_TIMESTAMP_RE.sub(timestamp, normalized)
    normalized = normalized.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"[ \t]+", " ", normalized).strip()


def _format_template_value(value: Any, formatter: str) -> str:
    if value is None:
        return ""
    if formatter == "duration":
        return _spoken_duration(value)
    if formatter in {"local_time", "time"}:
        return _spoken_local_time(value)
    if formatter in {"spoken", "speech", "temperature"}:
        return speech_text(str(value))
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _spoken_duration(value: Any) -> str:
    try:
        total_seconds = max(0, int(value))
    except (TypeError, ValueError):
        return ""
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour" if hours == 1 else f"{hours} hours")
    if minutes:
        parts.append(f"{minutes} minute" if minutes == 1 else f"{minutes} minutes")
    if seconds or not parts:
        parts.append(f"{seconds} second" if seconds == 1 else f"{seconds} seconds")
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _spoken_local_time(value: Any) -> str:
    try:
        if isinstance(value, int | float) or str(value).isdigit():
            number = int(value)
            seconds = number / 1_000_000 if number > 10_000_000_000 else number
            moment = datetime.fromtimestamp(seconds).astimezone()
        else:
            encoded = str(value).replace("Z", "+00:00")
            moment = datetime.fromisoformat(encoded)
            if moment.tzinfo is None:
                moment = moment.astimezone()
    except (OSError, OverflowError, TypeError, ValueError):
        return str(value)
    clock = moment.strftime("%I:%M %p").lstrip("0")
    return clock.replace("AM", "A.M.").replace("PM", "P.M.")


def context_for_prompt(context: dict[str, Any]) -> str:
    """Stable JSON rendering used in model prompts."""

    return json.dumps(context, indent=2, sort_keys=True, ensure_ascii=True)


def _parse_step(
    raw: Any,
    *,
    registry: dict[str, ContextField] | None = None,
    registry_enabled: bool = False,
) -> WorkflowStep:
    if not isinstance(raw, dict):
        raise ValueError(f"workflow step must be a mapping: {raw!r}")
    step_id = int(raw["id"])
    local_fields = tuple(_parse_context_fields(raw.get("context_output") or {}))
    if registry_enabled:
        writes = raw.get("writes")
        if writes is None:
            write_names = tuple(item.name for item in local_fields)
        elif isinstance(writes, dict):
            write_names = tuple(str(name) for name in writes)
        else:
            write_names = _parse_field_names(writes, "writes")
        context_fields = tuple(_registered_field(name, registry or {}, step_id) for name in write_names)
        read_fields = _parse_field_names(raw.get("reads") or (), "reads")
    else:
        context_fields = local_fields
        read_fields = ()
    return WorkflowStep(
        id=step_id,
        name=str(raw.get("name") or f"Step {step_id}"),
        description=str(raw.get("description") or ""),
        vlm_prompt=str(raw.get("vlm_prompt") or ""),
        agent_prompt=str(raw.get("agent_prompt") or ""),
        context_fields=context_fields,
        mechanism=str(raw.get("mechanism") or "caption_agent").strip(),
        read_fields=read_fields,
        partial_context=registry_enabled,
        advance_when=dict(raw.get("advance_when") or {}),
        skip_defaults=dict(raw.get("skip_defaults") or {}),
        agent_tools=_parse_agent_tools(raw.get("agent_tools") or []),
        on_enter_message=str(raw.get("on_enter_message") or ""),
        on_reminder_message=str(raw.get("on_reminder_message") or ""),
        on_complete_message=str(raw.get("on_complete_message") or ""),
        on_skip_message=str(raw.get("on_skip_message") or ""),
    )


def _parse_agent_tools(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError("agent_tools must be a list of tool names")
    names = tuple(str(name).strip() for name in raw if str(name).strip())
    if len(names) != len(set(names)):
        raise ValueError("agent_tools must not contain duplicate names")
    return names


def _parse_context_fields(raw: Any, *, lazy: bool = False) -> list[ContextField]:
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
        initialize = not lazy or "initial" in spec
        default = spec.get(
            "initial",
            spec.get("default", _default_for_type(field_type)),
        )
        fields.append(
            ContextField(
                name=str(name),
                label=str(spec.get("label") or name),
                type=field_type,
                description=str(spec.get("description") or ""),
                default=default,
                required=bool(spec.get("required", False)),
                initialize=initialize,
            )
        )
    return fields


def _extend_registry_from_step_writes(
    fields: list[ContextField],
    raw_steps: list[Any],
) -> list[ContextField]:
    """Allow a step to declare a new write field inline when convenient."""

    by_name = {item.name: item for item in fields}
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict) or not isinstance(raw_step.get("writes"), dict):
            continue
        for item in _parse_context_fields(raw_step["writes"], lazy=True):
            by_name.setdefault(item.name, item)
    return list(by_name.values())


def _parse_field_names(raw: Any, key: str) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple):
        raise ValueError(f"step {key} must be a list or mapping")
    names = tuple(str(item).strip() for item in raw if str(item).strip())
    if len(names) != len(set(names)):
        raise ValueError(f"step {key} must not contain duplicate fields")
    return names


def _registered_field(
    name: str,
    registry: dict[str, ContextField],
    step_id: int,
) -> ContextField:
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(f"step {step_id} references unknown context field {name}") from exc


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

    registered = set(definition.field_map())
    for step in definition.steps:
        unknown_reads = set(step.read_fields) - registered
        if unknown_reads:
            names = ", ".join(sorted(unknown_reads))
            raise ValueError(f"step {step.id} reads unknown fields {names}: {path}")
        if step.is_idle:
            continue
        if not step.description.strip():
            raise ValueError(f"step {step.id} must define description: {path}")
        if not step.agent_prompt.strip():
            raise ValueError(f"step {step.id} must define agent_prompt: {path}")
        if not step.mechanism:
            raise ValueError(f"step {step.id} must define a mechanism: {path}")
        if not step.advance_when:
            raise ValueError(f"step {step.id} must define advance_when: {path}")

        writable = step.writable_fields
        unknown_defaults = set(step.skip_defaults) - writable
        if unknown_defaults:
            names = ", ".join(sorted(unknown_defaults))
            raise ValueError(f"step {step.id} skip_defaults use unknown fields {names}: {path}")


def _default_for_type(field_type: str) -> Any:
    return {
        "boolean": False,
        "array": [],
        "object": {},
        "integer": 0,
        "number": 0.0,
    }.get(field_type, "")


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


def _compare_number(value: Any, expected: Any, comparator: Any) -> bool:
    try:
        left = float(value)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    return bool(comparator(left, right))


__all__ = [
    "ContextField",
    "WorkflowDefinition",
    "WorkflowSession",
    "WorkflowStep",
    "context_for_prompt",
    "render_template",
    "speech_text",
]
