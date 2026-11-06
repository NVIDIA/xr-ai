# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the complete worker YAML into validated application configuration.

Worker YAML owns application prompts, service paths and
budgets. The SOP configuration supplies Python arguments; the runner owns step connections. Relative paths resolve
against the selected worker YAML; prompt text has no hidden Python/file fallback.
This module reads files only when ``load_config`` is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from xr_ai_models import DecisionQuestion

from .tea_sop_configuration import SOPConfig, VisualInstructions

_WORKER_PACKAGE = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Resolved application settings loaded from the worker YAML.

    Prompt and decision fields are explicit configuration, not generated defaults.
    The SOP runner receives ordinary Python policy, not a workflow-expression schema.
    """

    models_config: Path
    sop: SOPConfig
    voice_gate_yaml: Path
    artifacts_dir: Path
    rag_endpoint: str
    current_view_prompt: str
    conversation_system_prompt: str
    frame_max_age_s: float
    frame_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None
    web_events_host: str
    web_events_port: int
    web_events_max_events: int
    capture_enabled: bool = True
    context_budget_bytes: int = 12000
    sop_context_budget_bytes: int = 64000
    compaction_input_budget_bytes: int = 12000


def _resolve(config_path: Path | None, raw: str) -> Path:
    """Resolve an absolute path or a path relative to the selected worker YAML."""
    path = Path(raw)
    if config_path is not None and not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _read_config(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, failing clearly when an installed worker lacks its YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Worker YAML not found: {path}; pass --config with the sample worker YAML path.")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"worker config must be a YAML mapping: {path}")
    return data


def _prompt(data: dict[str, Any], name: str) -> str:
    """Require inline prompt text and reject the removed prompt-file configuration."""
    if f"{name}_file" in data:
        raise ValueError(f"{name}_file is no longer supported; put {name} text directly in worker YAML")
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text in worker YAML")
    return value.strip()


def _positive(data: dict[str, Any], name: str, default: float) -> float:
    """Read a strictly positive numeric setting, using its non-prompt default."""
    value = float(data.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _decision(data: dict[str, Any], name: str) -> DecisionQuestion:
    """Validate application instructions and criteria for the existing SDK choice type."""
    decision_configuration = data.get(name)
    if not isinstance(decision_configuration, dict):
        raise ValueError(f"{name} requires instructions and choice criteria in worker YAML")
    if set(decision_configuration) != {"instructions", "criteria"}:
        raise ValueError(f"{name} requires only instructions and criteria")
    instructions = decision_configuration["instructions"]
    criteria = decision_configuration["criteria"]
    if (
        not isinstance(instructions, str)
        or not instructions.strip()
        or not isinstance(criteria, dict)
        or len(criteria) < 2
        or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in criteria.items()
        )
    ):
        raise ValueError(f"{name} has invalid choice instructions/criteria")
    return DecisionQuestion(instructions.strip(), criteria)


def _sop(data: dict[str, Any]) -> SOPConfig:
    """Translate application YAML into ordinary Python constructor arguments."""
    configuration = data.get("sop")
    if not isinstance(configuration, dict):
        raise ValueError("sop application configuration is required")
    expected = set(SOPConfig.__dataclass_fields__)
    if set(configuration) - expected or expected - set(configuration) - {"instrument_unit_question"}:
        raise ValueError(f"sop configuration keys differ: {sorted(set(configuration) ^ expected)}")
    arguments = dict(configuration)
    arguments.setdefault("instrument_unit_question", None)
    if arguments["instrument_unit_question"] is not None:
        arguments["instrument_unit_question"] = _decision(configuration, "instrument_unit_question")
        if set(arguments["instrument_unit_question"].criteria) != {"celsius", "fahrenheit", "not_declared", "unclear"}:
            raise ValueError("instrument units require celsius/fahrenheit/not_declared/unclear")
    for name in ("water", "temperature", "tea_bag"):
        visual = arguments[name]
        if not isinstance(visual, dict) or set(visual) != {"question", "completion"}:
            raise ValueError(f"sop.{name} requires question and completion")
        decision = _decision({"completion": visual["completion"]}, "completion")
        if set(decision.criteria) != {"positive", "negative", "unknown"}:
            raise ValueError(f"sop.{name} completion requires positive/negative/unknown")
        arguments[name] = VisualInstructions(_prompt(visual, "question"), decision)
    for name in ("control_question", "authorization_question"):
        arguments[name] = _decision(configuration, name)
    if set(arguments["control_question"].criteria) != {"start", "stop", "next", "skip", "restart", "none"}:
        raise ValueError("sop control choices are fixed native controls")
    if set(arguments["authorization_question"].criteria) != {"workflow", "identify", "fill_water", "heat_water", "start_steeping", "steep_timer", "assistant_speech", "physical_activity", "other", "not_requested", "unclear"}:
        raise ValueError("sop requested referents must match the native workflow/step targets")
    if arguments["advancement"] not in {"automatic", "manual"}:
        raise ValueError("sop.advancement must be automatic or manual")
    for name in (
        "poll_interval_seconds",
        "inference_timeout_seconds",
        "required_positives",
        "confirmation_window_size",
        "history_limit",
        "observation_text_limit",
        "max_retrieved_passages",
        "retrieved_text_limit",
        "timer_update_interval_seconds",
        "default_temperature_c",
        "default_duration_seconds",
    ):
        if not isinstance(arguments[name], (float, int)) or isinstance(arguments[name], bool) or arguments[name] <= 0:
            raise ValueError(f"sop.{name} must be positive")
    for name in (
        "required_positives",
        "confirmation_window_size",
        "history_limit",
        "observation_text_limit",
        "max_retrieved_passages",
        "retrieved_text_limit",
    ):
        if type(arguments[name]) is not int:
            raise ValueError(f"sop.{name} must be an integer")
    if arguments["required_positives"] > arguments["confirmation_window_size"]:
        raise ValueError("sop positive threshold cannot exceed its window")
    if arguments["max_retrieved_passages"] > 20:
        raise ValueError("sop retrieval count cannot exceed 20")
    if arguments["history_max_age_seconds"] is not None and arguments["history_max_age_seconds"] <= 0:
        raise ValueError("sop history age must be positive or null")
    for name in (
        "identification_question",
        "extraction_instructions",
        "fallback_instructions",
        "retrieval_query_suffix",
        "workflow_description",
        "conversation_description",
        "conversation_instructions",
        "default_tea_name",
    ):
        arguments[name] = _prompt(configuration, name)
    required_messages = {
        "start",
        "stop",
        "restart",
        "next",
        "skip",
        "already_running",
        "not_running",
        "not_ready",
        "changed",
        "complete",
        "failed",
        "identify",
        "fill_water",
        "identification_complete",
        "identification_skipped",
        "water_verified",
        "water_skipped",
        "temperature_verified",
        "temperature_skipped",
        "heat_water",
        "start_steeping",
        "steep_timer",
        "tea_contact_verified",
        "tea_contact_skipped",
        "timer_complete",
        "timer_skipped",
    }
    if arguments["instrument_unit_question"] is not None:
        required_messages.add("unit_needed")
    if not isinstance(arguments["messages"], dict) or set(arguments["messages"]) != required_messages:
        raise ValueError("sop.messages must define all native control/step messages")
    if any(not isinstance(value, str) or not value.strip() for value in arguments["messages"].values()):
        raise ValueError("sop messages must be nonblank text")
    return SOPConfig(**arguments)


def load_config(path: Path | None) -> WorkerConfig:
    """Load complete worker YAML and resolve paths relative to it.

    Without a path, use the source sample's checked-in YAML, or yaml/ in the documented
    sample working directory for an installed worker. Model wording has no packaged
    file or Python fallback; custom YAML owns all prompt values.

    Args:
        path: Explicit worker YAML path, or None for the documented sample lookup.

    Returns:
        Validated application settings with absolute paths and inline model policy.

    Raises:
        FileNotFoundError: The selected/default YAML is absent; pass an explicit --config.
        ValueError: Configuration has an unsupported shape or invalid setting.
    """

    if path is None:
        path = _WORKER_PACKAGE.parents[1] / "yaml" / "tea_making_worker.yaml"
        if not path.is_file():
            path = Path.cwd() / "yaml" / "tea_making_worker.yaml"
    path = path.resolve()
    configuration = _read_config(path)
    idle_timeout_seconds = float(configuration.get("idle_timeout_secs", 0.0))
    web_events_host = str(configuration.get("web_events_host", "127.0.0.1")).strip()
    if not web_events_host:
        raise ValueError("web_events_host must not be empty")
    web_events_port = int(configuration.get("web_events_port", 8092))
    if not 0 <= web_events_port <= 65_535:
        raise ValueError("web_events_port must be between 0 and 65535")
    web_events_max_events = int(configuration.get("web_events_max_events", 5_000))
    if web_events_max_events <= 0:
        raise ValueError("web_events_max_events must be positive")
    capture_configuration = configuration.get("capture", {})
    if (
        not isinstance(capture_configuration, dict)
        or set(capture_configuration) - {"enabled"}
        or not isinstance(capture_configuration.get("enabled", True), bool)
    ):
        raise ValueError("capture requires an enabled boolean")
    unknown_keys = set(configuration) - (set(WorkerConfig.__dataclass_fields__) - {"capture_enabled"} | {"capture"})
    if unknown_keys:
        raise ValueError(f"unsupported worker settings: {sorted(unknown_keys)}; use inline application SOP arguments")
    return WorkerConfig(
        models_config=_resolve(path, str(configuration.get("models_config", "models.local.json"))),
        sop=_sop(configuration),
        voice_gate_yaml=_resolve(path, str(configuration.get("voice_gate_yaml", "voice_gate.yaml"))),
        artifacts_dir=_resolve(path, str(configuration.get("artifacts_dir", "../artifacts"))),
        rag_endpoint=str(configuration.get("rag_endpoint", "tcp://127.0.0.1:8340")),
        current_view_prompt=_prompt(configuration, "current_view_prompt"),
        conversation_system_prompt=_prompt(configuration, "conversation_system_prompt"),
        frame_max_age_s=_positive(configuration, "frame_max_age_s", 3.0),
        frame_timeout_s=_positive(configuration, "frame_timeout_s", 5.0),
        silence_duration=_positive(configuration, "silence_duration", 0.8),
        min_speech=_positive(configuration, "min_speech", 0.15),
        silero_threshold=_positive(configuration, "silero_threshold", 0.3),
        idle_timeout_secs=idle_timeout_seconds if idle_timeout_seconds > 0 else None,
        web_events_host=web_events_host,
        web_events_port=web_events_port,
        web_events_max_events=web_events_max_events,
        capture_enabled=capture_configuration.get("enabled", True),
        context_budget_bytes=int(_positive(configuration, "context_budget_bytes", 12000)),
        compaction_input_budget_bytes=int(_positive(configuration, "compaction_input_budget_bytes", 40000)),
        sop_context_budget_bytes=int(_positive(configuration, "sop_context_budget_bytes", 64000)),
    )


__all__ = ["WorkerConfig", "load_config"]
