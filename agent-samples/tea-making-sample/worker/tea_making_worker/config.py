# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the complete worker YAML into validated application configuration.

Worker YAML owns application prompts, background decisions, service paths and
budgets. The separate workflow YAML owns task/step policy. Relative paths resolve
against the selected worker YAML; prompt text has no hidden Python/file fallback.
This module reads files only when ``load_config`` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from xr_ai_models import DecisionQuestion

from .flows import ObservationPolicy

_PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Resolved application settings loaded from the worker YAML.

    Prompt and decision fields are explicit configuration, not generated defaults.
    Workflow policy is loaded separately from ``workflow_config`` during composition.
    """

    models_config: Path
    workflow_config: Path
    voice_gate_yaml: Path
    artifacts_dir: Path
    rag_endpoint: str
    foreground_prompt: str
    current_view_prompt: str
    change_watch_decision: DecisionQuestion
    video_delta_decision: DecisionQuestion
    change_watch_caption_prompt: str
    change_watch_event_prompt: str
    transcript_summary_prompt: str
    video_caption_prompt: str
    video_delta_prompt: str
    change_watch_default_instruction: str
    change_watch_interval_s: float
    transcript_summary_interval_s: float
    video_log_interval_s: float
    background_history_size: int
    frame_max_age_s: float
    frame_timeout_s: float
    vlm_timeout_s: float
    silence_duration: float
    min_speech: float
    silero_threshold: float
    idle_timeout_secs: float | None
    web_events_host: str
    web_events_port: int
    web_events_max_events: int
    capture_enabled: bool = True
    context_budget_bytes: int = 12000
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
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _decision(data: dict[str, Any], name: str) -> DecisionQuestion:
    """Validate an inline three-outcome decision using the shared observation schema."""
    decision_configuration = data.get(name)
    if not isinstance(decision_configuration, dict):
        raise ValueError(f"{name} requires instructions and choice criteria in worker YAML")
    policy = ObservationPolicy.model_validate({"kind": "choice", "updates": {}, **decision_configuration})
    return DecisionQuestion(policy.instructions, policy.criteria)


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
        path = _PACKAGE.parents[1] / "yaml" / "tea_making_worker.yaml"
        if not path.is_file():
            path = Path.cwd() / "yaml" / "tea_making_worker.yaml"
    path = path.resolve()
    configuration = _read_config(path)
    background_history_size = int(configuration.get("background_history_size", 50))
    if background_history_size <= 0:
        raise ValueError("background_history_size must be positive")
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
    return WorkerConfig(
        models_config=_resolve(path, str(configuration.get("models_config", "models.local.json"))),
        workflow_config=_resolve(path, str(configuration.get("workflow_config", "workflow.yaml"))),
        voice_gate_yaml=_resolve(path, str(configuration.get("voice_gate_yaml", "voice_gate.yaml"))),
        artifacts_dir=_resolve(path, str(configuration.get("artifacts_dir", "../artifacts"))),
        rag_endpoint=str(configuration.get("rag_endpoint", "tcp://127.0.0.1:8340")),
        foreground_prompt=_prompt(configuration, "foreground_prompt"),
        current_view_prompt=_prompt(configuration, "current_view_prompt"),
        change_watch_decision=_decision(configuration, "change_watch_decision"),
        video_delta_decision=_decision(configuration, "video_delta_decision"),
        change_watch_caption_prompt=_prompt(configuration, "change_watch_caption_prompt"),
        change_watch_event_prompt=_prompt(configuration, "change_watch_event_prompt"),
        transcript_summary_prompt=_prompt(configuration, "transcript_summary_prompt"),
        video_caption_prompt=_prompt(configuration, "video_caption_prompt"),
        video_delta_prompt=_prompt(configuration, "video_delta_prompt"),
        change_watch_default_instruction=_prompt(configuration, "change_watch_default_instruction"),
        change_watch_interval_s=_positive(configuration, "change_watch_interval_s", 2.0),
        transcript_summary_interval_s=_positive(configuration, "transcript_summary_interval_s", 120.0),
        video_log_interval_s=_positive(configuration, "video_log_interval_s", 2.0),
        background_history_size=background_history_size,
        frame_max_age_s=_positive(configuration, "frame_max_age_s", 3.0),
        frame_timeout_s=_positive(configuration, "frame_timeout_s", 5.0),
        vlm_timeout_s=_positive(configuration, "vlm_timeout_s", 15.0),
        silence_duration=_positive(configuration, "silence_duration", 0.8),
        min_speech=_positive(configuration, "min_speech", 0.15),
        silero_threshold=_positive(configuration, "silero_threshold", 0.3),
        idle_timeout_secs=idle_timeout_seconds if idle_timeout_seconds > 0 else None,
        web_events_host=web_events_host,
        web_events_port=web_events_port,
        web_events_max_events=web_events_max_events,
        capture_enabled=capture_configuration.get("enabled", True),
        context_budget_bytes=int(_positive(configuration, "context_budget_bytes", 12000)),
        compaction_input_budget_bytes=int(_positive(configuration, "compaction_input_budget_bytes", 12000)),
    )


__all__ = ["WorkerConfig", "load_config"]
