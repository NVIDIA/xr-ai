# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load guide selection and verification settings for guided procedures.

Read verification timeouts and the guide-run trace directory from
guide-agent-config.yaml, and accept the guide path resolved by the launcher. Shared voice, frame,
and model settings remain in the worker's config.py.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .policy import GuidePolicy, default_policy_path, load_policy


@dataclass(frozen=True, slots=True)
class GuideConfig:
    guide_path: Path
    vision_timeout_s: float = 15.0
    decision_timeout_s: float = 15.0
    traces_dir: Path | None = None
    policy: GuidePolicy = field(default_factory=load_policy)
    policy_path: Path = field(default_factory=default_policy_path)


def load_guide_config(path: Path, *, guide_path: str | None) -> GuideConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(guide_path, str) or not guide_path.strip():
        raise ValueError("--mode guide requires a non-empty GUIDE_PATH")
    timeout = float(data.get("vision_timeout_s", 15.0))
    if not 0 < timeout < float("inf"):
        raise ValueError("vision_timeout_s must be finite and positive")
    if isinstance(data.get("decision_timeout_s"), bool):
        raise ValueError("decision_timeout_s must be finite and positive")
    decision_timeout = float(data.get("decision_timeout_s", 15.0))
    if not 0 < decision_timeout < float("inf"):
        raise ValueError("decision_timeout_s must be finite and positive")
    trace_path = data.get("traces_dir", "../artifacts/guide-runs")
    if trace_path is not None and (not isinstance(trace_path, str) or not trace_path.strip()):
        raise ValueError("traces_dir must be a path or null")
    traces_dir = (path.parent / trace_path).resolve() if trace_path is not None else None
    policy_path = data.get("policy")
    if policy_path is not None and (not isinstance(policy_path, str) or not policy_path.strip()):
        raise ValueError("policy must be a nonempty YAML path")
    source = ((path.parent / policy_path).resolve() if policy_path is not None
              else default_policy_path())
    return GuideConfig(Path(guide_path), timeout, decision_timeout, traces_dir, load_policy(source), source)
