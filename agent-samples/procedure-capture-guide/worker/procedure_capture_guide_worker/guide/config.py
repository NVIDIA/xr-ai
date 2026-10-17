# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load guide selection and verification settings for guided procedures.

Read guide-agent-config.yaml into GuideConfig: the guide directory and visual
verification timeout, together with the guide name supplied on the command line.
Resolve the guide directory relative to that YAML file. Shared voice, frame,
and model settings remain in the worker's config.py.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class GuideConfig:
    guides_dir: Path
    guide_name: str
    vision_timeout_s: float = 15.0


def load_guide_config(path: Path, *, guide_name: str | None) -> GuideConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(guide_name, str) or not guide_name.strip():
        raise ValueError("--mode guide requires a non-empty GUIDE_NAME")
    timeout = float(data.get("vision_timeout_s", 15.0))
    if not 0 < timeout < float("inf"):
        raise ValueError("vision_timeout_s must be finite and positive")
    return GuideConfig(path.resolve().parent / data.get("guides_dir", "../guides"), guide_name.strip(), timeout)
