# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the guide selection and verification settings for replay.

Shared worker settings and model endpoints remain in the parent config module.
This module resolves replay paths relative to their owning YAML file.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    guides_dir: Path
    guide_name: str
    vision_timeout_s: float = 15.0


def load_replay_config(path: Path, *, guide_name: str | None) -> ReplayConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(guide_name, str) or not guide_name.strip():
        raise ValueError("--mode replay requires a non-empty GUIDE_NAME")
    timeout = float(data.get("vision_timeout_s", 15.0))
    if not 0 < timeout < float("inf"):
        raise ValueError("vision_timeout_s must be finite and positive")
    return ReplayConfig(path.resolve().parent / data.get("guides_dir", "../guides"), guide_name.strip(), timeout)
