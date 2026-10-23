# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load guide selection and verification settings for guided procedures.

Read the visual verification timeout from guide-agent-config.yaml and accept
the guide path resolved by the launcher. Shared voice, frame,
and model settings remain in the worker's config.py.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class GuideConfig:
    guide_path: Path
    vision_timeout_s: float = 15.0


def load_guide_config(path: Path, *, guide_path: str | None) -> GuideConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(guide_path, str) or not guide_path.strip():
        raise ValueError("--mode guide requires a non-empty GUIDE_PATH")
    timeout = float(data.get("vision_timeout_s", 15.0))
    if not 0 < timeout < float("inf"):
        raise ValueError("vision_timeout_s must be finite and positive")
    return GuideConfig(Path(guide_path), timeout)
