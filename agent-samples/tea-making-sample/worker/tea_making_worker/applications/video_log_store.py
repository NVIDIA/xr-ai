# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-participant state for rolling visual activity logs."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VideoLogState:
    path: Path
    captions: deque[str]
    writes: list[dict[str, Any]] = field(default_factory=list)
    active: bool = True
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


__all__ = ["VideoLogState"]
