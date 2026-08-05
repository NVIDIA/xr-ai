# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small structured events for replaying human guidance tests from logs."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


def emit(name: str, **fields: Any) -> None:
    record = {"event": name, **fields}
    logger.bind(event=name, **fields).info("event {}", json.dumps(record, ensure_ascii=False, default=str))


__all__ = ["emit"]
