# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render compact internal values as natural user-facing text."""

import re
from collections.abc import Mapping
from typing import Any

_FIELD = re.compile(r"{{\s*([a-z][a-z0-9_-]*)(?:\s*\|\s*([a-z_]+))?\s*}}")


def render_message(template: str, values: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = values.get(match.group(1), "unknown")
        formatter = match.group(2)
        if formatter is None:
            return str(value)
        if formatter == "temperature_c":
            return f"{_number(value)} degrees Celsius"
        if formatter == "duration":
            return _duration(int(value))
        raise ValueError(f"unknown message formatter: {formatter}")

    return _FIELD.sub(replace, template)


def _duration(seconds: int) -> str:
    minutes, remainder = divmod(seconds, 60)
    parts = []
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if remainder or not parts:
        parts.append(f"{remainder} second{'s' if remainder != 1 else ''}")
    return " and ".join(parts)


def _number(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


__all__ = ["render_message"]
