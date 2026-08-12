# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation at the NeMo Relay service boundary."""

from __future__ import annotations


def headers_from_relay(raw: object) -> dict[str, str]:
    """Return string-only headers supplied by a Relay request intercept."""

    if not isinstance(raw, dict):
        raise TypeError("Relay LLM request headers must be an object")
    headers: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("Relay LLM request headers must be strings")
        headers[name] = value
    return headers


__all__ = ["headers_from_relay"]
