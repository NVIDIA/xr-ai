# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover the one model error that a corrected tool call can repair."""

from __future__ import annotations

from collections.abc import Callable

from nat.builder.function import Function
from pydantic import ValidationError


async def invoke_with_tool_retry(
    agent: Function,
    payload: str,
    *,
    retry: Callable[[str], None],
    skip_repeated_invalid: bool = False,
) -> str:
    for attempt in range(2):
        try:
            return await agent.ainvoke(payload, to_type=str)
        except Exception as exc:
            if not _is_tool_schema_error(exc):
                raise
            if attempt == 1:
                if skip_repeated_invalid:
                    return ""
                raise
            feedback = _schema_feedback(exc)
            retry(feedback)
            payload = f"{payload}\nRetry tool arguments: {feedback}"
    raise AssertionError("unreachable")


def _is_tool_schema_error(exc: Exception) -> bool:
    return isinstance(getattr(exc, "source", None), ValidationError)


def _schema_feedback(exc: Exception) -> str:
    source: ValidationError = getattr(exc, "source")
    return "; ".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
        for error in source.errors(include_url=False, include_input=False)
    )


__all__ = ["invoke_with_tool_retry"]
