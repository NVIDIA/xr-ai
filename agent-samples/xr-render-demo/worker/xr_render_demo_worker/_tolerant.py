# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expected-degradation boundary for subagent tool loops.

Rejected input (a ValueError anywhere in the failure chain, e.g. an
unresolvable object description) becomes an error payload the model reads,
and the turn continues. Transport and availability failures are converted
to such rejections by ``as_unavailable``; anything else propagates and
aborts the turn.
"""

import json
import re
from collections.abc import Iterable
from typing import NoReturn

from loguru import logger
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tools import ToolInvocationResult

_TRANSPORT_TYPE_NAMES = frozenset({"RPCError", "FrameUnavailable"})

# Relay layers re-raise with the original exception's class name embedded in
# the message and the cause chain erased, so the wrapped text is the only
# place the real type survives. Match class-name tokens, never ordinary
# English words ("unavailable", "disabled") that self-match error prose.
_TRANSPORT_TOKENS = re.compile(
    r"\b(?:RPCError|FrameUnavailable|TimeoutError|ConnectionError|"
    r"connection refused|StatusCode\.UNAVAILABLE)\b",
    re.IGNORECASE,
)


def as_unavailable(error: BaseException, what: str) -> ValueError | None:
    """Classify transport/availability failures as expected degradation.

    Returns a model-readable ValueError for outages of an external feed,
    None for anything else so genuine bugs keep propagating.
    """
    seen: set[int] = set()
    node: BaseException | None = error
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if (
            isinstance(node, (TimeoutError, ConnectionError))
            or type(node).__name__ in _TRANSPORT_TYPE_NAMES
        ):
            return ValueError(f"{what} is unavailable ({type(node).__name__}: {node})")
        node = node.__cause__ or node.__context__
    if _TRANSPORT_TOKENS.search(str(error)):
        return ValueError(f"{what} is unavailable ({type(error).__name__}: {error})")
    return None


def reraise_unavailable(error: BaseException, what: str) -> NoReturn:
    """Re-raise *error* as expected degradation when it is a transport
    failure of *what*, unchanged otherwise."""
    degraded = as_unavailable(error, what)
    if degraded is None:
        raise error
    logger.debug("{} degraded: {}", what, degraded)
    raise degraded from error


def _rejection(exc: BaseException) -> str | None:
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, ValueError):
            return str(node)
        node = node.__cause__ or node.__context__
    # Same Relay rewrap as _TRANSPORT_TOKENS handles above.
    detail = str(exc)
    if "ValueError: " in detail:
        return detail.split("ValueError: ", 1)[1]
    return None


class _TolerantTool(Tool):
    async def invoke(self, arguments: str) -> ToolInvocationResult:
        try:
            return await super().invoke(arguments)
        except Exception as exc:
            detail = _rejection(exc)
            if detail is None:
                logger.exception("tool {} failed unexpectedly", self.name)
                raise
            logger.debug("tool {} rejected input: {!r}", self.name, detail)
            return ToolInvocationResult(
                content=json.dumps({"error": "ValueError", "detail": detail}),
                return_direct=False,
            )


def tolerant_toolset(tools: Iterable[Tool]) -> ToolSet:
    return ToolSet([
        _TolerantTool(
            tool.name, tool.description, tool.request_model, tool.result_model,
            tool.handler, return_direct=tool.return_direct,
        )
        for tool in tools
    ])


__all__ = ["as_unavailable", "reraise_unavailable", "tolerant_toolset"]
