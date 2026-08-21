# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expected-degradation boundary for subagent tool loops.

Rejected input (a ValueError anywhere in the failure chain, e.g. an
unresolvable object description) becomes an error payload the model reads,
and the turn continues. Any other failure propagates and aborts the turn.
"""

import json
from collections.abc import Iterable

from loguru import logger
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tools import ToolInvocationResult

_TRANSPORT_MARKERS = ("RPCError", "Timeout", "Connection", "unavailable", "not started", "disabled")


def as_unavailable(error: BaseException, what: str) -> ValueError | None:
    """Classify transport/availability failures as expected degradation.

    Returns a model-readable ValueError for outages of an external feed,
    None for anything else so genuine bugs keep propagating.
    """
    seen: set[int] = set()
    node: BaseException | None = error
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, (TimeoutError, ConnectionError, OSError)):
            return ValueError(f"{what} is unavailable ({type(node).__name__}: {node})")
        node = node.__cause__ or node.__context__
    text = str(error)
    if any(marker in text for marker in _TRANSPORT_MARKERS):
        return ValueError(f"{what} is unavailable ({type(error).__name__}: {text})")
    return None


def _rejection(exc: BaseException) -> str | None:
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, ValueError):
            return str(node)
        node = node.__cause__ or node.__context__
    # Relay layers re-raise with the original message embedded rather than
    # chained; fall back to the message text.
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


__all__ = ["as_unavailable", "tolerant_toolset"]
