# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expected-degradation boundary for subagent tool loops.

An explicit, plain ValueError from a sample handler (e.g. an unresolvable
object description) becomes an error payload the model reads, and the turn
continues. Validation of model arguments stays in Tool.invoke; internal
validation failures and other exception types propagate, regardless of their
cause or context. Transport failures become rejections through as_unavailable.
"""

import json
from collections.abc import Iterable
from typing import NoReturn

from loguru import logger
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.tools import ToolInvocationResult

_TRANSPORT_TYPE_NAMES = frozenset({"RPCError", "FrameUnavailable"})


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
    return None


def reraise_unavailable(error: BaseException, what: str) -> NoReturn:
    """Re-raise *error* as expected degradation when it is a transport
    failure of *what*, unchanged otherwise."""
    degraded = as_unavailable(error, what)
    if degraded is None:
        raise error
    logger.debug("{} degraded: {}", what, degraded)
    raise degraded from error


class _TolerantTool(Tool):
    async def invoke(self, arguments: str) -> ToolInvocationResult:
        try:
            return await super().invoke(arguments)
        except ValueError as exc:
            # Only sample handlers' explicit rejections are model-repairable;
            # subclasses such as internal Pydantic errors are not user input.
            if type(exc) is not ValueError:
                raise
            detail = str(exc)
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
