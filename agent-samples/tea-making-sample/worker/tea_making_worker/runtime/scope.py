# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-local participant state available to native functions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from .state import Session


@dataclass(slots=True)
class Invocation:
    session: Session
    trace_id: str
    route_operation: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


_CURRENT: ContextVar[Invocation | None] = ContextVar("tea_guidance_invocation", default=None)


@contextmanager
def invocation_scope(session: Session, trace_id: str) -> Iterator[None]:
    token = _CURRENT.set(Invocation(session=session, trace_id=trace_id))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_invocation() -> Invocation:
    invocation = _CURRENT.get()
    if invocation is None:
        raise RuntimeError("workflow function invoked outside a guidance turn")
    return invocation


__all__ = ["Invocation", "current_invocation", "invocation_scope"]
