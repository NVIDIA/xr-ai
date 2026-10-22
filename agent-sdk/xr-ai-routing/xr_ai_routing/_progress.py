# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind response-start notifications within the ``xr_ai_routing`` package."""

from __future__ import annotations

import contextvars
from collections.abc import Callable

_response_started: contextvars.ContextVar[Callable[[], bool] | None] = contextvars.ContextVar(
    "xr_ai_routing_response_started", default=None,
)


def notify_response_started() -> bool:
    """Notify the active router that this destination is starting its response.

    Call immediately before delivering the first meaningful output or
    application action. Destinations publish output independently of the router.
    The notification contains no response text, emits no output, and
    suppresses an optional acknowledgment that has not already been committed
    to its output callback. It also supplies the first-response timing sample
    if this answer completes successfully. Calling it when work merely begins
    would suppress the acknowledgment before a response is ready.

    Returns whether the response leads the turn's output. ``False`` means the
    router already delivered or committed control text for this turn, such as
    an admission preamble or acknowledgment, so the response should follow
    that output instead of superseding it. Outside a routed answer nothing
    precedes the response and the result is ``True``.

    The notification is bound to the current ``answer()`` execution through
    its asynchronous context, including inherited child-task contexts. Only
    the first notification of an active response changes router state;
    repeated notifications return the same result. Notifications from
    completed or interrupted answers return ``False`` without changing router
    state. Call from the answer's event loop; the destination owns and drains
    its child tasks before returning.

    Return values and exceptions report completion, focus, and failure separately.
    """

    notify = _response_started.get()
    return notify() if notify is not None else True
