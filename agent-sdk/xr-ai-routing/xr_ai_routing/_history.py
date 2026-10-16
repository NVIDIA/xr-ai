# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share private history metadata within the ``xr_ai_routing`` package."""

from __future__ import annotations

import contextvars
from datetime import UTC, datetime

from ._contracts import RoutingTurnContext

_active_history_timestamps: contextvars.ContextVar[tuple[int | None, ...]] = contextvars.ContextVar(
    "xr_ai_conversation_history_timestamps", default=()
)
_active_history_exchange_ids: contextvars.ContextVar[tuple[object | None, ...]] = contextvars.ContextVar(
    "xr_ai_conversation_history_exchange_ids", default=()
)
_active_exchange_id: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "xr_ai_conversation_exchange_id", default=None
)


def _format_minute(timestamp_us: int) -> str:
    """Format an exchange timestamp at the routing and prompt display precision."""

    return datetime.fromtimestamp(timestamp_us / 1_000_000, UTC).strftime("%Y-%m-%d %H:%M UTC")


def current_turn_context() -> RoutingTurnContext:
    """Return a stable snapshot of metadata for the active routed turn.

    Agent code called within a routed turn can use this accessor to associate
    work with the current exchange and its bounded history. The returned
    snapshot does not change when routing advances to another exchange. Outside
    an active routed turn, it contains ``None`` and empty history tuples.
    """

    return RoutingTurnContext(
        exchange_id=_active_exchange_id.get(),
        history_exchange_ids=_active_history_exchange_ids.get(),
        history_timestamps=_active_history_timestamps.get(),
    )
