# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share private history metadata within the ``xr_ai_routing`` package."""

from __future__ import annotations

import contextvars
from datetime import UTC, datetime

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
