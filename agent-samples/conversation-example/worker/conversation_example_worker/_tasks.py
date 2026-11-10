# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wait for owned tasks without interrupting their cleanup."""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from typing import Any


async def wait_despite_cancellation(tasks: Collection[asyncio.Task[Any]]) -> None:
    """Wait until every task is done, even if the caller is cancelled meanwhile.

    The wait never cancels the tasks. A cancellation of the caller received while waiting is
    raised once they are done, so it cannot cut short their own cancellation cleanup.
    """

    cancelled = False
    pending = set(tasks)
    while pending:
        try:
            _, pending = await asyncio.wait(pending)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
