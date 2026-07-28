# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness polling used by :class:`xr_ai_voice.VoiceSession`."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

ProbeFn = Callable[[], Awaitable[bool]]


async def wait_for_services(
    probes: dict[str, ProbeFn],
    *,
    poll_interval: float = 5.0,
) -> None:
    """Block until every named probe returns True."""
    pending = set(probes)
    while pending:
        for name in list(pending):
            if await probes[name]():
                logger.info("{} ready", name)
                pending.discard(name)
        if pending:
            logger.info("still waiting for: {}", ", ".join(sorted(pending)))
            await asyncio.sleep(poll_interval)
