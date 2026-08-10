# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic unit tests for video_mcp_server startup / ready-file logic.

These tests exercise ``_wait_until_bound`` directly — no GPU, no NVENC, no
real uvicorn server.  They cover the three outcomes that the polling loop must
handle:

1. Server starts normally   → ``started`` flips True, function returns.
2. Serve task exits early   → task is done before ``started`` is set,
                              function returns without hanging.
3. Startup timeout          → neither condition satisfied within the deadline,
                              task is cancelled and function returns.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from video_mcp_server.__main__ import _wait_until_bound

pytestmark = pytest.mark.asyncio


def _mock_server(*, started: bool = False) -> SimpleNamespace:
    return SimpleNamespace(started=started)


async def test_wait_until_bound_success() -> None:
    """started becomes True quickly → _wait_until_bound returns, task runs on."""
    server = _mock_server(started=False)

    async def _flip_then_run() -> None:
        await asyncio.sleep(0.02)
        server.started = True
        await asyncio.sleep(10)  # stays alive after binding

    task = asyncio.create_task(_flip_then_run())
    await _wait_until_bound(server, task)

    assert server.started
    assert not task.done()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_wait_until_bound_task_exits_early() -> None:
    """Task finishes (raises) before started is set → loop exits, task is done."""
    server = _mock_server(started=False)

    async def _fail_fast() -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("bind failed")

    task = asyncio.create_task(_fail_fast())
    await _wait_until_bound(server, task)

    assert not server.started
    assert task.done()
    assert not task.cancelled()
    assert isinstance(task.exception(), RuntimeError)


async def test_wait_until_bound_timeout_cancels_task() -> None:
    """Neither condition satisfied → task is cancelled before returning."""
    server = _mock_server(started=False)

    async def _hang() -> None:
        await asyncio.sleep(9999)

    task = asyncio.create_task(_hang())
    with patch("video_mcp_server.__main__._STARTUP_TIMEOUT_S", 0.1):
        await _wait_until_bound(server, task)

    assert not server.started
    assert task.done()
    assert task.cancelled()
