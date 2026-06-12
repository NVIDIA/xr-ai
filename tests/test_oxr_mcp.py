# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placeholder smoke test for the oxr-mcp server — skipped on CPU.

oxr-mcp imports ``isaacteleop`` (native OpenXR + HeadTracker bindings) at
module top, and its server opens a headless OpenXR session against a running
CloudXR runtime. Neither is installable or runnable on a CPU-only CI box, so
oxr-mcp is intentionally absent from ``tests/pyproject.toml`` and this test
self-skips. See ``tests/README.md`` for the full rationale and the manual
GPU-host verification path. The ``importorskip`` collects as a skip so the
suite stays green without pulling the native dependency into the test venv.
"""
from __future__ import annotations

import pytest

# isaacteleop is the native OpenXR/HeadTracker package oxr-mcp imports at
# module top; it is not available (or meaningful) on a CPU-only host.
pytest.importorskip(
    "isaacteleop",
    reason="oxr-mcp needs native isaacteleop + a CloudXR OpenXR runtime; "
           "not CPU-viable. See tests/README.md.",
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.skip(reason="oxr-mcp smoke test requires a GPU host with CloudXR — see tests/README.md")
async def test_oxr_mcp_placeholder():
    pass
