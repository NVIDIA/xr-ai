# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-viable smoke test for the vec-mcp server.

vec-mcp is pure math (no GPU, Docker, OpenXR, or model weights), so the
whole tool surface is exercisable in CI. Spawns ``python -m vec_mcp_server``
against a free port, polls readiness via ``McpClient.list_tools`` (same
pattern as ``test_transcript_mcp.py``), then drives the four FastMCP tools
over StreamableHTTP and checks the arithmetic, including the
coincident-origin error branch in ``along_direction``.
"""
from __future__ import annotations

import asyncio
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml
from fastmcp import Client as McpClient

from _helpers_subprocess import pick_free_port

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _tool_payload(result):
    """Extract structured output from a FastMCP CallToolResult."""
    if hasattr(result, "data") and result.data is not None:
        return result.data
    return getattr(result, "structured_content", None)


async def _wait_ready(url: str, proc: subprocess.Popen, timeout: float) -> None:
    """Poll list_tools() until the server answers or the subprocess dies."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vec_mcp_server exited early (rc={proc.returncode})")
        try:
            async with McpClient(url) as mcp:
                await mcp.list_tools()
                return
        # Connect refused / handshake error while the server is still booting — retry.
        except Exception:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"vec_mcp_server at {url} not ready within {timeout}s")


async def test_vec_mcp_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        tmp         = Path(td)
        config_path = tmp / "vec_mcp_server.yaml"
        port        = pick_free_port()
        url         = f"http://127.0.0.1:{port}/mcp"

        config_path.write_text(yaml.safe_dump({"host": "127.0.0.1", "port": port}))

        env = {**os.environ, "XR_AI_LOG_ROOT": str(tmp / "logs")}

        proc = subprocess.Popen(
            [sys.executable, "-m", "vec_mcp_server", "--config", str(config_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        try:
            await _wait_ready(url, proc, timeout=15.0)

            async with McpClient(url) as mcp:
                tools = {t.name for t in await mcp.list_tools()}
                assert tools == {
                    "between_anchors", "world_offset",
                    "along_direction", "scale_value",
                }

                mid = _tool_payload(await mcp.call_tool(
                    "between_anchors",
                    {"a_x": 0.0, "a_y": 0.0, "a_z": 0.0,
                     "b_x": 2.0, "b_y": 4.0, "b_z": -6.0},
                ))
                assert mid == {"x": 1.0, "y": 2.0, "z": -3.0}

                off = _tool_payload(await mcp.call_tool(
                    "world_offset",
                    {"origin_x": 0.0, "origin_y": 1.5, "origin_z": -1.5, "dy": 0.3},
                ))
                assert off == {"x": 0.0, "y": 1.8, "z": -1.5}

                # 3-4-5 triangle: unit step of 1.0 toward the target lands at
                # (0.6, 0.8, 0) from the origin.
                along = _tool_payload(await mcp.call_tool(
                    "along_direction",
                    {"origin_x": 0.0, "origin_y": 0.0, "origin_z": 0.0,
                     "target_x": 3.0, "target_y": 4.0, "target_z": 0.0,
                     "distance": 1.0},
                ))
                assert math.isclose(along["x"], 0.6, abs_tol=1e-3)
                assert math.isclose(along["y"], 0.8, abs_tol=1e-3)
                assert math.isclose(along["z"], 0.0, abs_tol=1e-3)

                # Coincident origin/target → guarded error, no division by zero.
                degenerate = _tool_payload(await mcp.call_tool(
                    "along_direction",
                    {"origin_x": 1.0, "origin_y": 1.0, "origin_z": 1.0,
                     "target_x": 1.0, "target_y": 1.0, "target_z": 1.0},
                ))
                assert "error" in degenerate

                scaled = _tool_payload(await mcp.call_tool(
                    "scale_value", {"current": 0.4, "factor": 3.0},
                ))
                assert scaled == {"value": 1.2}

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
