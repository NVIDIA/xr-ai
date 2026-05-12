# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI-viable integration test for the transcript-mcp server.

Spawns ``python -m transcript_mcp_server`` as a subprocess against a
temporary transcripts directory, polls readiness via ``McpClient.list_tools``
(same pattern as production ``mcp_probe``), then drives the four public
FastMCP tools end-to-end over StreamableHTTP and verifies the on-disk
JSONL artefacts. No GPU, Docker, or model weights.

Note: the ``--ready-file`` codepath in ``__main__.py`` calls
``app.add_event_handler`` which Starlette 1.0 removed, so this test polls
the MCP surface for readiness instead. That bug is out of scope here.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml
from fastmcp import Client as McpClient

pytestmark = pytest.mark.asyncio


# ── helpers ──────────────────────────────────────────────────────────────────

def _free_port() -> int:
    """Pick a free TCP port; binding to 0 lets the kernel pick atomically."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
            raise RuntimeError(
                f"transcript_mcp_server exited early (rc={proc.returncode})"
            )
        try:
            async with McpClient(url) as mcp:
                await mcp.list_tools()
                return
        except Exception:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"transcript_mcp_server at {url} not ready within {timeout}s")


# ── test ─────────────────────────────────────────────────────────────────────

async def test_transcript_mcp_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        tmp           = Path(td)
        transcripts   = tmp / "transcripts"
        config_path   = tmp / "transcript_mcp_server.yaml"
        port          = _free_port()
        url           = f"http://127.0.0.1:{port}/mcp"

        config_path.write_text(yaml.safe_dump({
            "transcripts_dir": str(transcripts),
            "host":            "127.0.0.1",
            "port":            port,
        }))

        # Redirect loguru's on-disk sink under the tmpdir so the test
        # doesn't write into the developer's $HOME log root.
        env = {**os.environ, "XR_AI_LOG_ROOT": str(tmp / "logs")}

        proc = subprocess.Popen(
            [sys.executable, "-m", "transcript_mcp_server",
             "--config", str(config_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        try:
            await _wait_ready(url, proc, timeout=15.0)

            async with McpClient(url) as mcp:
                tools = {t.name for t in await mcp.list_tools()}
                assert tools == {
                    "query_transcripts", "add_transcript",
                    "list_sources",      "get_transcript_stats",
                }

                # Two utterances from one source plus one from another;
                # the empty-text guard must produce an error dict.
                add_alice_1 = _tool_payload(await mcp.call_tool(
                    "add_transcript",
                    {"source_id": "alice@home", "timestamp_us": 1_000_000, "text": "hello"},
                ))
                add_alice_2 = _tool_payload(await mcp.call_tool(
                    "add_transcript",
                    {"source_id": "alice@home", "timestamp_us": 2_000_000, "text": "world"},
                ))
                add_bob = _tool_payload(await mcp.call_tool(
                    "add_transcript",
                    {"source_id": "agent-vlm", "timestamp_us": 3_000_000, "text": "frame seen"},
                ))
                add_empty = _tool_payload(await mcp.call_tool(
                    "add_transcript",
                    {"source_id": "alice@home", "timestamp_us": 4_000_000, "text": "   "},
                ))
                assert add_alice_1 == {"ok": True}
                assert add_alice_2 == {"ok": True}
                assert add_bob     == {"ok": True}
                assert "error" in add_empty

                sources = _tool_payload(await mcp.call_tool("list_sources", {}))
                assert set(sources) == {"alice@home", "agent-vlm"}

                # Inclusive window covering both alice utterances.
                q = _tool_payload(await mcp.call_tool(
                    "query_transcripts",
                    {"source_id": "alice@home", "start_us": 0, "end_us": 5_000_000},
                ))
                assert [r["text"] for r in q] == ["hello", "world"]
                assert [r["timestamp_us"] for r in q] == [1_000_000, 2_000_000]

                # Half-open window — only the later utterance lands inside.
                q_late = _tool_payload(await mcp.call_tool(
                    "query_transcripts",
                    {"source_id": "alice@home", "start_us": 1_500_000, "end_us": 5_000_000},
                ))
                assert [r["text"] for r in q_late] == ["world"]

                stats = _tool_payload(await mcp.call_tool(
                    "get_transcript_stats", {"source_id": "alice@home"},
                ))
                assert stats["source_id"]   == "alice@home"
                assert stats["count"]       == 2
                assert stats["total_chars"] == len("hello") + len("world")
                assert stats["earliest_us"] == 1_000_000
                assert stats["latest_us"]   == 2_000_000

                missing_stats = _tool_payload(await mcp.call_tool(
                    "get_transcript_stats", {"source_id": "never-seen"},
                ))
                assert "error" in missing_stats

            # On-disk artefacts — _safe_name turns '@' into '_', so the
            # alice file is sanitized while agent-vlm passes through.
            alice_jsonl = transcripts / "alice_home.jsonl"
            bob_jsonl   = transcripts / "agent-vlm.jsonl"
            assert alice_jsonl.exists()
            assert bob_jsonl.exists()
            alice_lines = [json.loads(ln) for ln in alice_jsonl.read_text().splitlines() if ln]
            assert alice_lines == [
                {"timestamp_us": 1_000_000, "text": "hello"},
                {"timestamp_us": 2_000_000, "text": "world"},
            ]
            # Sidecar preserves the raw source_id with the unsanitized '@'.
            assert (transcripts / "alice_home.identity").read_text() == "alice@home"

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
