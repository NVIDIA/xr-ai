# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for video-memory storage, service, and NAT functions."""

import asyncio
import contextlib
import json
import uuid
from pathlib import Path

import pytest
from fastmcp import Client as McpClient
from nat.builder.workflow_builder import WorkflowBuilder
from video_mcp_server.__main__ import build_mcp
from video_memory_service.service import VideoMemoryService
from video_memory_service.store import ChunkStore
from xr_ai_nat.functions._rpc import RPCServer
from xr_ai_nat.functions.video_memory import VideoMemoryFunctionsConfig


class _NoLiveFrames:
    def participants(self) -> list[str]:
        return ["connected-user"]

    async def fetch_latest(self, _participant_id: str):
        return None


class _UnusedClient:
    pass


def _recording(root: Path, participant_id: str) -> None:
    directory = root / "safe-user"
    directory.mkdir(parents=True)
    (directory / ".identity").write_text(participant_id, encoding="utf-8")
    first = b"first"
    second = b"second"
    for timestamp, data in ((1_000_000, first), (2_000_000, second)):
        (directory / f"{timestamp}.264").write_bytes(data)
        (directory / f"{timestamp}.json").write_text(
            json.dumps(
                {
                    "start_us": timestamp,
                    "end_us": timestamp + 500_000,
                    "size_bytes": len(data),
                }
            ),
            encoding="utf-8",
        )


@contextlib.asynccontextmanager
async def _running_server(endpoint: str, dispatch):
    server = RPCServer(endpoint, dispatch)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.02)
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_chunk_store_preserves_identities_and_windows(tmp_path: Path) -> None:
    _recording(tmp_path, "user/name")
    store = ChunkStore(tmp_path)

    assert store.participants() == ["user/name"]
    assert store.stats("user/name")["total_bytes"] == len(b"firstsecond")
    assert store.query("user/name", 1_100_000, 2_100_000) == b"firstsecond"
    assert store.frame_chunk("user/name", 2_200_000)[0].name == "2000000.264"


@pytest.mark.asyncio
async def test_video_mcp_preserves_conditional_tool_sets() -> None:
    live_only = build_mcp(_UnusedClient(), recording_enabled=False)
    recorded = build_mcp(_UnusedClient(), recording_enabled=True)

    async with McpClient(live_only) as client:
        live_names = {tool.name for tool in await client.list_tools()}
    async with McpClient(recorded) as client:
        recorded_names = {tool.name for tool in await client.list_tools()}

    assert live_names == {
        "get_frame_from_time",
        "get_latest_frame",
        "list_live_participants",
    }
    assert recorded_names == {
        "get_frame_from_time",
        "get_video_stats",
        "list_live_participants",
        "list_recorded_participants",
        "query_video",
    }


@pytest.mark.asyncio
async def test_video_memory_functions_call_typed_service(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    _recording(recordings, "user/name")
    service = VideoMemoryService(
        provider=_NoLiveFrames(),
        store=ChunkStore(recordings),
        out_dir=tmp_path / "output",
        gpu_id=0,
    )
    endpoint = f"ipc:///tmp/video-{uuid.uuid4().hex}"

    async with _running_server(endpoint, service.dispatch), WorkflowBuilder() as builder:
        await builder.add_function_group(
            "video",
            VideoMemoryFunctionsConfig(endpoint=endpoint),
        )
        group = await builder.get_function_group("video")
        functions = await group.get_all_functions()
        live = await functions["video__list_live_participants"].ainvoke({})
        recorded = await functions["video__list_recorded_participants"].ainvoke({})
        stats = await functions["video__get_video_stats"].ainvoke(
            {"participant_id": "user/name"}
        )
        clip = await functions["video__query_video"].ainvoke(
            {"participant_id": "user/name", "start_us": 1_100_000, "end_us": 2_100_000}
        )

    assert live == ["connected-user"]
    assert recorded == ["user/name"]
    assert stats.num_chunks == 2
    assert Path(clip.path).read_bytes() == b"firstsecond"
