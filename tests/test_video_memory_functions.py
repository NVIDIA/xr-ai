# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for video-memory storage, service, and NAT functions."""

import asyncio
import contextlib
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client as McpClient
from nat.builder.workflow_builder import WorkflowBuilder
from video_mcp_server.__main__ import _recording_enabled, build_mcp
from video_mcp_server.live import _frame_to_rgb
from video_memory_service.service import VideoMemoryService, select_decoded_frame
from video_memory_service.store import ChunkStore
from xr_ai_agent import (
    FrameData,
    FrameSignal,
    LiveFrameSource,
    ParticipantEvent,
    PixelFormat,
)
from xr_ai_nat.functions._rpc import RPCError, RPCServer
from xr_ai_nat.functions.video_memory import VideoMemoryFunctionsConfig
from xr_ai_nat.functions.video_memory.schemas import HistoricalFrameRequest


class _LiveFrames:
    def participants(self) -> list[str]:
        return ["connected-user"]

    async def get_latest(self, participant_id: str) -> dict:
        return {
            "path": f"/tmp/{participant_id}.png",
            "width": 1,
            "height": 1,
            "timestamp_us": 1,
        }


class _UnusedClient:
    pass


class _UnavailableRecordedClient:
    async def list_recorded_participants(self):
        raise RPCError("video service unavailable", code="connection_error")


class _UnavailableStartupClient:
    async def get_health(self):
        raise RPCError("video service unavailable", code="connection_error")


class _FrameEndpoint:
    def __init__(self, frame: FrameData) -> None:
        self._frame = frame
        self._callbacks = []
        self._participant_callbacks = []

    def on_frame(self, callback) -> None:
        self._callbacks.append(callback)

    def on_participant(self, callback) -> None:
        self._participant_callbacks.append(callback)

    async def request_frame(self, _signal: FrameSignal) -> FrameData:
        return self._frame

    async def send(self, signal: FrameSignal) -> None:
        for callback in self._callbacks:
            await callback(signal)

    async def send_participant(self, event: ParticipantEvent) -> None:
        for callback in self._participant_callbacks:
            await callback(event)


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
    (tmp_path / "safe-user" / "interrupted-upload.264").write_bytes(b"ignore")
    store = ChunkStore(tmp_path)

    assert store.participants() == ["user/name"]
    assert [path.name for path, _metadata in store.chunks("user/name")] == [
        "1000000.264",
        "2000000.264",
    ]
    assert store.stats("user/name")["total_bytes"] == len(b"firstsecond")
    assert store.query("user/name", 1_100_000, 2_100_000) == b"firstsecond"
    assert store.frame_chunk("user/name", 2_200_000)[0].name == "2000000.264"


def test_chunk_store_path_escape_has_a_stable_rpc_error(tmp_path: Path) -> None:
    store = ChunkStore(tmp_path / "recordings")

    with pytest.raises(RPCError) as error:
        store._check(tmp_path / "outside")

    assert error.value.code == "path_escape"


def test_chunk_store_does_not_follow_identity_or_directory_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".identity").write_text("outside-user", encoding="utf-8")
    (root / "outside-link").symlink_to(outside, target_is_directory=True)
    store = ChunkStore(root)

    with pytest.raises(RPCError) as directory_error:
        store.chunks("outside-user")

    assert directory_error.value.code == "path_escape"
    (root / "outside-link").unlink()
    participant = root / "recorded-user"
    participant.mkdir()
    (participant / ".identity").symlink_to(outside / ".identity")

    with pytest.raises(RPCError) as identity_error:
        store.participants()

    assert identity_error.value.code == "path_escape"


@pytest.mark.asyncio
async def test_video_mcp_preserves_conditional_tool_sets() -> None:
    live_only = build_mcp(_UnusedClient(), _LiveFrames(), recording_enabled=False)
    recorded = build_mcp(_UnusedClient(), _LiveFrames(), recording_enabled=True)

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
async def test_video_mcp_recorded_discovery_reports_service_failures() -> None:
    mcp = build_mcp(_UnavailableRecordedClient(), _LiveFrames(), recording_enabled=True)

    async with McpClient(mcp) as client:
        result = await client.call_tool("list_recorded_participants", {})

    assert result.data == {"error": "video service unavailable"}


@pytest.mark.asyncio
async def test_video_mcp_starts_live_only_when_recorded_service_is_unavailable() -> None:
    assert await _recording_enabled(_UnavailableStartupClient()) is False


@pytest.mark.asyncio
async def test_live_frame_source_stays_with_the_calling_process() -> None:
    now_us = time.time_ns() // 1_000
    frame = FrameData(
        seq=1,
        pts_us=now_us,
        width=1,
        height=1,
        fmt=PixelFormat.RGB24,
        data=b"\x00\x00\x00",
        participant_id="live-user",
        track_id="camera",
    )
    endpoint = _FrameEndpoint(frame)
    source = LiveFrameSource(endpoint)
    await endpoint.send(
        FrameSignal(
            slot=0,
            seq=1,
            pts_us=now_us,
            width=1,
            height=1,
            fmt=PixelFormat.RGB24,
            data_sz=3,
            participant_id="live-user",
            track_id="camera",
        )
    )

    assert source.participants() == ["live-user"]
    assert await source.get("live-user") == frame


@pytest.mark.asyncio
async def test_live_frame_source_releases_departed_participants() -> None:
    now_us = time.time_ns() // 1_000
    frame = FrameData(
        seq=1,
        pts_us=now_us - 10_000_000,
        width=1,
        height=1,
        fmt=PixelFormat.RGB24,
        data=b"\x00\x00\x00",
        participant_id="departed-user",
        track_id="camera",
    )
    endpoint = _FrameEndpoint(frame)
    source = LiveFrameSource(endpoint)
    await endpoint.send(
        FrameSignal(
            slot=0,
            seq=1,
            pts_us=frame.pts_us,
            width=1,
            height=1,
            fmt=PixelFormat.RGB24,
            data_sz=3,
            participant_id="departed-user",
            track_id="camera",
        )
    )
    waiter = asyncio.create_task(source.get("departed-user"))
    await asyncio.sleep(0)

    assert source._latest
    assert "departed-user" in source._events

    await endpoint.send_participant(
        ParticipantEvent(participant_id="departed-user", joined=False, pts_us=now_us)
    )

    assert source._latest == {}
    assert source._events == {}

    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter


def test_live_png_export_converts_nv12_planes() -> None:
    frame = FrameData(
        seq=1,
        pts_us=1,
        width=2,
        height=2,
        fmt=PixelFormat.NV12,
        data=bytes([16, 16, 16, 16, 128, 128]),
    )

    rgb = _frame_to_rgb(frame)

    assert rgb.shape == (2, 2, 3)


@pytest.mark.asyncio
async def test_video_memory_functions_call_typed_service(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    _recording(recordings, "user/name")
    service = VideoMemoryService(
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
        recorded = await functions["video__list_recorded_participants"].ainvoke({})
        stats = await functions["video__get_video_stats"].ainvoke(
            {"participant_id": "user/name"}
        )
        clip = await functions["video__query_video"].ainvoke(
            {"participant_id": "user/name", "start_us": 1_100_000, "end_us": 2_100_000}
        )

    assert recorded.participants == ["user/name"]
    assert stats.num_chunks == 2
    assert Path(clip.path).read_bytes() == b"firstsecond"
    with pytest.raises(RPCError, match="unknown operation"):
        await service.dispatch("list_live_participants", {})


@pytest.mark.asyncio
async def test_recorded_frame_reports_frame_export_errors(tmp_path: Path, monkeypatch) -> None:
    chunk = tmp_path / "chunk.264"
    chunk.write_bytes(b"h264")
    store = ChunkStore(tmp_path / "recordings")
    monkeypatch.setattr(
        store,
        "frame_chunk",
        lambda _participant_id, _target_us: (
            chunk,
            {"start_us": 1, "end_us": 1, "num_frames": 1},
        ),
    )
    service = VideoMemoryService(store=store, out_dir=tmp_path / "output", gpu_id=0)

    async def run_sync(function, *args):
        return function(*args)

    monkeypatch.setattr("video_memory_service.service.asyncio.to_thread", run_sync)
    monkeypatch.setattr(
        "video_memory_service.service.decode_h264",
        lambda _data, _gpu_id: [SimpleNamespace(shape=(3, 2))],
    )

    def fail_export(*_args) -> None:
        raise OSError("invalid NV12 frame")

    monkeypatch.setattr("video_memory_service.service.nv12_to_rgb", fail_export)

    with pytest.raises(RPCError) as error:
        await service.dispatch(
            "get_frame_from_time",
            {"participant_id": "user", "reference_time_us": 1},
        )

    assert error.value.code == "frame_export_error"


def test_historical_frame_schema_requires_an_absolute_reference() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        HistoricalFrameRequest(participant_id="user", reference_time_us=0)

    schema = HistoricalFrameRequest.model_json_schema()
    assert "Unix-epoch timestamp" in schema["properties"]["reference_time_us"]["description"]
    assert "Whole seconds" in schema["properties"]["second_ago"]["description"]


@pytest.mark.parametrize(
    ("target_us", "declared_frames", "decoded_frames", "expected"),
    [
        (900, 4, 4, (0, 1_000)),
        (5_000, 4, 4, (3, 4_000)),
        (2_500, 1, 1, (0, 1_000)),
        (4_000, 4, 2, (1, 2_000)),
    ],
)
def test_select_decoded_frame_clamps_to_recorded_boundaries(
    target_us: int,
    declared_frames: int,
    decoded_frames: int,
    expected: tuple[int, int],
) -> None:
    assert select_decoded_frame(
        start_us=1_000,
        end_us=4_000,
        declared_frames=declared_frames,
        decoded_frames=decoded_frames,
        target_us=target_us,
    ) == expected
