# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for video-memory storage, service and native tools."""

import asyncio
import contextlib
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import video_memory_service.__main__ as video_memory_main
from PIL import Image
from video_memory_service.service import VideoMemoryService, select_decoded_frame
from video_memory_service.store import ChunkStore
from xr_ai_hub import (
    FrameData,
    FrameSignal,
    LiveFrameSource,
    ParticipantEvent,
    PixelFormat,
)
from xr_ai_tools.rpc import RPCError, RPCServer
from xr_ai_tools.types import EmptyRequest
from xr_ai_tools.video_memory import (
    HistoricalFrameRequest,
    QueryVideoRequest,
    VideoMemoryTools,
    VideoStatsRequest,
)


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
async def test_list_recorded_participants_tool_schema_is_strict_empty() -> None:
    video = VideoMemoryTools("ipc:///tmp/unused")
    try:
        schema = video.list_recorded_participants.request_model.model_json_schema()
    finally:
        await video.close()

    assert schema.get("properties", {}) == {}
    assert schema.get("additionalProperties") is False


def test_video_entrypoints_use_defaults_when_packaged_config_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(video_memory_main, "_DEFAULT_CONFIG", missing)

    assert video_memory_main._load_config(None) == {}
    with pytest.raises(SystemExit, match="config file not found"):
        video_memory_main._load_config(missing)


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

    await endpoint.send(
        FrameSignal(
            slot=0,
            seq=2,
            pts_us=now_us,
            width=1,
            height=1,
            fmt=PixelFormat.RGB24,
            data_sz=3,
            participant_id="departed-user",
            track_id="camera",
        )
    )

    assert await waiter == frame
    assert source._events == {}


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

    async with _running_server(endpoint, service.dispatch):
        video = VideoMemoryTools(endpoint)
        try:
            recorded = await video.list_recorded_participants.execute(EmptyRequest())
            stats = await video.get_video_stats.execute(
                VideoStatsRequest(participant_id="user/name")
            )
            clip = await video.query_video.execute(
                QueryVideoRequest(
                    participant_id="user/name",
                    start_us=1_100_000,
                    end_us=2_100_000,
                )
            )
            health = await video.get_health()
        finally:
            await video.close()

    assert recorded.participants == ["user/name"]
    assert stats.num_chunks == 2
    assert Path(clip.path).read_bytes() == b"firstsecond"
    assert health.ready is True
    with pytest.raises(RPCError, match="unknown operation"):
        await service.dispatch("list_live_participants", {})


@pytest.mark.asyncio
async def test_video_memory_service_validation_and_disabled_mode(
    tmp_path: Path,
) -> None:
    service = VideoMemoryService(store=None, out_dir=tmp_path / "output", gpu_id=0)

    assert await service.dispatch("get_health", {}) == {
        "ready": True,
        "recording_enabled": False,
    }
    with pytest.raises(RPCError) as invalid:
        await service.dispatch("get_health", {"unexpected": True})
    with pytest.raises(RPCError) as disabled:
        await service.dispatch("get_video_stats", {"participant_id": "alice"})

    assert invalid.value.code == "invalid_request"
    assert disabled.value.code == "recording_disabled"


@pytest.mark.asyncio
async def test_recorded_frame_decodes_and_exports_png_through_native_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk = tmp_path / "chunk.264"
    chunk.write_bytes(b"h264")
    store = ChunkStore(tmp_path / "recordings")
    monkeypatch.setattr(
        store,
        "frame_chunk",
        lambda _participant_id, _target_us: (
            chunk,
            {
                "start_us": 1_000_000,
                "end_us": 1_000_000,
                "num_frames": 1,
                "width": 2,
                "height": 2,
            },
        ),
    )
    nv12 = np.array([[16, 16], [16, 16], [128, 128]], dtype=np.uint8)
    monkeypatch.setattr(
        "video_memory_service.service.decode_h264",
        lambda _data, _gpu_id: [nv12],
    )
    service = VideoMemoryService(store=store, out_dir=tmp_path / "output", gpu_id=0)
    endpoint = f"ipc:///tmp/video-{uuid.uuid4().hex}"

    async with _running_server(endpoint, service.dispatch):
        video = VideoMemoryTools(endpoint)
        try:
            result = await video.get_frame_from_time.execute(
                HistoricalFrameRequest(
                    participant_id="alice",
                    second_ago=1,
                    reference_time_us=2_000_000,
                )
            )
        finally:
            await video.close()

    with Image.open(result.path) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)
    assert result.timestamp_us == 1_000_000
    assert result.actual_second_ago == 1.0


@pytest.mark.asyncio
async def test_video_memory_process_touches_ready_file_after_rpc_bind(
    tmp_path: Path,
) -> None:
    endpoint = f"ipc://{tmp_path / (uuid.uuid4().hex + '.sock')}"
    ready_file = tmp_path / "video.ready"
    task = asyncio.create_task(
        video_memory_main._serve(
            {"endpoint": endpoint, "out_dir": str(tmp_path / "output")},
            ready_file,
        )
    )
    try:
        for _ in range(100):
            if ready_file.exists():
                break
            await asyncio.sleep(0.01)
        assert ready_file.exists()
        video = VideoMemoryTools(endpoint)
        try:
            health = await video.get_health()
        finally:
            await video.close()
        assert health.ready is True
        assert health.recording_enabled is False
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


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
