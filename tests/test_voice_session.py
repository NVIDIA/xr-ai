# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for shared voice-worker lifecycle primitives."""
from __future__ import annotations

from xr_ai_hub import DataMessage
from xr_ai_voice import TextMessageInput, VadConfig, VoiceSession
from xr_ai_voicegate import VoiceGateConfig


class _Endpoint:
    def on_data(self, callback) -> None:
        self.callback = callback


class _Transport:
    def __init__(self) -> None:
        self.endpoint = _Endpoint()
        self.target_participant = ""
        self.shutdown_called = False

    def set_target_participant(self, participant_id: str) -> None:
        self.target_participant = participant_id

    def shutdown(self) -> None:
        self.shutdown_called = True


class _Session:
    def __init__(self) -> None:
        self.transport = _Transport()
        self.queries: list[tuple[str, str, bool, int | None]] = []

    async def enqueue_query(
        self,
        participant_id: str,
        text: str,
        *,
        fresh_match: bool = False,
        pts_us: int | None = None,
    ) -> None:
        self.queries.append((participant_id, text, fresh_match, pts_us))


class _Service:
    def __init__(self) -> None:
        self.closed = 0

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed += 1


async def test_data_query_adapter_routes_text_and_ignores_control_topics() -> None:
    session = _Session()
    TextMessageInput(
        session=session,  # type: ignore[arg-type]
        ignore_topics={"control"},
        transform=str.upper,
        fresh_match=True,
    )

    await session.transport.endpoint.callback(DataMessage(
        participant_id="alice",
        topic="control",
        pts_us=1,
        data=b"ignored",
    ))
    await session.transport.endpoint.callback(DataMessage(
        participant_id="alice",
        topic="",
        pts_us=2,
        data=b"hello",
    ))

    assert session.transport.target_participant == "alice"
    assert session.queries == [("alice", "HELLO", True, 2)]


async def test_voice_session_owns_readiness_ready_file_and_cleanup(tmp_path) -> None:
    stt = _Service()
    tts = _Service()
    extra = _Service()
    transport = _Transport()
    ready_file = tmp_path / "ready"
    session = VoiceSession(
        stt=stt,  # type: ignore[arg-type]
        tts=tts,  # type: ignore[arg-type]
        vad=VadConfig(),
        voice_gate=VoiceGateConfig(),
        probes={"extra": extra.health},
        ready_file=ready_file,
        closeables=(extra, extra),
        transport=transport,  # type: ignore[arg-type]
    )

    async with session:
        assert ready_file.exists()

    assert transport.shutdown_called
    assert stt.closed == 1
    assert tts.closed == 1
    assert extra.closed == 1
