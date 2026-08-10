# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for shared voice-worker lifecycle primitives."""
from __future__ import annotations

import asyncio

import pytest
from xr_ai_hub import DataMessage
from xr_ai_voice import TextMessageInput, VadConfig, VoiceSession
from xr_ai_voice import _session as session_module
from xr_ai_voicegate import VoiceGateConfig


class _Endpoint:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.republish_calls = 0

    def on_data(self, callback) -> None:
        self.callback = callback

    async def set_status(self, status: str) -> None:
        self.statuses.append(status)

    async def mark_ready(self) -> None:
        await self.set_status("ready")

    async def republish_statuses(self) -> None:
        self.republish_calls += 1


class _Transport:
    def __init__(self) -> None:
        self.endpoint = _Endpoint()
        self.target_participant = ""
        self.shutdown_called = False
        self.started = asyncio.Event()

    async def wait_until_started(self) -> None:
        await self.started.wait()

    def set_target_participant(self, participant_id: str) -> None:
        self.target_participant = participant_id

    def shutdown(self) -> None:
        self.shutdown_called = True


class _Session:
    def __init__(self) -> None:
        self.transport = _Transport()
        self.queries: list[tuple[str, str, bool, int | None]] = []
        self.is_running = True

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


async def test_data_query_adapter_drops_text_while_session_is_stopped() -> None:
    session = _Session()
    session.is_running = False
    TextMessageInput(session=session)  # type: ignore[arg-type]

    await session.transport.endpoint.callback(DataMessage(
        participant_id="alice",
        topic="",
        pts_us=2,
        data=b"hello",
    ))

    assert session.transport.target_participant == ""
    assert session.queries == []


async def test_voice_session_owns_readiness_ready_file_and_cleanup(
    monkeypatch,
    tmp_path,
) -> None:
    runner_started = asyncio.Event()
    runner_finished = asyncio.Event()

    class _Worker:
        async def cancel(self) -> None:
            runner_finished.set()

    class _Runner:
        async def run(self, _worker) -> None:
            runner_started.set()
            await runner_finished.wait()

    monkeypatch.setattr(
        session_module,
        "_build_voice_pipeline",
        lambda **_kwargs: (object(), _Worker()),
    )
    monkeypatch.setattr(session_module, "PipelineRunner", _Runner)

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

    async def handler(_query) -> str:
        return "unused"

    async with session:
        assert not ready_file.exists()
        run_task = asyncio.create_task(session.run(handler))
        await runner_started.wait()
        await asyncio.sleep(0)
        assert not ready_file.exists()

        transport.started.set()
        while not ready_file.exists():
            await asyncio.sleep(0)

        assert ready_file.exists()
        assert transport.endpoint.statuses == ["ready"]

        runner_finished.set()
        assert await run_task is None

    assert transport.shutdown_called
    assert stt.closed == 1
    assert tts.closed == 1
    assert extra.closed == 1


async def test_voice_session_defers_default_transport_until_services_are_ready(
    monkeypatch,
) -> None:
    transports: list[_Transport] = []

    class ProbeService(_Service):
        async def health(self) -> bool:
            assert transports == []
            return True

    def make_transport() -> _Transport:
        transport = _Transport()
        transports.append(transport)
        return transport

    monkeypatch.setattr(session_module, "HubVoiceTransport", make_transport)
    session = VoiceSession(
        stt=ProbeService(),  # type: ignore[arg-type]
        tts=ProbeService(),  # type: ignore[arg-type]
        vad=VadConfig(),
        voice_gate=VoiceGateConfig(),
    )

    assert transports == []
    async with session:
        assert session.transport is transports[0]

    assert transports[0].shutdown_called


async def test_voice_session_cleans_up_when_readiness_fails(monkeypatch) -> None:
    class FailingService(_Service):
        async def health(self) -> bool:
            raise RuntimeError("unavailable")

    transports: list[_Transport] = []
    monkeypatch.setattr(
        session_module,
        "HubVoiceTransport",
        lambda: transports.append(_Transport()),
    )
    stt = FailingService()
    tts = _Service()
    session = VoiceSession(
        stt=stt,  # type: ignore[arg-type]
        tts=tts,  # type: ignore[arg-type]
        vad=VadConfig(),
        voice_gate=VoiceGateConfig(),
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        async with session:
            pass

    assert transports == []
    assert stt.closed == 1
    assert tts.closed == 1
