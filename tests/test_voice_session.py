# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for shared voice-worker lifecycle primitives."""
from __future__ import annotations

import asyncio

import pytest
from xr_ai_voice import VadConfig, VoiceSession
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


class _HandlerProcessor:
    def __init__(self) -> None:
        self.responses: list[tuple[str, object, bool, int | None]] = []

    async def enqueue_response(
        self,
        participant_id: str,
        response: object,
        *,
        interrupt: bool = False,
        pts_us: int | None = None,
    ) -> None:
        self.responses.append((participant_id, response, interrupt, pts_us))


class _Service:
    def __init__(self) -> None:
        self.closed = 0

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed += 1


class _PipelineTask:
    def __init__(self, on_cancel=None) -> None:
        self.cancel_calls = 0
        self._on_cancel = on_cancel

    async def cancel(self) -> None:
        self.cancel_calls += 1
        if self._on_cancel is not None:
            self._on_cancel()


async def _ignore_query(_query) -> None:
    pass


async def test_voice_session_queues_external_responses_through_active_processor() -> None:
    service = _Service()
    session = VoiceSession(
        stt=service,  # type: ignore[arg-type]
        tts=service,  # type: ignore[arg-type]
        vad=VadConfig(),
        voice_gate=VoiceGateConfig(),
    )
    processor = _HandlerProcessor()
    session._io_processor = processor  # type: ignore[assignment]  # noqa: SLF001

    await session.enqueue_response(
        "alice",
        "Careful.",
        interrupt=True,
        pts_us=12,
    )

    assert processor.responses == [("alice", "Careful.", True, 12)]


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

    async def input_sink(_query) -> None:
        pass

    async with session:
        assert not ready_file.exists()
        run_task = asyncio.create_task(session.run(input_sink))
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


async def test_voice_session_reannounces_current_status(monkeypatch) -> None:
    runner_started = asyncio.Event()
    runner_finished = asyncio.Event()

    class _Runner:
        async def run(self, _worker) -> None:
            runner_started.set()
            await runner_finished.wait()

    worker = _PipelineTask(on_cancel=runner_finished.set)
    monkeypatch.setattr(
        session_module,
        "_build_voice_pipeline",
        lambda **_kwargs: (object(), worker),
    )
    monkeypatch.setattr(session_module, "PipelineRunner", _Runner)
    monkeypatch.setattr(session_module, "_STATUS_REANNOUNCE_INTERVAL_S", 0.01)

    service = _Service()
    transport = _Transport()
    session = VoiceSession(
        stt=service,  # type: ignore[arg-type]
        tts=service,  # type: ignore[arg-type]
        vad=VadConfig(),
        voice_gate=VoiceGateConfig(),
        transport=transport,  # type: ignore[arg-type]
    )

    async with session:
        run_task = asyncio.create_task(session.run(_ignore_query))
        await runner_started.wait()
        transport.started.set()
        for _ in range(20):
            if transport.endpoint.republish_calls:
                break
            await asyncio.sleep(0.01)

        assert transport.endpoint.statuses == ["ready"]
        assert transport.endpoint.republish_calls > 0

        runner_finished.set()
        assert await run_task is None


async def test_voice_session_allows_early_clean_pipeline_exit(monkeypatch) -> None:
    class _Runner:
        async def run(self, _worker) -> None:
            return

    worker = _PipelineTask()
    monkeypatch.setattr(
        session_module,
        "_build_voice_pipeline",
        lambda **_kwargs: (object(), worker),
    )
    monkeypatch.setattr(session_module, "PipelineRunner", _Runner)

    service = _Service()
    transport = _Transport()
    session = VoiceSession(
        stt=service,  # type: ignore[arg-type]
        tts=service,  # type: ignore[arg-type]
        vad=VadConfig(),
        voice_gate=VoiceGateConfig(),
        transport=transport,  # type: ignore[arg-type]
    )

    async with session:
        assert await session.run(_ignore_query) is None

    assert worker.cancel_calls == 0
    assert transport.endpoint.statuses == []


async def test_voice_session_propagates_pipeline_runner_error(monkeypatch) -> None:
    class _Runner:
        async def run(self, _worker) -> None:
            raise ValueError("runner failure")

    worker = _PipelineTask()
    monkeypatch.setattr(
        session_module,
        "_build_voice_pipeline",
        lambda **_kwargs: (object(), worker),
    )
    monkeypatch.setattr(session_module, "PipelineRunner", _Runner)

    service = _Service()
    session = VoiceSession(
        stt=service,  # type: ignore[arg-type]
        tts=service,  # type: ignore[arg-type]
        vad=VadConfig(),
        voice_gate=VoiceGateConfig(),
        transport=_Transport(),  # type: ignore[arg-type]
    )

    async with session:
        with pytest.raises(ValueError, match="runner failure"):
            await session.run(_ignore_query)

    assert worker.cancel_calls == 0


async def test_voice_session_cancels_pipeline_when_ready_file_touch_fails(
    monkeypatch,
) -> None:
    runner_started = asyncio.Event()
    runner_finished = asyncio.Event()

    class _Runner:
        async def run(self, _worker) -> None:
            runner_started.set()
            await runner_finished.wait()

    class _FailingReadyFile:
        def touch(self) -> None:
            raise OSError("ready file unavailable")

    worker = _PipelineTask(on_cancel=runner_finished.set)
    monkeypatch.setattr(
        session_module,
        "_build_voice_pipeline",
        lambda **_kwargs: (object(), worker),
    )
    monkeypatch.setattr(session_module, "PipelineRunner", _Runner)

    service = _Service()
    transport = _Transport()
    session = VoiceSession(
        stt=service,  # type: ignore[arg-type]
        tts=service,  # type: ignore[arg-type]
        vad=VadConfig(),
        voice_gate=VoiceGateConfig(),
        ready_file=_FailingReadyFile(),  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
    )

    async with session:
        run_task = asyncio.create_task(session.run(_ignore_query))
        await runner_started.wait()
        transport.started.set()
        with pytest.raises(OSError, match="ready file unavailable"):
            await run_task

    assert worker.cancel_calls == 1
    assert transport.endpoint.statuses == []


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
    with pytest.raises(RuntimeError, match="not ready"):
        _ = session.endpoint
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
    await session.close()

    assert transports == []
    assert stt.closed == 1
    assert tts.closed == 1
