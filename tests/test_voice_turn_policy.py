# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic admission uses selected partials and rejects stale lifecycle work."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import InterruptionFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from xr_ai_interaction import Interrupted, UserQuery
from xr_ai_models import ModelsConfig
from xr_ai_runtime import Topic
from xr_ai_voice import TranscriptUpdate, VadConfig, VoiceAgent, VoiceOutputState, VoiceTurnAction
from xr_ai_voice._frames import (
    GatedQueryFrame,
    _TurnInterruptFrame,
    _TurnResetFrame,
    _TurnTranscriptFrame,
)
from xr_ai_voice._processors.io import _VoiceIOProcessor
from xr_ai_voice._processors.speaker_stt import _SpeakerSttProcessor
from xr_ai_voice._processors.turn_policy import _TurnPolicyProcessor
from xr_ai_voice._processors.voice_gate import VoiceGateProcessor
from xr_ai_voicegate import load_voice_gate_config

_CFG = Path(__file__).resolve().parents[1] / "agent-samples/conversation-example/yaml/voice_gate.yaml"


class _Policy:
    settle_s = 0

    def __init__(self):
        self.latest = {}
        self.entered = asyncio.Event()
        self.release = None
        self.calls = 0
        self.states = []
        self.finished = asyncio.Event()
        self.admitted = set()

    def reset(self, pid):
        self.latest.pop(pid, None)

    def update(self, update):
        self.latest[update.participant_id] = update

    def output(self, state):
        self.states.append(state)

    async def decide(self, pid):
        self.calls += 1
        update = self.latest[pid]
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        self.finished.set()
        if (pid, update.text) in self.admitted:
            return []
        self.admitted.add((pid, update.text))
        return [VoiceTurnAction("submit", "request-1", update.text, update.timestamp_us)]


@pytest.mark.asyncio
async def test_selected_partial_dispatches_before_any_stop_or_final_without_vad():
    cfg = load_voice_gate_config(_CFG)
    speaker = _SpeakerSttProcessor(cfg=cfg._speaker, transcript_updates=True)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._semantic_turns = True
    policy = _Policy()
    admission = _TurnPolicyProcessor(policy, authorize=gate._accepts_turn_snapshot)
    seen = []
    submitted = asyncio.Event()

    async def capture(frame, *args):
        seen.append(frame)
        if isinstance(frame, GatedQueryFrame):
            submitted.set()

    async def to_gate(frame, *args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def to_admission(frame, *args):
        await admission.process_frame(frame, FrameDirection.DOWNSTREAM)

    speaker.push_frame = to_gate
    gate.push_frame = to_admission
    admission.push_frame = capture
    admission._worker = asyncio.create_task(admission._run())
    try:
        # Unenrolled text cannot authorize semantic admission.
        await speaker._event("wearer", {"kind": "partial", "text": "unselected words"})
        assert not policy.latest
        await speaker._event("wearer", {"kind": "enrolled"})
        speaker._audio_pts["wearer"] = 123_000
        await speaker._event("wearer", {"kind": "speech_start"})
        await speaker._event("wearer", {"kind": "partial", "text": "Show the pressure"})
        await asyncio.wait_for(submitted.wait(), 1)
        queries = [f for f in seen if isinstance(f, GatedQueryFrame)]
        assert [f.text for f in queries] == ["Show the pressure"]
        assert queries[0].request_id == "request-1"
        assert policy.latest["wearer"].final is False
        assert not any(isinstance(f, UserStartedSpeakingFrame) for f in seen)
        assert not any(isinstance(f, InterruptionFrame) for f in seen)
        # Existing ASR silence-finalized transcript is not a second query.
        policy.finished.clear()
        await speaker._event(
            "wearer",
            {
                "kind": "transcript",
                "text": "Show the pressure",
                "pts_us": 123_000,
                "speaker_id": 0,
            },
        )
        await asyncio.wait_for(policy.finished.wait(), 1)
        assert len([f for f in seen if isinstance(f, GatedQueryFrame)]) == 1
    finally:
        await admission.cleanup()
        await speaker.cleanup()
        await gate.cleanup()


@pytest.mark.asyncio
async def test_reset_rejects_inflight_result_even_if_source_epoch_is_reused():
    policy = _Policy()
    policy.release = asyncio.Event()
    admission = _TurnPolicyProcessor(policy)
    admission.push_frame = AsyncMock()
    admission._worker = asyncio.create_task(admission._run())
    update = TranscriptUpdate("a", "source", 0, 0, "Old words", 100)
    try:
        await admission.process_frame(_TurnTranscriptFrame(update), FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(policy.entered.wait(), 1)
        await admission.process_frame(_TurnResetFrame("a"), FrameDirection.DOWNSTREAM)
        # Recreate the epoch without scheduling new inference. The old policy
        # deliberately returns stale actions; SDK lifecycle identity rejects it.
        admission._epochs["a"] = "source"
        admission._identities["a"] = object()
        policy.release.set()
        await asyncio.wait_for(policy.finished.wait(), 1)
        assert not any(isinstance(c.args[0], GatedQueryFrame) for c in admission.push_frame.call_args_list)
    finally:
        await admission.cleanup()


@pytest.mark.asyncio
async def test_output_notifications_are_correlated_and_silence_does_not_poll():
    policy = _Policy()
    admission = _TurnPolicyProcessor(policy)
    admission._epochs["a"] = "source"
    admission._requests["a"] = "active"
    admission.output(VoiceOutputState("a", "stale", "done"))
    assert policy.states == []
    admission.output(VoiceOutputState("a", "active", "committed"))
    assert policy.states == [VoiceOutputState("a", "active", "committed")]
    assert "a" in admission._dirty
    await admission.cleanup()


@pytest.mark.parametrize(
    "args",
    [
        ("bad", "id", "text"),
        ("submit", "", "text"),
        ("submit", "id", ""),
        ("interrupt", "id", "text"),
    ],
)
def test_invalid_policy_actions_cannot_be_admitted(args):
    with pytest.raises(ValueError):
        VoiceTurnAction(*args)


def test_policy_factory_rejects_unsupported_input_before_allocating_batch_client(monkeypatch, tmp_path):
    from xr_ai_voice import _runtime

    path = tmp_path / "voice.yaml"
    path.write_text(_CFG.read_text().replace("backend: required", "backend: auto"))
    cfg = load_voice_gate_config(path)
    allocated = []
    monkeypatch.setattr(_runtime, "make_stt", lambda *args: allocated.append(args))
    with pytest.raises(ValueError, match="required speaker ASR"):
        VoiceAgent.from_models(
            models=ModelsConfig({}),
            query_topic=Topic("semantic.query", UserQuery),
            interrupted_topic=Topic("semantic.interrupted", Interrupted),
            turn_policy=_Policy(),
            tts=SimpleNamespace(),
            vad=VadConfig(),
            voice_gate=cfg,
        )
    assert allocated == []


def test_policy_identity_preserves_ordinary_serialized_queries():
    assert UserQuery(text="hello", timestamp_us=100).model_dump() == {
        "text": "hello",
        "timestamp_us": 100,
    }
    assert UserQuery(text="hello", timestamp_us=100, request_id="policy:1").model_dump()["request_id"] == "policy:1"


@pytest.mark.asyncio
async def test_withheld_control_prefix_is_restored_before_semantic_admission():
    cfg = load_voice_gate_config(_CFG)
    speaker = _SpeakerSttProcessor(cfg=cfg._speaker, transcript_updates=True)
    speaker.push_frame = AsyncMock()
    try:
        await speaker._event("wearer", {"kind": "enrolled"})
        await speaker._event("wearer", {"kind": "speech_start"})
        await speaker._event("wearer", {"kind": "speech_stop"})
        await speaker._event(
            "wearer",
            {
                "kind": "control_pending",
                "text": "hey agent",
                "pts_us": 100,
                "speaker_id": 0,
            },
        )
        await speaker._event("wearer", {"kind": "speech_start"})
        await speaker._event("wearer", {"kind": "partial", "text": "explain the chart"})
        await speaker._event("wearer", {"kind": "speech_stop"})
        await speaker._event(
            "wearer",
            {
                "kind": "transcript",
                "text": "explain the chart",
                "released": "hey agent",
                "pts_us": 200,
                "speaker_id": 0,
            },
        )
        snapshots = [
            c.args[0].update for c in speaker.push_frame.call_args_list if isinstance(c.args[0], _TurnTranscriptFrame)
        ]
        assert [u.text for u in snapshots] == ["hey agent"] + ["hey agent explain the chart"] * 2
        assert snapshots[-2].segment_id == snapshots[-1].segment_id
        assert snapshots[-1].final is True
    finally:
        await speaker.cleanup()


@pytest.mark.asyncio
async def test_retired_request_cannot_start_if_interrupt_overtakes_queued_query():
    sink = AsyncMock()
    io = _VoiceIOProcessor(sink)
    io._enable_direct_mode = True
    io.push_frame = AsyncMock()
    interruption = _TurnInterruptFrame(request_id="retired")
    interruption.transport_source = "a"
    try:
        await io.process_frame(interruption, FrameDirection.DOWNSTREAM)
        await io._spawn_query(GatedQueryFrame("a", "Old intent", False, 100, request_id="retired"))
        assert not sink.called
        assert not io._input_tasks
    finally:
        await io.cleanup()


@pytest.mark.asyncio
async def test_late_request_scoped_interrupt_does_not_cancel_newer_input_task():
    entered, release = asyncio.Event(), asyncio.Event()

    async def sink(query):
        entered.set()
        await release.wait()
        assert query.request_id == "new"

    io = _VoiceIOProcessor(sink)
    io._enable_direct_mode = True
    io.push_frame = AsyncMock()
    try:
        await io._spawn_query(GatedQueryFrame("a", "New intent", False, 200, request_id="new"))
        await asyncio.wait_for(entered.wait(), 1)
        interruption = _TurnInterruptFrame(request_id="old")
        interruption.transport_source = "a"
        await io.process_frame(interruption, FrameDirection.DOWNSTREAM)
        assert io._active_request["a"] == "new"
        tasks = tuple(io._input_tasks)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
    finally:
        await io.cleanup()


@pytest.mark.asyncio
async def test_policy_admission_and_shutdown_through_real_framework_queues():
    from pipecat.frames.frames import EndFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.workers.runner import WorkerRunner
    from test_voice_pipeline import _CaptureSink
    from xr_ai_voice._frames import _SpeakerEnrollmentFrame

    cfg = load_voice_gate_config(_CFG)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._semantic_turns = True
    gate._use_speaker_asr = True
    admission = _TurnPolicyProcessor(_Policy(), authorize=gate._accepts_turn_snapshot)
    received = []
    accepted = asyncio.Event()

    async def sink(query):
        received.append(query)
        accepted.set()

    io = _VoiceIOProcessor(sink)
    io._authorize_turn = gate._accepts_turn_snapshot
    io._reset_turn_policy = admission._reset
    io._on_turn_output = admission.output
    worker = PipelineWorker(
        Pipeline([gate, admission, io, _CaptureSink()]),
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )
    runner = WorkerRunner()
    await runner.add_workers(worker)
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(*args):
        started.set()

    async def drive():
        await started.wait()
        await worker.queue_frame(_SpeakerEnrollmentFrame("a", "enrolled"))
        await worker.queue_frame(_TurnTranscriptFrame(TranscriptUpdate("a", "source", 0, 0, "Show pressure", 100)))
        await asyncio.wait_for(accepted.wait(), 1)
        await worker.queue_frame(EndFrame())

    await asyncio.wait_for(asyncio.gather(runner.run(), drive()), 3)
    assert len(received) == 1 and received[0].request_id == "request-1"
    assert admission._worker is None and admission._closed


@pytest.mark.asyncio
async def test_queued_snapshot_and_query_lose_authority_when_conversation_closes():
    cfg = load_voice_gate_config(_CFG)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._semantic_turns = True
    gate.push_frame = AsyncMock()
    policy = _Policy()
    admission = _TurnPolicyProcessor(policy, authorize=gate._accepts_turn_snapshot)
    sink = AsyncMock()
    io = _VoiceIOProcessor(sink)
    io._authorize_turn = gate._accepts_turn_snapshot
    io.push_frame = AsyncMock()
    try:
        await gate._set_conversation_active("a", True)
        snapshot = _TurnTranscriptFrame(TranscriptUpdate("a", "source", 0, 0, "Old words", 100))
        await gate.process_frame(snapshot, FrameDirection.DOWNSTREAM)
        assert snapshot.authority is not None
        query = GatedQueryFrame("a", "Old words", False, 100, request_id="old", _turn_authority=snapshot.authority)
        await gate._set_conversation_active("a", False)
        # Both items were queued while open and arrive after the closing
        # control. Neither can recreate admission, irrespective of queue order.
        await admission.process_frame(snapshot, FrameDirection.DOWNSTREAM)
        await io._spawn_query(query)
        assert not policy.latest
        assert not sink.called and not io._input_tasks
    finally:
        await admission.cleanup()
        await io.cleanup()
        await gate.cleanup()


@pytest.mark.asyncio
async def test_semantic_interruption_drains_only_its_participants_input_publication():
    entered = {pid: asyncio.Event() for pid in ("a", "b")}
    release = asyncio.Event()
    completed = []

    async def sink(query):
        entered[query.participant_id].set()
        await release.wait()
        completed.append(query.participant_id)

    io = _VoiceIOProcessor(sink)
    io._enable_direct_mode = True
    io._reset_turn_policy = lambda pid: None
    io.push_frame = AsyncMock()
    try:
        for pid in ("a", "b"):
            await io._spawn_query(GatedQueryFrame(pid, "Query", False, 100, request_id=pid))
            await asyncio.wait_for(entered[pid].wait(), 1)
        stop = InterruptionFrame()
        stop.transport_source = "a"
        await io.process_frame(stop, FrameDirection.DOWNSTREAM)
        assert not any(owner == "a" for owner in io._input_participants.values())
        release.set()
        await asyncio.wait_for(asyncio.gather(*tuple(io._input_tasks)), 1)
        assert completed == ["b"]
    finally:
        await io.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,text", [("partial", ""), ("partial", "hey agent"), ("control_pending", "hey agent")])
async def test_same_segment_retraction_reaches_policy_without_control_words(kind, text):
    cfg = load_voice_gate_config(_CFG)
    speaker = _SpeakerSttProcessor(cfg=cfg._speaker, transcript_updates=True)
    gate = VoiceGateProcessor(cfg=cfg, tts=SimpleNamespace())
    gate._semantic_turns = True
    snapshots = []

    async def capture(frame, *args):
        if isinstance(frame, _TurnTranscriptFrame):
            snapshots.append(frame.update)

    async def to_gate(frame, *args):
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    speaker.push_frame = to_gate
    gate.push_frame = capture
    try:
        await speaker._event("wearer", {"kind": "enrolled"})
        await speaker._event("wearer", {"kind": "speech_start"})
        await speaker._event("wearer", {"kind": "partial", "text": "Move the panel"})
        await speaker._event("wearer", {"kind": kind, "text": text, "pts_us": 123_000, "speaker_id": 0})
        assert [u.text for u in snapshots] == ["Move the panel", ""]
        assert snapshots[0].source_epoch == snapshots[1].source_epoch
        assert snapshots[0].segment_id == snapshots[1].segment_id
        assert snapshots[1].revision > snapshots[0].revision
    finally:
        await speaker.cleanup()
        await gate.cleanup()
