# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual procedure router and backends with controlled model decisions."""

import asyncio
import copy
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from xr_ai_hub import ParticipantEvent
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionResponse, ToolCall, load_models_config, make_decision
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC, UserQuery, VoiceOutput

from _helpers import wait_for

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide"
sys.path.insert(0, str(_SAMPLE / "worker"))

from procedure_capture_guide_worker._routing import ROUTED_QUERY_TOPIC, ProcedureRouterAgent  # noqa: E402
from procedure_capture_guide_worker.capture import CaptureAgent  # noqa: E402
from procedure_capture_guide_worker.guide import GuideAgent, approved_guides  # noqa: E402


class _Output(Agent):
    def __init__(self):
        super().__init__()
        self.messages = []

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def output(self, message: VoiceOutput, ctx: RuntimeContext):
        self.messages.append((ctx.metadata.participant_id, message.text))


class _Decisions:
    def __init__(self):
        self.choice = "help"
        self.calls = []
        self.block = None

    async def decide(self, state, questions):
        self.calls.append((state, questions))
        if self.block is not None:
            await self.block.wait()
        labels = questions["destination"].criteria
        assert self.choice in labels
        answer = DecisionAnswer(self.choice, 1.0, {name: float(name == self.choice) for name in labels})
        return DecisionResponse({"destination": answer}, "fake-decision", {}, {})


@pytest.fixture
def catalog(tmp_path):
    template = yaml.safe_load((_SAMPLE / "guides/tea-making.guide.yaml").read_text())
    for name, title, purpose in (
        ("chamomile", "Chamomile Tea", "Prepare chamomile tea"),
        ("bicycle", "Repair Bicycle Tire", "Patch a punctured bicycle tire"),
    ):
        document = copy.deepcopy(template)
        document["task"].update(id=name, name=title, foreground_prompt=purpose)
        (tmp_path / f"{name}.guide.yaml").write_text(yaml.safe_dump(document))
    return tmp_path


@pytest.fixture
async def demo(catalog, request):
    active = set()
    captured = []

    async def start(pid, *, cancelled, timestamp_us):
        if not cancelled.is_set():
            active.add(pid)

    async def finish(pid, **kwargs):
        active.discard(pid)

    backend = Mock(
        start_capture=AsyncMock(side_effect=start), finish_capture=AsyncMock(side_effect=finish),
        is_capturing=lambda pid: pid in active, stop=AsyncMock(),
        capture_transcript=AsyncMock(side_effect=lambda pid, text, timestamp: captured.append((pid, text, timestamp))),
    )
    live = getattr(request, "param", None) == "clef"
    decisions = make_decision(load_models_config(_SAMPLE / "yaml/models.json"), "router") if live else _Decisions()
    if live and not await decisions.health():
        await decisions.close()
        pytest.skip("Start the configured Clef service to evaluate real routing decisions")
    aggregation = Mock(release=AsyncMock())
    llm = Mock(chat=AsyncMock(return_value=ChatResponse("Guide answer", None, [], "stop", {})))
    waiting = asyncio.Event()
    frames = Mock(execute=AsyncMock(side_effect=lambda *_args, **_kw: waiting.wait()))
    guides = tuple((selected, GuideAgent(
        guide=selected, llm=llm, current_frame=frames, image_query=Mock(),
        aggregation=aggregation, vision_timeout_s=1,
    )) for selected in approved_guides(catalog))
    # Hold camera inference pending without substituting the execution backend.
    async def frame(*_args, **_kwargs):
        await waiting.wait()
    frames.execute.side_effect = frame
    runtime = AgentRuntime()
    output = runtime.register("output", _Output())
    agent = runtime.register("router", ProcedureRouterAgent(
        decisions=decisions, capture=CaptureAgent(backend), guides=guides, aggregation=aggregation,
    ))
    agent.bind_runtime(runtime)
    async with runtime:
        await agent.receive(ParticipantEvent("alice", True, 1, participant_session_id="first"))
        try:
            yield SimpleNamespace(
                agent=agent, runtime=runtime, output=output, decisions=decisions, backend=backend,
                guides={selected.workflow.id: guide for selected, guide in guides}, captured=captured,
                aggregation=aggregation, llm=llm,
            )
        finally:
            await agent.stop()
            if live:
                await decisions.close()


async def say(demo, text, *, choice=None, pid="alice", wait=True, timestamp=None):
    if choice is not None:
        demo.decisions.choice = choice
    query = UserQuery(text=text, timestamp_us=timestamp or time.time_ns() // 1000)
    await demo.runtime.publish(ROUTED_QUERY_TOPIC, query, participant_id=pid)
    task = demo.agent._turns.get(pid)
    if wait and task is not None:
        await task
        await demo.agent._capture.drain(pid)
    return query


async def test_capture_start_narration_stop_then_guide(demo):
    await say(demo, "I want to start capture for my task", choice="capture")
    assert demo.output.messages == [("alice", "Capture started.")]
    start_time = demo.backend.start_capture.call_args.kwargs["timestamp_us"]
    assert start_time > 1
    narration = await say(demo, "Attach the wheel, then explain how to make tea")
    assert demo.captured == [("alice", narration.text, narration.timestamp_us)]
    assert len(demo.decisions.calls) == 1  # Focus bypasses selection during narration.
    assert len(demo.output.messages) == 1
    await say(demo, "stop capture")
    assert demo.output.messages[-1] == ("alice", "Capture stopped.")
    await say(demo, "Guide me through brewing chamomile", choice="guide:chamomile")
    assert demo.guides["chamomile"].is_active("alice")
    assert not demo.guides["bicycle"].is_active("alice")
    assert demo.output.messages[-1][1].startswith("Starting Chamomile Tea.")
    demo.llm.chat.assert_not_called()  # No second model decision selects the guide.


async def test_guide_controls_and_switch_after_stop(demo):
    await say(demo, "Help me patch the tire", choice="guide:bicycle")
    demo.llm.chat.return_value = ChatResponse("", None, [
        ToolCall(id="stop", name="workflow__reset", arguments="{}"),
    ], "tool_calls", {})
    await say(demo, "stop guide", choice="guide:bicycle")
    assert not demo.guides["bicycle"].is_active("alice")
    assert "Guide stopped" in demo.output.messages[-1][1]
    await say(demo, "Record a demonstration", choice="capture")
    assert demo.output.messages[-1][1] == "Capture started."


async def test_completion_restores_all_destinations_without_stale_focus(demo):
    await say(demo, "Start the bicycle guide", choice="guide:bicycle")
    # Simulate the execution backend's automatic terminal transition.
    demo.guides["bicycle"]._backend._sessions["alice"].active = False
    await say(demo, "Start the bicycle guide again", choice="guide:bicycle")
    assert demo.guides["bicycle"].is_active("alice")
    assert demo.output.messages[-1][1].startswith("Starting Repair Bicycle Tire.")


async def test_help_does_not_start_capture_or_a_guide_and_keeps_clarification_history(demo):
    await say(demo, "What guides are available?", choice="help")
    assert "Chamomile Tea" in demo.output.messages[-1][1]
    assert "Repair Bicycle Tire" in demo.output.messages[-1][1]
    demo.backend.start_capture.assert_not_called()
    assert not any(guide.is_active("alice") for guide in demo.guides.values())
    await say(demo, "The bicycle one", choice="guide:bicycle")
    assert demo.guides["bicycle"].is_active("alice")
    assert "Which guide" in demo.decisions.calls[-1][0]["completed_history"][-1]["assistant"]


async def test_active_guide_blocks_other_destinations_and_participants_are_independent(demo):
    await say(demo, "Start chamomile", choice="guide:chamomile")
    await say(demo, "How do I do this?", choice="guide:chamomile")
    routes = {item["name"]: item for item in demo.decisions.calls[-1][0]["destinations"]}
    assert routes["guide:chamomile"]["eligible"]
    assert not routes["guide:bicycle"]["eligible"] and not routes["capture"]["eligible"]
    await demo.agent.receive(ParticipantEvent("bob", True, 1, participant_session_id="bob-session"))
    await say(demo, "Start capture", choice="capture", pid="bob")
    assert demo.output.messages[-1] == ("bob", "Capture started.")
    assert demo.guides["chamomile"].is_active("alice")


async def test_disconnect_cancels_pending_selection_and_rejects_old_transcript(demo):
    demo.decisions.block = asyncio.Event()
    await say(demo, "Start capture", choice="capture", wait=False)
    await asyncio.sleep(0.01)
    await asyncio.wait_for(demo.agent.receive(ParticipantEvent(
        "alice", False, 2, participant_session_id="first",
    )), 1)
    demo.decisions.block.set()
    demo.backend.start_capture.assert_not_called()
    assert demo.output.messages == []
    await demo.agent.receive(ParticipantEvent("alice", True, 100, participant_session_id="second"))
    await say(demo, "Start capture", timestamp=50)
    demo.backend.start_capture.assert_not_called()


async def test_superseding_capture_response_keeps_accepted_start_and_narration(demo):
    gate = asyncio.Event()
    original = demo.backend.start_capture.side_effect
    async def delayed(pid, **kwargs):
        await gate.wait()
        await original(pid, **kwargs)
    demo.backend.start_capture.side_effect = delayed
    await say(demo, "Record my task", choice="capture", wait=False)
    await wait_for(lambda: demo.backend.start_capture.called)
    await say(demo, "Attach the bracket", choice="capture", wait=False)
    gate.set()
    await demo.agent._turns["alice"]
    assert demo.captured[0][1] == "Attach the bracket"
    assert demo.backend.start_capture.call_count == 1


async def test_stop_releases_focus_before_slow_finalization_and_next_guide_waits(demo):
    await say(demo, "Start capture", choice="capture")
    finishing = asyncio.Event()
    original = demo.backend.finish_capture.side_effect
    async def delayed(pid, **kwargs):
        await finishing.wait()
        await original(pid, **kwargs)
    demo.backend.finish_capture.side_effect = delayed
    await say(demo, "Stop capture", wait=False)
    await demo.agent._turns["alice"]
    assert "alice" not in demo.agent._router._focus
    next_turn = asyncio.create_task(say(demo, "Start the bicycle guide", choice="guide:bicycle"))
    await asyncio.sleep(0.01)
    assert not demo.guides["bicycle"].is_active("alice")
    finishing.set()
    await asyncio.wait_for(next_turn, 1)
    assert demo.guides["bicycle"].is_active("alice")
    assert demo.backend.start_capture.call_count == 1


async def test_stop_without_active_capture_cannot_start_recording(demo):
    await say(demo, "Stop capture", choice="capture")
    demo.backend.start_capture.assert_not_called()
    assert not demo.agent._capture.has_capture("alice")


def test_catalog_only_registers_approved_unique_valid_snapshots(catalog):
    initial = approved_guides(catalog)
    path = catalog / "chamomile.guide.yaml"
    content = yaml.safe_load(path.read_text())
    content["task"]["status"] = "draft"
    path.write_text(yaml.safe_dump(content))
    assert [guide.workflow.id for guide in approved_guides(catalog)] == ["bicycle"]
    assert initial[1].workflow.runnable  # Existing workers retain the validated bytes.
    content["task"].update(status="approved", id="bicycle")
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="unique task IDs"):
        approved_guides(catalog)
    path.write_text("invalid: [")
    with pytest.raises(ValueError, match="Invalid guides"):
        approved_guides(catalog)


async def test_composed_router_worker_wires_one_input_path_and_drains_both_backends(monkeypatch):
    from dataclasses import replace

    from procedure_capture_guide_worker import app
    from procedure_capture_guide_worker.config import load_config
    from procedure_capture_guide_worker.guide.config import load_guide_config

    capture_config = load_config(_SAMPLE / "yaml/worker.yaml")
    config = replace(capture_config, mode="router", guide=load_guide_config(
        _SAMPLE / "yaml/guide-agent-config.yaml", guide_name=None,
    ))
    decisions = _Decisions()
    decisions.health = AsyncMock(return_value=True)
    decisions.close = AsyncMock()
    clients = [Mock(health=AsyncMock(return_value=True), close=AsyncMock()) for _ in range(4)]
    for factory, client in zip(("make_stt", "make_tts", "make_vlm", "make_llm"), clients, strict=True):
        monkeypatch.setattr(app, factory, lambda *_args, client=client: client)
    monkeypatch.setattr(app, "make_decision", lambda *_args: decisions)
    forever = asyncio.Event()
    endpoint = Mock(run=forever.wait, wait_until_running=AsyncMock())
    monkeypatch.setattr(app, "ProcessorEndpoint", lambda **_: endpoint)
    frames = Mock(execute=AsyncMock(side_effect=lambda *_args, **_kwargs: forever.wait()))
    async def pending_frame(*_args, **_kwargs):
        await forever.wait()
    frames.execute.side_effect = pending_frame
    monkeypatch.setattr(app, "CurrentFrameTool", lambda **_: frames)
    active = set()
    async def start(pid, **_kwargs):
        active.add(pid)
    async def finish(pid, **_kwargs):
        active.discard(pid)
    backend = Mock(start_capture=AsyncMock(side_effect=start), finish_capture=AsyncMock(side_effect=finish),
                   is_capturing=lambda pid: pid in active, stop=AsyncMock())
    monkeypatch.setattr(app, "_Capture", lambda **_: backend)
    messages = []

    class Voice(Agent):
        def __init__(self, **kwargs):
            super().__init__()
            assert kwargs["query_topic"] == ROUTED_QUERY_TOPIC
            assert set(kwargs["probes"]) == {"stt", "tts", "vlm", "llm", "router"}
            assert kwargs["text_input"] is True

        @subscribe(VOICE_OUTPUT_TOPIC)
        async def output(self, message: VoiceOutput, ctx: RuntimeContext):
            messages.append(message.text)

        async def run(self, runtime):
            endpoint.on_participant.assert_called_once()
            await endpoint.on_participant.call_args.args[0](ParticipantEvent(
                "alice", True, 1, participant_session_id="session",
            ))
            for text, choice, expected in (
                ("Record my demonstration", "capture", "Capture started."),
                ("Stop capture", "capture", "Capture stopped."),
                ("Start a chamomile tea guide", "guide:tea_making", "Starting Tea Making."),
            ):
                decisions.choice = choice
                await runtime.publish(ROUTED_QUERY_TOPIC, UserQuery(
                    text=text, timestamp_us=time.time_ns() // 1000,
                ), participant_id="alice")
                await wait_for(lambda: any(message.startswith(expected) for message in messages))

    monkeypatch.setattr(app, "VoiceAgent", Voice)
    await asyncio.wait_for(app.run_app(config), 5)
    assert backend.start_capture.call_count == 1
    assert backend.finish_capture.call_count == 1
    backend.stop.assert_awaited_once()
    decisions.close.assert_awaited_once()
    clients[2].close.assert_awaited_once()
    clients[3].close.assert_awaited_once()
    endpoint.close.assert_called_once()


@pytest.mark.gpu
@pytest.mark.parametrize("demo", ["clef"], indirect=True)
@pytest.mark.parametrize(("query", "expected"), [
    ("I want to start capture for my task", "capture"),
    ("Record a demonstration of folding this towel", "capture"),
    ("Help me brew chamomile", "chamomile"),
    ("Start a guide to patch my bike tire", "bicycle"),
    ("What guides do you have?", "help"),
    ("Start a guide", "help"),
    ("Guide me through replacing a laptop battery", "help"),
    ("Do not start capture", "help"),
    ("What happens when I say start capture?", "help"),
])
async def test_local_clef_routes_natural_language_without_second_selection(demo, query, expected):
    """Only destination selection uses a live model; camera, speech, and storage stay simulated."""
    await say(demo, query)
    owner = demo.agent._owner("alice")
    assert owner == (None if expected == "help" else "capture" if expected == "capture" else f"guide:{expected}")
    assert demo.output.messages
