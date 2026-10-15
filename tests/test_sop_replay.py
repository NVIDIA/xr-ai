# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay selection, participant isolation, evidence guards, and repeated runs."""

import asyncio
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from xr_ai_models import ChatResponse, ToolCall, load_models_config, make_llm
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageReference
from xr_ai_tools.types import EmptyRequest
from xr_ai_tools.vision import ImageQueryResult
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceAggregationAgent,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-replay"
sys.path.insert(0, str(_SAMPLE / "worker"))

from procedure_capture_replay_worker.config import WorkerConfig, load_config  # noqa: E402
from procedure_capture_replay_worker.replay._workflow_spec import parse_workflow  # noqa: E402
from procedure_capture_replay_worker.replay.agent import (  # noqa: E402
    ReplayAgent,
    _tools_for_query,
)
from procedure_capture_replay_worker.replay.config import ReplayConfig  # noqa: E402
from procedure_capture_replay_worker.replay.events import (  # noqa: E402
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
)
from procedure_capture_replay_worker.replay.guides import SelectedGuide, select_guide  # noqa: E402
from procedure_capture_replay_worker.replay.replaying import (  # noqa: E402
    _AdvanceGuideRequest,
    _CommitRequest,
    _FrameIdentity,
    _Observation,
    _Replay,
)


@pytest.fixture
def document():
    def step(name, next_step):
        return {
            "id": name, "title": name, "reads": [], "writes": [name],
            "trigger": {"function": "current_view", "interval_s": 1, "arguments": {"question": "Aligned?"}},
            "agent": {"prompt": "Commit only if aligned.", "tools": []},
            "voice": {"prompt": "Help align the parts.", "tools": ["current_view"]},
            "evidence": {"pattern": "YES", "consecutive": 2, "commit": {name: True}},
            "complete_when": {name: True}, "next": next_step,
            "messages": {"enter": f"Do {name}.", "complete": f"{name} complete. Say next.", "skip": "Skipped."},
        }
    return {
        "schema_version": 1,
        "task": {"id": "align_parts", "name": "Align Parts", "version": 1, "status": "approved",
                 "source_session": "test", "start_step": "base", "foreground_prompt": "Guide alignment.",
                 "complete_message": "Alignment complete."},
        "state": {name: {"type": "boolean", "initial": False, "description": "Verified alignment"}
                  for name in ("base", "lid")},
        "steps": [step("base", "lid"), step("lid", None)],
    }


def selected(document):
    content = yaml.safe_dump(document).encode()
    return SelectedGuide(Path("test.guide.yaml"), hashlib.sha256(content).hexdigest(), parse_workflow(content))


def observation(text, sequence):
    return _Observation(True, text, _FrameIdentity(
        participant_id="alice", track_id="cam", timestamp_us=sequence * 1000, sequence=sequence,
    ))


def context(pid="alice"):
    return SimpleNamespace(metadata=SimpleNamespace(participant_id=pid, correlation_id="turn"))


@pytest.fixture
async def connected_engine(document):
    agent = ReplayAgent(
        guide=selected(document), llm=Mock(chat=AsyncMock()),
        current_frame=Mock(execute=AsyncMock()), image_query=Mock(execute=AsyncMock()),
        vision_timeout_s=1, aggregation=Mock(release=AsyncMock()),
    )
    agent.bind_runtime(Mock(running=True, publish=AsyncMock()))
    # Tick explicitly to make evidence and cancellation tests independent of time.
    agent._backend._start_monitor = Mock()
    await agent.participant_joined(None, context())
    yield agent
    await agent.stop()


@pytest.fixture
def engine(connected_engine):
    # Verification tests begin with an active run; connection tests use the
    # inactive fixture and exercise the foreground start command explicitly.
    connected_engine._backend._start("alice")
    return connected_engine


def write_guide(directory, document, name="test.guide.yaml"):
    path = directory / name
    path.write_text(yaml.safe_dump(document))
    return path


def test_approved_exact_name_and_id_are_pinned(tmp_path, document):
    path = write_guide(tmp_path, document)
    original = path.read_bytes()
    guide = select_guide(tmp_path, "align parts")
    assert guide.sha256 == hashlib.sha256(original).hexdigest()
    assert select_guide(tmp_path, "ALIGN_PARTS").path == path
    document["task"]["status"] = "draft"
    write_guide(tmp_path, document)
    assert guide.workflow.status == "approved"
    with pytest.raises(ValueError, match="draft"):
        select_guide(tmp_path, "Align Parts")
    with pytest.raises(ValueError, match="No guide"):
        select_guide(tmp_path, "align")


def test_ambiguity_and_symlinks_are_rejected(tmp_path, document):
    path = write_guide(tmp_path, document)
    write_guide(tmp_path, document, "other.guide.yaml")
    with pytest.raises(ValueError, match="Ambiguous"):
        select_guide(tmp_path, "Align Parts")
    sub = tmp_path / "linked"
    sub.mkdir()
    (sub / "link.guide.yaml").symlink_to(path)
    with pytest.raises(ValueError, match="symbolic-link"):
        select_guide(sub, "Align Parts")


@pytest.mark.parametrize("trigger", [
    {"function": "current_view", "arguments": {}},
    {"function": "current_view", "arguments": {"question": "x" * 501}},
    {"function": "current_view", "arguments": {"question": "Visible?"}, "result_field": "bad"},
    {"function": "clock__timer", "arguments": {"started_at_us": 1}},
    {"function": "clock__timer", "arguments": {"started_at_us": 1, "duration_s": 5}, "result_field": "bad"},
    {"function": "clock__timer", "arguments": {"started_at_us": True, "duration_s": 5}},
    {"function": "clock__timer", "arguments": {"started_at_us": 1, "duration_s": 0}},
    {"function": "clock__timer", "arguments": {"duration_s": True}},
    {"function": "clock__timer", "arguments": {"duration_s": 0}},
    {"function": "clock__timer", "arguments": {"duration_s": 5}, "result_field": "bad"},
])
def test_invalid_trigger_rejected_before_replay(tmp_path, document, trigger):
    document["steps"][0]["trigger"] = {"interval_s": 1, **trigger}
    write_guide(tmp_path, document)
    with pytest.raises(ValueError, match="Invalid guides"):
        select_guide(tmp_path, "Align Parts")


@pytest.mark.parametrize("path", [
    ("steps", 0, "evidence", "pattern"),
    ("steps", 0, "agent", "prompt"),
    ("steps", 0, "voice", "prompt"),
    ("steps", 0, "messages", "enter"),
    ("steps", 0, "messages", "complete"),
    ("steps", 0, "messages", "skip"),
    ("task", "name"),
    ("task", "foreground_prompt"),
    ("task", "complete_message"),
    ("state", "base", "description"),
])
@pytest.mark.parametrize("value", [["YES"], {"YES": True}, True, 23, 1.5, None])
def test_non_string_text_rejected_before_replay(tmp_path, document, path, value):
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    write_guide(tmp_path, document)
    with pytest.raises(ValueError, match="must be a string"):
        select_guide(tmp_path, "Align Parts")


def test_string_regex_and_whitespace_keep_their_meaning(document):
    document["steps"][0]["evidence"]["pattern"] = "  [YES]  "
    document["steps"][0]["agent"]["prompt"] = "  Observe carefully.  "
    step = parse_workflow(yaml.safe_dump(document)).steps["base"]
    assert step.agent.prompt == "Observe carefully."
    assert step.evidence.pattern == "[YES]"
    assert re.fullmatch(step.evidence.pattern, "Y")
    assert not re.fullmatch(step.evidence.pattern, "YES")
    document["steps"][0]["evidence"]["pattern"] = " \n "
    with pytest.raises(ValueError, match="must not be empty"):
        parse_workflow(yaml.safe_dump(document))


@pytest.mark.parametrize("original,replacement,key", [
    ("steps:\n", "steps: []\nsteps:\n", "steps"),
    ("    initial: false", "    initial: false\n    initial: true", "initial"),
    ("    initial: false", '    initial: false\n    "initial": true', "initial"),
    ("  evidence:\n", "  evidence: {pattern: wrong, consecutive: 1}\n  evidence:\n", "evidence"),
    ("    consecutive: 2", "    consecutive: 2\n    consecutive: 2", "consecutive"),
    ("  complete_when:\n    base: true", "  complete_when:\n    base: true\n  complete_when:\n    base: false",
     "complete_when"),
    ("    initial: false", "    <<: {initial: false}\n    initial: true", "initial"),
    ("    initial: false", "    <<: {initial: false}\n    <<: {description: Merged}", "<<"),
])
def test_duplicate_requirements_rejected_before_replay(tmp_path, document, original, replacement, key):
    source = yaml.safe_dump(document, sort_keys=False)
    assert original in source
    changed = source.replace(original, replacement, 1)
    if key == "complete_when":
        # The permissive loader would accept the initial state as already complete.
        assert yaml.safe_load(changed)["steps"][0]["complete_when"] == {"base": False}
    with pytest.raises(yaml.constructor.ConstructorError, match=f"duplicate mapping key '{key}'") as error:
        parse_workflow(changed)
    assert error.value.problem_mark.line >= 0
    path = tmp_path / "invalid.guide.yaml"
    path.write_text(changed)
    with pytest.raises(ValueError, match=f"duplicate mapping key '{key}'"):
        select_guide(tmp_path, "Align Parts")
    assert path.read_text() == changed


def test_safe_aliases_and_non_overlapping_merges_remain_supported(document):
    document["steps"][0]["voice"] = document["steps"][0]["agent"]
    source = yaml.safe_dump(document, sort_keys=False)
    assert "&id" in source and "*id" in source
    source = source.replace("    type: boolean", "    <<: {type: boolean}")
    workflow = parse_workflow(source)
    assert workflow.steps["base"].voice == workflow.steps["base"].agent
    assert workflow.state_fields["base"].type == "boolean"


def test_guide_loader_is_local_and_still_safe():
    assert yaml.safe_load("value: 1\nvalue: 2") == {"value": 2}
    with pytest.raises(yaml.constructor.ConstructorError):
        parse_workflow("!!python/object/apply:builtins.str [YES]")


async def test_starts_on_command_and_completes_twice_without_disconnect(connected_engine):
    engine = connected_engine
    session = engine._backend._sessions["alice"]
    assert session.step_id == "base"
    assert "Align Parts" in engine._runtime.publish.call_args.args[1].text
    assert "Say start replay" in engine._runtime.publish.call_args.args[1].text
    assert not session.active and session.timer is None
    engine._backend._start_monitor.assert_not_called()
    await engine._backend._tick(session)
    engine._backend._current_frame.execute.assert_not_awaited()
    engine._llm.chat.return_value = ChatResponse("", None, [
        ToolCall(id="start", name="workflow__restart", arguments="{}"),
    ], "tool_calls", {})
    for _ in range(2):
        assert await engine._route("start replay", "alice") == "Do base."
        assert session.active
        for name in ("base", "lid"):
            session.evidence_hits = 2
            assert engine._backend._commit(session, session.step, {name: True}, "")[1]
            response = await control(engine._backend, "advance", skip=False)
        assert "Alignment complete. Say start replay to begin again." == response
        assert session.step_id == "base" and not session.active
        assert session.state == {"base": False, "lid": False}
        assert not session.notices and session.evidence_hits == 0
    await engine.participant_joined(None, context())
    assert engine._backend._sessions["alice"] is session


async def test_connected_timer_does_not_start_until_start_replay(connected_engine, document, monkeypatch):
    backend = connected_engine._backend
    document["steps"][0]["trigger"] = {
        "function": "clock__timer", "interval_s": 1,
        "arguments": {"duration_s": 60}, "result_field": "expired",
    }
    backend._guide = selected(document)
    await connected_engine.participant_left(None, context())
    await connected_engine.participant_joined(None, context())
    session = backend._sessions["alice"]
    assert not session.active and session.timer is None
    monkeypatch.setattr("procedure_capture_replay_worker.replay.replaying.time", SimpleNamespace(monotonic=lambda: 500))
    await control(backend, "restart")
    assert session.active and session.timer.started == 500
    assert session.timer.read().remaining_s == 60


async def test_visual_confirmation_requires_evidence_and_announces_once(engine):
    session = engine._backend._sessions["alice"]
    engine._runtime.publish.reset_mock()
    engine._backend._trigger = AsyncMock(side_effect=[
        observation("NO", 1), observation("YES", 2), observation("YES", 3),
    ])
    assert "not complete" in await control(engine._backend, "advance", skip=False)
    for complete in (False, False, True):
        session.next_tick = 0
        await engine._backend._tick(session)
        assert session.step.is_complete(session.state) is complete
    assert session.step_id == "base"
    engine._runtime.publish.assert_awaited_once()
    assert "base complete" in engine._runtime.publish.call_args.args[1].text
    await engine._backend._tick(session)
    engine._runtime.publish.assert_awaited_once()
    engine._llm.chat.assert_not_called()


async def test_unavailable_and_wrong_direction_never_complete(engine):
    session = engine._backend._sessions["alice"]
    engine._backend._trigger = AsyncMock(side_effect=[
        observation("YES", 1), _Observation(False, "no camera"), observation("NO", 3),
    ])
    for _ in range(3):
        session.next_tick = 0
        await engine._backend._tick(session)
    assert not session.state["base"]
    assert session.evidence_hits == 0


async def test_stale_commit_and_notices_discarded_on_transition(engine):
    session = engine._backend._sessions["alice"]
    commit = engine._backend._commit_tool(session, session.step, session.revision)
    session.notices.append("Old completion")
    await control(engine._backend, "advance", skip=True)
    result = await commit.execute(_CommitRequest(updates={"base": True}))
    assert not result.accepted
    engine._runtime.publish.reset_mock()
    await engine._backend._publish_notices(session)
    engine._runtime.publish.assert_not_awaited()
    assert session.step_id == "lid" and not session.state["base"]


async def test_disconnect_cancels_work_and_reconnect_starts_clean(engine):
    old = engine._backend._sessions["alice"]
    pending = asyncio.create_task(asyncio.Event().wait())
    engine._turns["alice"] = pending
    await engine.participant_joined(None, context("bob"))
    await engine.participant_left(None, context())
    assert pending.cancelled() and "alice" not in engine._backend._sessions
    assert "bob" in engine._backend._sessions
    assert not engine._backend._current(old, old.step, old.revision)
    engine._backend._current_frame.release.assert_called_once_with("alice")
    await engine.participant_joined(None, context())
    assert engine._backend._sessions["alice"] is not old
    assert engine._backend._sessions["alice"].step_id == "base"
    assert not engine._backend._sessions["alice"].active


async def test_departure_while_input_cleanup_awaits_does_not_restart_turn(engine):
    entered, release = asyncio.Event(), asyncio.Event()

    async def cleanup(_pid):
        entered.set()
        await release.wait()

    engine._aggregation.release.side_effect = cleanup
    query = asyncio.create_task(engine.user_query(UserQuery(text="next", timestamp_us=1), context()))
    await entered.wait()
    leaving = asyncio.create_task(engine.participant_left(None, context()))
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(query, leaving), 2)
    assert not engine._turns and not engine._backend._sessions
    engine._llm.chat.assert_not_called()


async def test_timer_cannot_complete_early_and_does_not_use_camera(engine, document, monkeypatch):
    now = 100.0
    monkeypatch.setattr("procedure_capture_replay_worker.replay.replaying.time.monotonic", lambda: now)
    step = document["steps"][0]
    step["trigger"] = {"function": "clock__timer", "interval_s": 1, "result_field": "expired",
                       "arguments": {"duration_s": 10}}
    step["evidence"] = {"pattern": "true", "consecutive": 1, "commit": {"base": True}}
    step["voice"]["tools"] = ["clock__timer"]
    engine._backend._guide = selected(document)
    engine._backend._start("alice")
    session = engine._backend._sessions["alice"]
    await engine._backend._tick(session)
    assert not session.state["base"]
    session.evidence_hits = 1
    assert engine._backend._commit(session, session.step, {"base": True}, "")[0] is False
    now += 10
    session.next_tick = 0
    await engine._backend._tick(session)
    assert session.state["base"]
    engine._backend._current_frame.execute.assert_not_called()
    engine._backend._image_query.execute.assert_not_called()


async def test_model_written_timestamp_cannot_backdate_runtime_timer(engine, document, monkeypatch):
    from procedure_capture_replay_worker.replay import replaying

    now = 100.0
    monkeypatch.setattr(replaying, "time", SimpleNamespace(monotonic=lambda: now, time_ns=lambda: 10**20))
    document["state"]["started_us"] = {"type": "integer", "description": "Model-supplied observation"}
    document["steps"][0]["writes"].append("started_us")
    document["steps"][0].pop("evidence")
    wait = document["steps"][1]
    wait["trigger"] = {"function": "clock__timer", "interval_s": 1, "result_field": "expired",
                       "arguments": {"duration_s": 60}}
    wait["evidence"] = {"pattern": "true", "consecutive": 1, "commit": {"lid": True}}
    backend = engine._backend
    backend._guide = selected(document)
    backend._start("alice")
    session = backend._sessions["alice"]
    backend._current_frame.execute.return_value = camera_frame(1)
    backend._image_query.execute.return_value = ImageQueryResult(text="Aligned")
    engine._llm.chat.return_value = ChatResponse("", None, [ToolCall(
        id="commit", name="workflow__commit", arguments='{"updates":{"base":true,"started_us":1}}',
    )], "tool_calls", {})
    await backend._tick(session)
    assert session.state["started_us"] == 1 and session.state["base"]
    await control(backend, "advance", skip=False)
    assert session.timer.started == now
    timer_tool = backend._timer_tool("alice")
    assert (await timer_tool.execute(EmptyRequest())).remaining_s == 60
    await backend._tick(session)
    assert not session.state["lid"]
    # Even a direct model commit cannot substitute a fabricated observation for elapsed time.
    result = await backend._commit_tool(session, session.step, session.revision).execute(
        _CommitRequest(updates={"lid": True}),
    )
    assert not result.accepted and result.message == "timer has not expired"
    now += 59
    session.next_tick = 0
    await backend._tick(session)
    assert not session.state["lid"]
    now += 1
    session.next_tick = 0
    await backend._tick(session)
    assert session.state["lid"]
    await control(backend, "advance", skip=False)
    assert not session.active and session.timer is None
    await control(backend, "restart")
    await control(backend, "advance", skip=True)
    assert session.timer.started == now
    assert not (await backend._timer_tool("alice").execute(EmptyRequest())).expired
    with pytest.raises(Exception, match="timer changed"):
        await timer_tool.execute(EmptyRequest())


async def test_active_timer_duration_cannot_be_rewritten(engine, document, monkeypatch):
    from procedure_capture_replay_worker.replay import replaying

    now = 10.0
    monkeypatch.setattr(replaying, "time", SimpleNamespace(monotonic=lambda: now))
    document["state"]["duration"] = {"type": "integer", "description": "Configured wait", "initial": 60}
    step = document["steps"][0]
    step["writes"].append("duration")
    step.pop("evidence")
    step["trigger"] = {"function": "clock__timer", "interval_s": 1,
                       "arguments": {"duration_s": "$state.duration"}, "result_field": "expired"}
    backend = engine._backend
    backend._guide = selected(document)
    backend._start("alice")
    session = backend._sessions["alice"]
    assert backend._commit(session, session.step, {"duration": 1}, "")[0]
    now += 2
    assert (await backend._timer_tool("alice").execute(EmptyRequest())).remaining_s == 58
    assert not backend._commit(session, session.step, {"base": True}, "")[0]
    await control(backend, "restart")
    assert session.timer.duration_s == 60 and session.timer.started == now


async def test_timer_advance_checks_runtime_even_with_preexisting_completion(engine, document, monkeypatch):
    from procedure_capture_replay_worker.replay import replaying

    now = 100.0
    monkeypatch.setattr(replaying, "time", SimpleNamespace(monotonic=lambda: now))
    document["state"]["base"]["initial"] = True
    document["steps"][0]["trigger"] = {"function": "clock__timer", "interval_s": 1,
                                      "arguments": {"duration_s": 60}, "result_field": "expired"}
    backend = engine._backend
    backend._guide = selected(document)
    backend._start("alice")
    assert "not complete" in await control(backend, "advance", skip=False)
    assert "Complete" not in await control(backend, "status")
    now += 60
    assert await control(backend, "advance", skip=False) == "Do lid."


async def control(backend, action, **arguments):
    guidance = await backend.guidance("alice")
    tool = guidance.tools.get(f"workflow__{action}")
    request = _AdvanceGuideRequest(**arguments) if action == "advance" else EmptyRequest()
    return (await tool.execute(request)).message


def camera_frame(sequence, *, timestamp_us=None, track_id="cam"):
    return ImageFrame(
        image=ImageReference(uri="xr-image://test"), width=16, height=16, participant_id="alice",
        track_id=track_id, sequence=sequence, timestamp_us=timestamp_us or sequence * 1000,
    )


@pytest.mark.parametrize("model_commit", [False, True])
@pytest.mark.parametrize("track_id", ["cam", ""])
async def test_consecutive_visual_evidence_requires_distinct_source_frames(engine, document, model_commit, track_id):
    backend = engine._backend
    if model_commit:
        document["steps"][0]["evidence"].pop("commit")
        backend._guide = selected(document)
        backend._start("alice")
        engine._llm.chat.return_value = ChatResponse("", None, [ToolCall(
            id="commit", name="workflow__commit", arguments='{"updates":{"base":true}}',
        )], "tool_calls", {})
    session = backend._sessions["alice"]
    sequence = 1 if track_id else 0  # On-demand snapshots have no track sequence.
    backend._current_frame.execute.return_value = camera_frame(sequence, timestamp_us=1000, track_id=track_id)
    backend._image_query.execute.return_value = ImageQueryResult(text="YES")
    for _ in range(3):
        session.next_tick = 0
        await backend._tick(session)
    assert session.evidence_hits == 1 and not session.state["base"]
    assert engine._llm.chat.call_count == (1 if model_commit else 0)
    if model_commit:
        prompt = json.loads(engine._llm.chat.call_args.args[0][1].content)
        assert prompt["frame"] == {"participant_id": "alice", "track_id": track_id,
                                   "sequence": sequence, "timestamp_us": 1000}
    # An older frame cannot supply independent supporting evidence either.
    backend._current_frame.execute.return_value = camera_frame(sequence, timestamp_us=500, track_id=track_id)
    session.next_tick = 0
    await backend._tick(session)
    assert session.evidence_hits == 1
    backend._current_frame.execute.return_value = camera_frame(sequence, timestamp_us=2000, track_id=track_id)
    session.next_tick = 0
    await backend._tick(session)
    assert session.state["base"] and session.evidence_hits == 2


async def test_unavailable_frame_breaks_consecutive_evidence(engine):
    backend = engine._backend
    session = backend._sessions["alice"]
    backend._trigger = AsyncMock(side_effect=[
        observation("YES", 1), _Observation(False, "no frame"), observation("YES", 2), observation("YES", 3),
    ])
    for hits in (1, 0, 1, 2):
        session.next_tick = 0
        await backend._tick(session)
        assert session.evidence_hits == hits
        assert session.state["base"] is (hits == 2)


async def test_frame_inference_finishing_after_restart_cannot_change_progress(engine):
    backend = engine._backend
    session = backend._sessions["alice"]
    entered, release = asyncio.Event(), asyncio.Event()

    async def inspect(_request):
        entered.set()
        await release.wait()
        return ImageQueryResult(text="YES")

    backend._current_frame.execute.return_value = camera_frame(1)
    backend._image_query.execute.side_effect = inspect
    tick = asyncio.create_task(backend._tick(session))
    await entered.wait()
    await control(backend, "restart")
    release.set()
    await tick
    assert not session.state["base"] and session.evidence_hits == 0
    assert not session.observed_frames


async def test_foreground_controls_use_tools_and_cannot_commit(engine):
    engine._llm.chat.return_value = ChatResponse("", None, [ToolCall(
        id="next", name="workflow__advance", arguments='{"skip":false}',
    )], "tool_calls", {})
    response = await engine._route("Please continue", "alice")
    assert "not complete" in response
    tools = engine._llm.chat.call_args.kwargs["tools"]
    assert "workflow__advance" in {tool.name for tool in tools}
    assert "workflow__commit" not in {tool.name for tool in tools}
    assert not any("record" in tool.name for tool in tools)
    engine._llm.chat.return_value = ChatResponse("This step aligns the parts.", None, None, "stop", {})
    before = copy.deepcopy(engine._backend._sessions["alice"].state)
    assert "aligns" in await engine._route("Why would I skip?", "alice")
    assert before == engine._backend._sessions["alice"].state
    engine._backend._image_query.execute.assert_not_called()


@pytest.mark.parametrize("query", [
    "Don't advance", "My coworker said skip", "If I say restart, what happens?",
    "How do I reset?", "Should I continue?", "Start recording",
    "What does start replay do?", "Don't start replay", "How do I stop replay?",
])
async def test_informational_and_capture_requests_have_no_mutating_tools(engine, query):
    tools = (await engine._backend.guidance("alice")).tools
    names = {name for name, _ in _tools_for_query(tools, query).items()}
    assert not names.intersection({"workflow__advance", "workflow__reset", "workflow__restart"})


@pytest.mark.parametrize("query,answer", [
    ("What color is the lid?", "The lid is blue."),
    ("How many bolts are attached?", "Three bolts are attached."),
])
async def test_visual_questions_keep_camera_and_use_its_result(engine, query, answer):
    image = ImageReference(uri="xr-image://test-frame")
    engine._backend._current_frame.execute.return_value = camera_frame(1).model_copy(update={"image": image})
    engine._backend._image_query.execute.return_value = ImageQueryResult(text=answer)

    async def chat(messages, *, tools, **kwargs):
        names = {tool.name for tool in tools}
        assert names == {"current_view", "workflow__status"}
        assert json.loads(messages[1].content)["active_trigger"] is None
        if messages[-1].role != "tool":
            return ChatResponse("", None, [ToolCall(
                id="inspect", name="current_view", arguments=json.dumps({"question": query}),
            )], "tool_calls", {})
        assert messages[-1].content == answer
        return ChatResponse(answer, None, None, "stop", {})

    engine._llm.chat.side_effect = chat
    before = copy.deepcopy(engine._backend._sessions["alice"].state)
    assert await engine._route(query, "alice") == answer
    engine._backend._current_frame.execute.assert_awaited_once()
    assert engine._backend._current_frame.execute.call_args.args[0].participant_id == "alice"
    request = engine._backend._image_query.execute.call_args.args[0]
    assert (request.image, request.query) == (image, query)
    assert engine._backend._sessions["alice"].state == before


async def test_read_only_filter_does_not_grant_undeclared_tools(engine, document):
    document["steps"][0]["voice"]["tools"] = ["clock__now"]
    engine._backend._guide = selected(document)
    engine._backend._start("alice")
    engine._llm.chat.return_value = ChatResponse("Camera checking is not available for this step.",
                                               None, None, "stop", {})
    await engine._route("What color is the lid?", "alice")
    assert {tool.name for tool in engine._llm.chat.call_args.kwargs["tools"]} == {
        "clock__now", "workflow__status",
    }
    engine._backend._current_frame.execute.assert_not_called()


@pytest.mark.parametrize("state_references", [False, True])
@pytest.mark.parametrize("query", ["How much time remains?", "Check the remaining time"])
async def test_timer_question_uses_configured_arguments(engine, document, monkeypatch, state_references, query):
    now = 100.0
    arguments = {"duration_s": 25}
    monkeypatch.setattr("procedure_capture_replay_worker.replay.replaying.time.monotonic", lambda: now)
    step = document["steps"][0]
    step["trigger"] = {"function": "clock__timer", "interval_s": 1, "result_field": "expired",
                       "arguments": dict(arguments)}
    step["voice"] = {"prompt": "Answer questions about this step's timer.", "tools": ["clock__timer"]}
    if state_references:
        for name, value in arguments.items():
            document["state"][name] = {"type": "integer", "description": name, "initial": value}
        step["reads"] = list(arguments)
        step["trigger"]["arguments"] = {name: f"$state.{name}" for name in arguments}
    engine._backend._guide = selected(document)
    engine._backend._start("alice")

    now += 7

    async def chat(messages, *, tools, **kwargs):
        names = {tool.name for tool in tools}
        assert {"clock__timer", "workflow__status"} <= names
        assert ("workflow__advance" in names) is query.startswith("Check")
        # Cover the composed system prompt on informational and command-shaped turns.
        system = messages[0].content
        assert "call clock__timer with no arguments ({})" in system
        assert "active_trigger.arguments" not in system
        assert "Answer questions about this step's timer." in system
        timer_definition = next(tool for tool in tools if tool.name == "clock__timer")
        assert not timer_definition.parameters.get("properties")
        context = json.loads(messages[1].content)
        assert context["active_trigger"] == {
            "function": "clock__timer", "duration_s": 25,
            "status": {"elapsed_s": 7, "remaining_s": 18, "expired": False},
        }
        if not state_references:
            assert not set(arguments).intersection(context["state"])
        if messages[-1].role != "tool":
            return ChatResponse("", None, [ToolCall(
                id="timer", name="clock__timer", arguments="{}",
            )], "tool_calls", {})
        assert json.loads(messages[-1].content) == {"elapsed_s": 7, "remaining_s": 18, "expired": False}
        return ChatResponse("18 seconds remain.", None, None, "stop", {})

    engine._llm.chat.side_effect = chat
    assert await engine._route(query, "alice") == "18 seconds remain."
    engine._backend._current_frame.execute.assert_not_called()
    assert not engine._backend._sessions["alice"].state["base"]


async def test_missing_timer_state_does_not_block_reset(engine, document):
    document["state"]["duration_s"] = {"type": "integer", "description": "Wait duration"}
    step = document["steps"][0]
    step["reads"] = ["duration_s"]
    step["trigger"] = {"function": "clock__timer", "interval_s": 1,
                       "arguments": {"duration_s": "$state.duration_s"}}
    engine._backend._guide = selected(document)
    engine._backend._start("alice")
    revision = engine._backend._sessions["alice"].revision
    engine._llm.chat.return_value = ChatResponse("", None, [ToolCall(
        id="restart", name="workflow__restart", arguments="{}",
    )], "tool_calls", {})
    assert await engine._route("start replay", "alice") == "Do base."
    trigger = json.loads(engine._llm.chat.call_args.args[0][1].content)["active_trigger"]
    assert "arguments" not in trigger
    assert "duration_s" in trigger["unavailable_reason"]
    assert engine._backend._sessions["alice"].revision > revision


async def test_foreground_guide_order_follows_links_not_yaml_order(engine, document):
    document["steps"].reverse()
    engine._backend._guide = selected(document)
    engine._backend._start("alice")
    assert list(engine._backend._sessions["alice"].workflow.steps) == ["lid", "base"]
    engine._llm.chat.return_value = ChatResponse("Align the base, then the lid.", None, None, "stop", {})
    await engine._route("What order do I follow?", "alice")
    context = json.loads(engine._llm.chat.call_args.args[0][1].content)
    assert context["guide_order"] == ["base", "lid"]


async def test_model_observation_uses_guarded_commit(engine, document):
    document["steps"][0].pop("evidence")
    engine._backend._guide = selected(document)
    engine._backend._start("alice")
    engine._backend._trigger = AsyncMock(return_value=observation("The parts are aligned.", 1))
    engine._llm.chat.side_effect = [
        ChatResponse("", None, [ToolCall(id="commit", name="workflow__commit",
                     arguments='{"updates":{"base":true}}')], "tool_calls", {}),
        ChatResponse("Done", None, None, "stop", {}),
    ]
    await engine._backend._tick(engine._backend._sessions["alice"])
    assert engine._backend._sessions["alice"].state["base"]


async def test_stale_control_after_restart_rejected(engine):
    tools = (await engine._backend.guidance("alice")).tools
    await control(engine._backend, "restart")
    result = await tools.get("workflow__advance").execute(_AdvanceGuideRequest(skip=True))
    assert "guide changed" in result.message
    assert engine._backend._sessions["alice"].step_id == "base"


async def test_stop_cancels_verification_and_restart_begins_fresh(engine):
    backend = engine._backend
    session = backend._sessions["alice"]
    pinned = session.guide
    stale_commit = backend._commit_tool(session, session.step, session.revision)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def inspect(_request):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    backend._current_frame.execute.return_value = camera_frame(1)
    backend._image_query.execute.side_effect = inspect
    backend._start_monitor = _Replay._start_monitor.__get__(backend)
    backend._start_monitor("alice")
    await asyncio.wait_for(entered.wait(), 2)
    engine._llm.chat.return_value = ChatResponse("", None, [ToolCall(
        id="stop", name="workflow__reset", arguments="{}",
    )], "tool_calls", {})
    assert await engine._route("stop replay", "alice") == "Replay stopped. Say start replay to begin again."
    # Native tool cancellation is delivered through Relay's dispatcher; observe
    # its cleanup rather than assuming it runs before the control reply returns.
    await asyncio.wait_for(cancelled.wait(), 2)
    assert cancelled.is_set() and "alice" not in backend._monitors
    assert not session.active and session.timer is None and session.guide is pinned
    engine._runtime.publish.reset_mock()
    for _ in range(3):
        await backend._tick(session)
    backend._image_query.execute.assert_awaited_once()
    assert not (await stale_commit.execute(_CommitRequest(updates={"base": True}))).accepted
    session.notices.append("Obsolete completion")
    await backend._publish_notices(session)
    engine._runtime.publish.assert_not_awaited()
    stopped = await backend.guidance("alice")
    assert stopped.context["active"] is False and stopped.context["active_trigger"] is None
    assert stopped.context["state"] == {}
    assert set(dict(stopped.tools.items())) == {"workflow__status", "workflow__reset", "workflow__restart"}
    assert "inactive" in await control(backend, "status")
    entered.clear()
    engine._llm.chat.return_value = ChatResponse("", None, [ToolCall(
        id="restart", name="workflow__restart", arguments="{}",
    )], "tool_calls", {})
    assert await engine._route("start replay", "alice") == "Do base."
    await asyncio.wait_for(entered.wait(), 2)
    assert session.active and session.step_id == "base" and session.state == {"base": False, "lid": False}
    assert session.guide is pinned and "alice" in backend._monitors


async def test_stopping_discards_timer_and_restart_starts_new_wait(engine, document, monkeypatch):
    from procedure_capture_replay_worker.replay import replaying

    now = 100.0
    monkeypatch.setattr(replaying, "time", SimpleNamespace(monotonic=lambda: now))
    document["steps"][0]["trigger"] = {
        "function": "clock__timer", "interval_s": 1, "arguments": {"duration_s": 60}, "result_field": "expired",
    }
    backend = engine._backend
    backend._guide = selected(document)
    backend._start("alice")
    session = backend._sessions["alice"]
    timer = backend._timer_tool("alice")
    now += 20
    await control(backend, "reset")
    assert not session.active and session.timer is None
    now += 100
    await backend._tick(session)
    assert not session.state["base"]
    with pytest.raises(Exception, match="timer changed"):
        await timer.execute(EmptyRequest())
    await control(backend, "restart")
    assert session.active and session.timer.started == now
    assert (await backend._timer_tool("alice").execute(EmptyRequest())).remaining_s == 60


async def test_guidance_snapshot_is_detached_and_answer_scope_expires(engine):
    backend = engine._backend
    session = backend._sessions["alice"]
    guidance = await backend.guidance("alice")
    guidance.context["state"]["base"] = True
    guidance.context["guide_order"].clear()
    assert not session.state["base"]
    assert (await backend.guidance("alice")).context["guide_order"] == ["base", "lid"]
    await guidance.tools.get("workflow__status").execute(EmptyRequest())
    await control(backend, "restart")
    assert "guide changed" in await guidance.finish("The old step was complete.")


async def test_control_checks_scope_after_acquiring_execution_lock(engine):
    backend = engine._backend
    session = backend._sessions["alice"]
    guidance = await backend.guidance("alice")
    async with session.lock:
        pending = asyncio.create_task(guidance.tools.get("workflow__advance").execute(_AdvanceGuideRequest(skip=True)))
        await asyncio.sleep(0)
        backend._reset_progress(session, active=False)
    response = await asyncio.wait_for(pending, 2)
    assert "guide changed" in response.message
    assert not session.active and session.step_id == "base"


@pytest.mark.parametrize("depart", [False, True])
async def test_accepted_stop_survives_foreground_cancellation_and_is_drained(engine, depart):
    backend = engine._backend
    entered, release = asyncio.Event(), asyncio.Event()

    async def invalidate(_pid):
        entered.set()
        await release.wait()

    backend._invalidate = invalidate
    guidance = await backend.guidance("alice")
    pending = asyncio.create_task(guidance.tools.get("workflow__reset").execute(EmptyRequest()))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert backend._control_tasks["alice"]
        if depart:
            after = asyncio.create_task(engine.participant_left(None, context()))
        else:
            after = asyncio.create_task(backend.guidance("alice"))
        await asyncio.sleep(0)
        assert not after.done()
        release.set()
        result = await asyncio.wait_for(after, 2)
        if depart:
            assert not backend._sessions and not backend._control_tasks
        else:
            assert result.context["active"] is False
            assert not backend._control_tasks["alice"]
        assert not backend._monitors
    finally:
        release.set()


async def test_foreground_uses_only_backend_lifecycle_and_guidance_interface(engine):
    backend = engine._backend
    engine._backend = SimpleNamespace(
        join=backend.join, leave=backend.leave, stop=backend.stop, guidance=backend.guidance,
    )
    engine._llm.chat.return_value = ChatResponse("Align the marked edges.", None, None, "stop", {})
    await engine.participant_joined(None, context("bob"))
    assert await engine._route("What do I align?", "bob") == "Align the marked edges."
    await engine.participant_left(None, context("bob"))


@pytest.mark.parametrize("markers,expected_cleanups", [
    (set(), 0), ({"gpu"}, 2), ({"gpu", "external_models"}, 0),
])
def test_external_model_evaluations_do_not_remove_deployments(monkeypatch, markers, expected_cleanups):
    import conftest

    cleanup = Mock()
    monkeypatch.setattr(conftest, "_purge_xr_ai_vllm_containers", cleanup)
    request = SimpleNamespace(node=SimpleNamespace(
        get_closest_marker=lambda name: name if name in markers else None,
    ))
    lifecycle = conftest._purge_xr_ai_vllm_between_gpu_tests.__wrapped__(request)
    next(lifecycle)
    with pytest.raises(StopIteration):
        next(lifecycle)
    assert cleanup.call_count == expected_cleanups


def test_cli_selects_replay_without_capture_and_quotes_name(monkeypatch):
    spec = importlib.util.spec_from_file_location("sop_replay_main", _SAMPLE / "main.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    requests = []

    def launch(processes, base):
        assert [p.name for p in processes] == ["hub", "worker"]
        assert processes[-1].command == "procedure_capture_replay_worker"
        requests.append(json.loads(processes[-1].config.read_text()))
        assert base == _SAMPLE

    monkeypatch.setattr(main, "run_stack", launch)
    main.run(["--mode", "replay", "Align Parts"])
    assert requests == [{"worker_config": str(_SAMPLE / "yaml/worker.yaml"), "mode": "replay",
                         "guide_name": "Align Parts", "output_root": None}]
    for arguments in (["--mode", "replay"], ["--mode", "replay", " "],
                      ["--mode", "capture", "Align Parts"], ["--mode", "unknown"],
                      ["--mode", "replay", "Align Parts", "--output-root", "output"]):
        with pytest.raises(SystemExit):
            main.run(arguments)
    assert len(requests) == 1


async def test_unapproved_selection_fails_before_model_construction(tmp_path, document, monkeypatch):
    from procedure_capture_replay_worker import app

    document["task"]["status"] = "draft"
    write_guide(tmp_path, document)
    models = Mock()
    monkeypatch.setattr(app, "load_models_config", models)
    with pytest.raises(ValueError, match="draft"):
        await app.run_app(WorkerConfig(
            mode="replay", models_config=_SAMPLE / "yaml/models.json",
            voice_gate_yaml=_SAMPLE / "yaml/voice_gate.replay.yaml",
            replay=ReplayConfig(tmp_path, "Align Parts"),
        ))
    models.assert_not_called()


def test_mode_configuration_is_isolated_and_models_are_shared(tmp_path):
    capture = load_config(_SAMPLE / "yaml/worker.yaml")
    assert capture.mode == "capture" and capture.replay is None
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "worker_config": str(_SAMPLE / "yaml/worker.yaml"), "mode": "replay", "guide_name": "Align Parts",
    }))
    replay = load_config(request)
    assert replay.mode == "replay" and replay.capture is None
    assert replay.models_config == capture.models_config
    assert replay.replay.guide_name == "Align Parts"
    assert replay.replay.guides_dir.resolve() == _SAMPLE / "guides"
    assert replay.voice_gate_yaml == _SAMPLE / "yaml/voice_gate.replay.yaml"


@pytest.mark.parametrize("mode", ["capture", "replay"])
def test_config_does_not_load_inactive_capability(tmp_path, mode):
    settings = tmp_path / "worker.yaml"
    settings.write_text("capture_config: capture.yaml\nreplay_config: replay.yaml\n")
    # Only the selected capability's file exists.
    (tmp_path / f"{mode}.yaml").write_text("{}")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "worker_config": str(settings), "mode": mode,
        "guide_name": "Align Parts" if mode == "replay" else None,
    }))
    config = load_config(request)
    assert (config.capture is not None) == (mode == "capture")
    assert (config.replay is not None) == (mode == "replay")


async def test_capture_composition_does_not_construct_replay_or_llm(tmp_path, monkeypatch):
    from dataclasses import replace

    from procedure_capture_replay_worker import app

    config = load_config(_SAMPLE / "yaml/worker.yaml")
    config = replace(config, capture=replace(config.capture, artifacts_dir=tmp_path))
    model = Mock(health=AsyncMock(return_value=True), close=AsyncMock())
    for name in ("make_vlm", "make_stt", "make_tts"):
        monkeypatch.setattr(app, name, lambda *_args: model)
    forbidden = Mock(side_effect=AssertionError("Capture must not construct replay or LLM"))
    for name in ("make_llm", "ReplayAgent", "VoiceAggregationAgent", "select_guide"):
        monkeypatch.setattr(app, name, forbidden)

    async def receive():
        await asyncio.Future()

    endpoint = Mock(run=receive, wait_until_running=AsyncMock())
    monkeypatch.setattr(app, "ProcessorEndpoint", lambda **kwargs: endpoint)

    class Voice(Agent):
        def __init__(self, **kwargs):
            super().__init__()
            assert set(kwargs["probes"]) == {"stt", "tts", "vlm"}
            assert kwargs["query_topic"].name == app.CAPTURE_INPUT_TOPIC.name
            assert not kwargs["text_input"]

        async def run(self, runtime):
            pass

    monkeypatch.setattr(app, "VoiceAgent", Voice)
    await app.run_app(config)
    forbidden.assert_not_called()
    handlers = [call.args[0] for call in endpoint.on_participant.call_args_list]
    assert any(isinstance(handler.__self__, app.CaptureAgent) for handler in handlers)
    endpoint.close.assert_called_once()
    model.close.assert_awaited_once()


async def test_skip_to_end_resets_without_claiming_verified_success(engine):
    await control(engine._backend, "advance", skip=True)
    response = await control(engine._backend, "advance", skip=True)
    assert "skipped steps" in response and "Alignment complete" not in response
    assert not engine._backend._sessions["alice"].skipped


def test_hash_and_approval_use_same_snapshot_during_edit(tmp_path, document, monkeypatch):
    import procedure_capture_replay_worker.replay.guides as guides

    path = write_guide(tmp_path, document)
    approved = path.read_bytes()
    parse = guides.parse_workflow

    def edit_during_parse(content):
        document["task"]["status"] = "draft"
        write_guide(tmp_path, document)
        return parse(content)

    monkeypatch.setattr(guides, "parse_workflow", edit_during_parse)
    pinned = guides.select_guide(tmp_path, "Align Parts")
    assert pinned.workflow.runnable
    assert pinned.sha256 == hashlib.sha256(approved).hexdigest()
    with pytest.raises(ValueError, match="draft"):
        guides.select_guide(tmp_path, "Align Parts")


async def test_replay_app_composition_starts_and_cleans_up(tmp_path, document, monkeypatch):
    from procedure_capture_replay_worker import app

    write_guide(tmp_path, document)
    config = WorkerConfig(
        mode="replay", models_config=_SAMPLE / "yaml/models.json",
        voice_gate_yaml=_SAMPLE / "yaml/voice_gate.replay.yaml", replay=ReplayConfig(tmp_path, "Align Parts"),
    )
    llm = Mock(health=AsyncMock(return_value=True), close=AsyncMock())
    for name in ("make_llm", "make_vlm", "make_stt", "make_tts"):
        monkeypatch.setattr(app, name, lambda *_args: llm)
    async def receive():
        await asyncio.Future()
    endpoint = Mock(run=receive, wait_until_running=AsyncMock())
    monkeypatch.setattr(app, "ProcessorEndpoint", lambda **kwargs: endpoint)
    capture = Mock(side_effect=AssertionError("Replay must not create a capture backend"))
    monkeypatch.setattr(app, "_Capture", capture)
    monkeypatch.setattr(_Replay, "_start_monitor", Mock())
    agents = []

    def replay(**kwargs):
        agent = ReplayAgent(**kwargs)
        agents.append(agent)
        return agent

    class Voice(Agent):
        def __init__(self, **kwargs):
            super().__init__()
            assert set(kwargs["probes"]) == {"llm", "vlm", "stt", "tts"}
            assert not kwargs["voice_gate"].magic_phrases
            assert not kwargs["voice_gate"].stop_commands_enabled

        async def run(self, runtime):
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), participant_id="alice")
            assert agents[0]._backend._sessions["alice"].step_id == "base"
            # Simulate worker shutdown while the participant is still connected.

    monkeypatch.setattr(app, "ReplayAgent", replay)
    monkeypatch.setattr(app, "VoiceAgent", Voice)
    await app.run_app(config)
    assert not agents[0]._backend._sessions and not agents[0]._connected
    assert agents[0]._aggregation._stopping
    capture.assert_not_called()
    endpoint.close.assert_called_once()


async def test_actual_runtime_and_aggregation_deliver_replay_and_release(document):
    class Sink(Agent):
        def __init__(self):
            super().__init__()
            self.outputs = asyncio.Queue()

        @subscribe(VOICE_OUTPUT_TOPIC)
        async def receive(self, output: VoiceOutput, ctx: RuntimeContext):
            await self.outputs.put((ctx.metadata.participant_id, output.text))

    llm = Mock(chat=AsyncMock(return_value=ChatResponse("Align the marked edges.", None, None, "stop", {})))
    runtime = AgentRuntime()
    aggregation = runtime.register("aggregation", VoiceAggregationAgent(
        llm=llm, coalesce_window_s=0.01, minimum_playback_s=0, speech_rate_wpm=60000,
    ))
    agent = runtime.register("replay", ReplayAgent(
        guide=selected(document), llm=llm, current_frame=Mock(), image_query=Mock(),
        vision_timeout_s=1, aggregation=aggregation,
    ))
    agent._backend._trigger = AsyncMock(return_value=_Observation(False, "Camera unavailable"))
    sink = runtime.register("sink", Sink())
    agent.bind_runtime(runtime)
    async with runtime:
        try:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), participant_id="alice")
            assert await asyncio.wait_for(sink.outputs.get(), 2) == (
                "alice", "Align Parts is ready. Say start replay to begin.",
            )
            assert not agent._backend._sessions["alice"].active
            assert not agent._backend._monitors
            llm.chat.return_value = ChatResponse("", None, [
                ToolCall(id="start", name="workflow__restart", arguments="{}"),
            ], "tool_calls", {})
            await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="start replay", timestamp_us=1),
                                  participant_id="alice")
            assert await asyncio.wait_for(sink.outputs.get(), 2) == ("alice", "Do base.")
            assert agent._backend._sessions["alice"].active
            llm.chat.return_value = ChatResponse("Align the marked edges.", None, None, "stop", {})
            await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="What do I align?", timestamp_us=1),
                                  participant_id="alice")
            assert await asyncio.wait_for(sink.outputs.get(), 2) == ("alice", "Align the marked edges.")
            await runtime.publish(PARTICIPANT_LEFT_TOPIC, VoiceParticipantLeft(), participant_id="alice")
            assert not agent._backend._sessions and not agent._backend._monitors and not agent._turns
            assert not aggregation._states
        finally:
            await agent.stop()
            await aggregation.stop()


@pytest.mark.parametrize(("control_name", "final_step"), [
    ("workflow__restart", False), ("workflow__reset", False),
    ("workflow__advance", False), ("workflow__advance", True),
])
async def test_transition_discards_notice_queued_during_foreground_turn(document, control_name, final_step):
    class Sink(Agent):
        def __init__(self):
            super().__init__()
            self.outputs = asyncio.Queue()

        @subscribe(VOICE_OUTPUT_TOPIC)
        async def receive(self, output: VoiceOutput, ctx: RuntimeContext):
            await self.outputs.put(output.text)

    entered, release = asyncio.Event(), asyncio.Event()

    async def chat(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return ChatResponse("", None, [ToolCall(id="control", name=control_name, arguments="{}")], "tool_calls", {})

    llm = Mock(chat=AsyncMock(side_effect=chat))
    runtime = AgentRuntime()
    # Hold non-urgent notices in the real aggregation queue until the transition.
    aggregation = runtime.register("aggregation", VoiceAggregationAgent(
        llm=llm, coalesce_window_s=60, minimum_playback_s=0, speech_rate_wpm=60000,
    ))
    agent = runtime.register("replay", ReplayAgent(
        guide=selected(document), llm=llm, current_frame=Mock(), image_query=Mock(),
        vision_timeout_s=1, aggregation=aggregation,
    ))
    backend = agent._backend
    backend._start_monitor = Mock()
    sink = runtime.register("sink", Sink())
    agent.bind_runtime(runtime)
    async with runtime:
        try:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), participant_id="alice")
            assert await asyncio.wait_for(sink.outputs.get(), 2) == "Align Parts is ready. Say start replay to begin."
            await control(backend, "restart")
            if final_step:
                await control(backend, "advance", skip=True)
            session = backend._sessions["alice"]
            session.evidence_hits = 2
            assert backend._commit(session, session.step, {session.step_id: True}, "")[1]
            # The observation committed before the foreground began, but its notice
            # is published after input-time aggregation cleanup and before control.
            await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="Continue now", timestamp_us=1),
                                  participant_id="alice")
            await asyncio.wait_for(entered.wait(), 2)
            await backend._publish_notices(session)
            obsolete = aggregation._states["alice"].task
            release.set()
            expected_step = "lid" if control_name == "workflow__advance" and not final_step else "base"
            output = await asyncio.wait_for(sink.outputs.get(), 2)
            assert obsolete.cancelled()
            if control_name == "workflow__reset":
                assert output == "Replay stopped. Say start replay to begin again."
                assert not session.active and "alice" not in backend._monitors
            elif final_step:
                assert output == "Replay ended with skipped steps. Say start replay to begin again."
                assert not session.active and "alice" not in backend._monitors
            else:
                assert output.endswith(f"Do {expected_step}.") and "complete. Say next." not in output
            assert session.step_id == expected_step and not session.state[expected_step]
            assert not session.notices and sink.outputs.empty()
            assert not aggregation._states["alice"].pending
        finally:
            release.set()
            await agent.stop()
            await aggregation.stop()


@pytest.mark.gpu
@pytest.mark.external_models
@pytest.mark.parametrize(("query", "expected", "skip"), [
    ("Please continue", "workflow__advance", False),
    ("Skip this step", "workflow__advance", True),
    ("Restart the guide", "workflow__restart", None),
    ("Start replay", "workflow__restart", None),
    ("Stop replay", "workflow__reset", None),
    ("Stop the guide", "workflow__reset", None),
    ("What is my status?", "workflow__status", None),
    ("What does next do?", None, None),
    ("Don't advance", None, None),
    ("My coworker said skip", None, None),
    ("If I say restart, what happens?", None, None),
    ("What does start replay do?", None, None),
    ("What should I do at this step?", None, None),
    ("Tell me the weather in Paris", None, None),
    ("Start recording", None, None),
    ("What color is the lid?", "current_view", None),
    ("How many bolts are attached?", "current_view", None),
])
async def test_live_replay_prompt_routes_intent(engine, query, expected, skip):
    """Opt-in evaluation against the configured local model; no camera or media writes."""
    model = make_llm(load_models_config(_SAMPLE / "yaml/models.json"), "llm")
    if query == "Start replay":
        await control(engine._backend, "reset")
    engine._backend._current_frame.execute.return_value = camera_frame(1)
    engine._backend._image_query.execute.return_value = ImageQueryResult(
        text="The lid is blue; three bolts are attached.",
    )
    calls = []

    async def chat(*args, **kwargs):
        response = await model.chat(*args, **kwargs)
        calls.extend(response.tool_calls or ())
        return response

    engine._llm = SimpleNamespace(chat=chat)
    try:
        response = await asyncio.wait_for(engine._route(query, "alice"), 60)
    finally:
        await model.close()
    assert response.strip()
    assert [call.name for call in calls] == ([] if expected is None else [expected])
    if skip is not None:
        assert json.loads(calls[0].arguments)["skip"] is skip


@pytest.mark.gpu
@pytest.mark.external_models
@pytest.mark.parametrize("query", ["How much time remains?", "Check the remaining time"])
async def test_live_timer_question_uses_trigger_inputs(engine, document, query):
    """The model reads the bound runtime timer without manufacturing its inputs."""
    arguments = {"duration_s": 83}
    step = document["steps"][0]
    step["trigger"] = {"function": "clock__timer", "interval_s": 1, "arguments": arguments,
                       "result_field": "expired"}
    step["voice"] = {"prompt": "Answer questions about this waiting step.", "tools": ["clock__timer"]}
    engine._backend._guide = selected(document)
    engine._backend._start("alice")
    model = make_llm(load_models_config(_SAMPLE / "yaml/models.json"), "llm")
    calls = []

    async def chat(*args, **kwargs):
        response = await model.chat(*args, **kwargs)
        calls.extend(response.tool_calls or ())
        return response

    engine._llm = SimpleNamespace(chat=chat)
    try:
        response = await asyncio.wait_for(engine._route(query, "alice"), 60)
    finally:
        await model.close()
    assert response.strip()
    assert [call.name for call in calls] == ["clock__timer"]
    assert json.loads(calls[0].arguments) == {}
