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
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from xr_ai_models import ChatResponse, ToolCall, load_models_config, make_llm
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.image import ImageReference
from xr_ai_tools.vision import ImageQueryResult
from xr_ai_voice import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceAggregationAgent,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
)

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/sop-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from sop_sample_worker._workflow_spec import parse_workflow  # noqa: E402
from sop_sample_worker.guides import SelectedGuide, select_guide  # noqa: E402
from sop_sample_worker.replay import ReplayAgent, _AdvanceGuideRequest, _CommitRequest, _tools_for_query  # noqa: E402
from sop_sample_worker.replay_events import (  # noqa: E402
    PARTICIPANT_JOINED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
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


def context(pid="alice"):
    return SimpleNamespace(metadata=SimpleNamespace(participant_id=pid, correlation_id="turn"))


@pytest.fixture
async def engine(document):
    agent = ReplayAgent(
        guide=selected(document), llm=Mock(chat=AsyncMock()),
        current_frame=Mock(execute=AsyncMock()), image_query=Mock(execute=AsyncMock()),
        vision_timeout_s=1, aggregation=Mock(release=AsyncMock()),
    )
    agent.bind_runtime(Mock(running=True, publish=AsyncMock()))
    # Tick explicitly to make evidence and cancellation tests independent of time.
    agent._start_monitor = Mock()
    await agent.participant_joined(None, context())
    yield agent
    await agent.stop()


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


async def test_starts_on_connect_and_completes_twice_without_disconnect(engine):
    session = engine._sessions["alice"]
    assert session.step_id == "base"
    assert "Align Parts" in engine._runtime.publish.call_args.args[1].text
    for _ in range(2):
        for name in ("base", "lid"):
            session.evidence_hits = 2
            assert engine._commit(session, session.step, {name: True}, "")[1]
            response = await engine._advance("alice", skip=False)
        assert "Alignment complete. Ready to replay again. Do base." == response
        assert session.step_id == "base"
        assert session.state == {"base": False, "lid": False}
        assert not session.notices and session.evidence_hits == 0
    await engine.participant_joined(None, context())
    assert engine._sessions["alice"] is session


async def test_visual_confirmation_requires_evidence_and_announces_once(engine):
    session = engine._sessions["alice"]
    engine._runtime.publish.reset_mock()
    engine._trigger = AsyncMock(side_effect=[(True, "NO"), (True, "YES"), (True, "YES")])
    assert "not complete" in await engine._advance("alice", skip=False)
    for complete in (False, False, True):
        session.next_tick = 0
        await engine._tick(session)
        assert session.step.is_complete(session.state) is complete
    assert session.step_id == "base"
    engine._runtime.publish.assert_awaited_once()
    assert "base complete" in engine._runtime.publish.call_args.args[1].text
    await engine._tick(session)
    engine._runtime.publish.assert_awaited_once()
    engine._llm.chat.assert_not_called()


async def test_unavailable_and_wrong_direction_never_complete(engine):
    session = engine._sessions["alice"]
    engine._trigger = AsyncMock(side_effect=[(True, "YES"), (False, "no camera"), (True, "NO")])
    for _ in range(3):
        session.next_tick = 0
        await engine._tick(session)
    assert not session.state["base"]
    assert session.evidence_hits == 0


async def test_stale_commit_and_notices_discarded_on_transition(engine):
    session = engine._sessions["alice"]
    commit = engine._commit_tool(session, session.step, session.revision)
    session.notices.append("Old completion")
    await engine._advance("alice", skip=True)
    result = await commit.execute(_CommitRequest(updates={"base": True}))
    assert not result.accepted
    engine._runtime.publish.reset_mock()
    await engine._publish_notices(session)
    engine._runtime.publish.assert_not_awaited()
    assert session.step_id == "lid" and not session.state["base"]


async def test_disconnect_cancels_work_and_reconnect_starts_clean(engine):
    old = engine._sessions["alice"]
    pending = asyncio.create_task(asyncio.Event().wait())
    engine._turns["alice"] = pending
    await engine.participant_joined(None, context("bob"))
    await engine.participant_left(None, context())
    assert pending.cancelled() and "alice" not in engine._sessions
    assert "bob" in engine._sessions
    assert not engine._current(old, old.step, old.revision)
    engine._current_frame.release.assert_called_once_with("alice")
    await engine.participant_joined(None, context())
    assert engine._sessions["alice"] is not old
    assert engine._sessions["alice"].step_id == "base"


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
    assert not engine._turns and not engine._sessions
    engine._llm.chat.assert_not_called()


async def test_timer_cannot_complete_early_and_does_not_use_camera(engine, document, monkeypatch):
    now = time.time_ns()
    monkeypatch.setattr("sop_sample_worker.replay.time.time_ns", lambda: now)
    step = document["steps"][0]
    step["trigger"] = {"function": "clock__timer", "interval_s": 1, "result_field": "expired",
                       "arguments": {"started_at_us": now // 1000, "duration_s": 10}}
    step["evidence"] = {"pattern": "true", "consecutive": 1, "commit": {"base": True}}
    step["voice"]["tools"] = ["clock__timer"]
    engine._guide = selected(document)
    engine._start("alice")
    session = engine._sessions["alice"]
    await engine._tick(session)
    assert not session.state["base"]
    session.evidence_hits = 1
    assert engine._commit(session, session.step, {"base": True}, "")[0] is False
    now += 10_000_000_000
    session.next_tick = 0
    await engine._tick(session)
    assert session.state["base"]
    engine._current_frame.execute.assert_not_called()
    engine._image_query.execute.assert_not_called()


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
    before = copy.deepcopy(engine._sessions["alice"].state)
    assert "aligns" in await engine._route("Why would I skip?", "alice")
    assert before == engine._sessions["alice"].state
    engine._image_query.execute.assert_not_called()


@pytest.mark.parametrize("query", [
    "Don't advance", "My coworker said skip", "If I say restart, what happens?",
    "How do I reset?", "Should I continue?", "Start recording",
])
async def test_informational_and_capture_requests_have_no_mutating_tools(engine, query):
    tools, _ = engine._guide_tools("alice")
    names = {name for name, _ in _tools_for_query(tools, query).items()}
    assert not names.intersection({"workflow__advance", "workflow__reset", "workflow__restart"})


@pytest.mark.parametrize("query,answer", [
    ("What color is the lid?", "The lid is blue."),
    ("How many bolts are attached?", "Three bolts are attached."),
])
async def test_visual_questions_keep_camera_and_use_its_result(engine, query, answer):
    image = ImageReference(uri="xr-image://test-frame")
    engine._current_frame.execute.return_value = SimpleNamespace(image=image)
    engine._image_query.execute.return_value = ImageQueryResult(text=answer)

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
    before = copy.deepcopy(engine._sessions["alice"].state)
    assert await engine._route(query, "alice") == answer
    engine._current_frame.execute.assert_awaited_once()
    assert engine._current_frame.execute.call_args.args[0].participant_id == "alice"
    request = engine._image_query.execute.call_args.args[0]
    assert (request.image, request.query) == (image, query)
    assert engine._sessions["alice"].state == before


async def test_read_only_filter_does_not_grant_undeclared_tools(engine, document):
    document["steps"][0]["voice"]["tools"] = ["clock__now"]
    engine._guide = selected(document)
    engine._start("alice")
    engine._llm.chat.return_value = ChatResponse("Camera checking is not available for this step.",
                                               None, None, "stop", {})
    await engine._route("What color is the lid?", "alice")
    assert {tool.name for tool in engine._llm.chat.call_args.kwargs["tools"]} == {
        "clock__now", "workflow__status",
    }
    engine._current_frame.execute.assert_not_called()


@pytest.mark.parametrize("state_references", [False, True])
async def test_timer_question_uses_configured_arguments(engine, document, monkeypatch, state_references):
    started = 1_800_000_000_000_000
    arguments = {"started_at_us": started, "duration_s": 25}
    monkeypatch.setattr("sop_sample_worker.replay.time.time_ns", lambda: (started + 7_000_000) * 1000)
    step = document["steps"][0]
    step["trigger"] = {"function": "clock__timer", "interval_s": 1, "result_field": "expired",
                       "arguments": dict(arguments)}
    step["voice"] = {"prompt": "Answer questions about this step's timer.", "tools": ["clock__timer"]}
    if state_references:
        for name, value in arguments.items():
            document["state"][name] = {"type": "integer", "description": name, "initial": value}
        step["reads"] = list(arguments)
        step["trigger"]["arguments"] = {name: f"$state.{name}" for name in arguments}
    engine._guide = selected(document)
    engine._start("alice")

    async def chat(messages, *, tools, **kwargs):
        assert {tool.name for tool in tools} == {"clock__timer", "workflow__status"}
        context = json.loads(messages[1].content)
        assert context["active_trigger"] == {"function": "clock__timer", "arguments": arguments}
        if not state_references:
            assert not set(arguments).intersection(context["state"])
        if messages[-1].role != "tool":
            return ChatResponse("", None, [ToolCall(
                id="timer", name="clock__timer", arguments=json.dumps(context["active_trigger"]["arguments"]),
            )], "tool_calls", {})
        assert json.loads(messages[-1].content) == {"elapsed_s": 7, "remaining_s": 18, "expired": False}
        return ChatResponse("18 seconds remain.", None, None, "stop", {})

    engine._llm.chat.side_effect = chat
    assert await engine._route("How much time remains?", "alice") == "18 seconds remain."
    engine._current_frame.execute.assert_not_called()
    assert not engine._sessions["alice"].state["base"]


async def test_missing_timer_state_does_not_block_reset(engine, document):
    document["state"]["started_us"] = {"type": "integer", "description": "Runtime timer start"}
    step = document["steps"][0]
    step["reads"] = ["started_us"]
    step["trigger"] = {"function": "clock__timer", "interval_s": 1,
                       "arguments": {"started_at_us": "$state.started_us", "duration_s": 25}}
    engine._guide = selected(document)
    engine._start("alice")
    revision = engine._sessions["alice"].revision
    engine._llm.chat.return_value = ChatResponse("", None, [ToolCall(
        id="restart", name="workflow__restart", arguments="{}",
    )], "tool_calls", {})
    assert await engine._route("Restart the guide", "alice") == "Do base."
    trigger = json.loads(engine._llm.chat.call_args.args[0][1].content)["active_trigger"]
    assert "arguments" not in trigger
    assert "started_us" in trigger["unavailable_reason"]
    assert engine._sessions["alice"].revision > revision


async def test_foreground_guide_order_follows_links_not_yaml_order(engine, document):
    document["steps"].reverse()
    engine._guide = selected(document)
    engine._start("alice")
    assert list(engine._sessions["alice"].workflow.steps) == ["lid", "base"]
    engine._llm.chat.return_value = ChatResponse("Align the base, then the lid.", None, None, "stop", {})
    await engine._route("What order do I follow?", "alice")
    context = json.loads(engine._llm.chat.call_args.args[0][1].content)
    assert context["guide_order"] == ["base", "lid"]


async def test_model_observation_uses_guarded_commit(engine, document):
    document["steps"][0].pop("evidence")
    engine._guide = selected(document)
    engine._start("alice")
    engine._trigger = AsyncMock(return_value=(True, "The parts are aligned."))
    engine._llm.chat.side_effect = [
        ChatResponse("", None, [ToolCall(id="commit", name="workflow__commit",
                     arguments='{"updates":{"base":true}}')], "tool_calls", {}),
        ChatResponse("Done", None, None, "stop", {}),
    ]
    await engine._tick(engine._sessions["alice"])
    assert engine._sessions["alice"].state["base"]


async def test_stale_control_after_restart_rejected(engine):
    tools, _ = engine._guide_tools("alice")
    await engine._restart("alice")
    result = await tools.get("workflow__advance").execute(_AdvanceGuideRequest(skip=True))
    assert "guide changed" in result.message
    assert engine._sessions["alice"].step_id == "base"


@pytest.mark.parametrize("capture", [False, True])
def test_cli_selects_replay_with_optional_media_capture_and_quotes_name(monkeypatch, capture):
    spec = importlib.util.spec_from_file_location("sop_replay_main", _SAMPLE / "main.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    requests = []

    def launch(processes, base):
        assert [p.name for p in processes] == (["hub", "capture", "worker"] if capture else ["hub", "worker"])
        assert processes[-1].command == "sop_sample_replay"
        requests.append(json.loads(processes[-1].config.read_text()))
        assert base == _SAMPLE

    monkeypatch.setattr(main, "run_stack", launch)
    main.run(["--replay", "Align Parts"] + (["--capture"] if capture else []))
    assert requests == [{"settings": str(_SAMPLE / "yaml/replay.yaml"), "guide_name": "Align Parts"}]
    with pytest.raises(SystemExit):
        main.run(["--replay", " "])


async def test_unapproved_selection_fails_before_model_construction(tmp_path, document, monkeypatch):
    from sop_sample_worker import replay_app

    document["task"]["status"] = "draft"
    write_guide(tmp_path, document)
    settings = tmp_path / "replay.yaml"
    settings.write_text("guides_dir: .\n")
    models = Mock()
    monkeypatch.setattr(replay_app, "load_models_config", models)
    with pytest.raises(ValueError, match="draft"):
        await replay_app.run_replay(settings, "Align Parts")
    models.assert_not_called()


async def test_skip_to_end_resets_without_claiming_verified_success(engine):
    await engine._advance("alice", skip=True)
    response = await engine._advance("alice", skip=True)
    assert "skipped steps" in response and "Alignment complete" not in response
    assert not engine._sessions["alice"].skipped


def test_hash_and_approval_use_same_snapshot_during_edit(tmp_path, document, monkeypatch):
    import sop_sample_worker.guides as guides

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
    from sop_sample_worker import replay_app

    write_guide(tmp_path, document)
    config = yaml.safe_load((_SAMPLE / "yaml/replay.yaml").read_text())
    config.update(guides_dir=str(tmp_path), models_config=str(_SAMPLE / "yaml/models.replay.json"),
                  voice_gate_yaml=str(_SAMPLE / "yaml/voice_gate.replay.yaml"))
    settings = tmp_path / "replay.yaml"
    settings.write_text(yaml.safe_dump(config))
    llm = Mock(health=AsyncMock(return_value=True), close=AsyncMock())
    for name in ("make_llm", "make_vlm", "make_stt", "make_tts"):
        monkeypatch.setattr(replay_app, name, lambda *_args: llm)
    monkeypatch.setattr(replay_app, "HubVoiceTransport", lambda: SimpleNamespace(endpoint=Mock()))
    monkeypatch.setattr(ReplayAgent, "_start_monitor", Mock())
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
            assert agents[0]._sessions["alice"].step_id == "base"
            # Simulate worker shutdown while the participant is still connected.

    monkeypatch.setattr(replay_app, "ReplayAgent", replay)
    monkeypatch.setattr(replay_app, "VoiceAgent", Voice)
    await replay_app.run_replay(settings, "Align Parts")
    assert not agents[0]._sessions and not agents[0]._connected
    assert agents[0]._aggregation._stopping


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
    agent._trigger = AsyncMock(return_value=(False, "Camera unavailable"))
    sink = runtime.register("sink", Sink())
    agent.bind_runtime(runtime)
    async with runtime:
        try:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), participant_id="alice")
            assert await asyncio.wait_for(sink.outputs.get(), 2) == ("alice", "Align Parts. Do base.")
            await runtime.publish(USER_QUERY_TOPIC, UserQuery(text="What do I align?", timestamp_us=1),
                                  participant_id="alice")
            assert await asyncio.wait_for(sink.outputs.get(), 2) == ("alice", "Align the marked edges.")
            await runtime.publish(PARTICIPANT_LEFT_TOPIC, VoiceParticipantLeft(), participant_id="alice")
            assert not agent._sessions and not agent._monitors and not agent._turns
            assert not aggregation._states
        finally:
            await agent.stop()
            await aggregation.stop()


@pytest.mark.gpu
@pytest.mark.parametrize(("query", "expected", "skip"), [
    ("Please continue", "workflow__advance", False),
    ("Skip this step", "workflow__advance", True),
    ("Restart the guide", "workflow__restart", None),
    ("Stop the guide", "workflow__reset", None),
    ("What is my status?", "workflow__status", None),
    ("What does next do?", None, None),
    ("Don't advance", None, None),
    ("My coworker said skip", None, None),
    ("If I say restart, what happens?", None, None),
    ("What should I do at this step?", None, None),
    ("Tell me the weather in Paris", None, None),
    ("Start recording", None, None),
    ("What color is the lid?", "current_view", None),
    ("How many bolts are attached?", "current_view", None),
])
async def test_live_replay_prompt_routes_intent(engine, query, expected, skip):
    """Opt-in evaluation against the configured local model; no camera or media writes."""
    model = make_llm(load_models_config(_SAMPLE / "yaml/models.replay.json"), "llm")
    engine._current_frame.execute.return_value = SimpleNamespace(image=ImageReference(uri="xr-image://test"))
    engine._image_query.execute.return_value = ImageQueryResult(text="The lid is blue; three bolts are attached.")
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
async def test_live_timer_question_uses_trigger_inputs(engine, document):
    """The model must obtain timer inputs from configuration, not guess from speech."""
    arguments = {"started_at_us": time.time_ns() // 1000, "duration_s": 83}
    step = document["steps"][0]
    step["trigger"] = {"function": "clock__timer", "interval_s": 1, "arguments": arguments,
                       "result_field": "expired"}
    step["voice"] = {"prompt": "Answer questions about this waiting step.", "tools": ["clock__timer"]}
    engine._guide = selected(document)
    engine._start("alice")
    model = make_llm(load_models_config(_SAMPLE / "yaml/models.replay.json"), "llm")
    calls = []

    async def chat(*args, **kwargs):
        response = await model.chat(*args, **kwargs)
        calls.extend(response.tool_calls or ())
        return response

    engine._llm = SimpleNamespace(chat=chat)
    try:
        response = await asyncio.wait_for(engine._route("How much time remains?", "alice"), 60)
    finally:
        await model.close()
    assert response.strip()
    assert [call.name for call in calls] == ["clock__timer"]
    assert json.loads(calls[0].arguments) == arguments
