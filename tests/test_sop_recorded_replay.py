# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise recorded packet evaluation through the production guide runtime.

Model responses are deterministic doubles. Camera selection, evidence
deduplication, conversation tools, timers and delivery run their normal code;
fault-injection tests also check that reported progress agrees with speech.
These regressions test the harness contract, not local-model perception.
"""

import asyncio
import importlib.util
import io
import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from PIL import Image

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide"
_TEST_DATA = Path(__file__).resolve().parent
_SCRIPTS = _SAMPLE / "evaluation/guide"


@pytest.fixture
def packets(monkeypatch):
    monkeypatch.syspath_prepend(str(_SAMPLE / "worker"))
    monkeypatch.syspath_prepend(str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location("recorded_camera_eval", _SCRIPTS / "_recorded_camera.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_packet(directory, count=16, color="blue"):
    frames = directory / "frames"
    frames.mkdir()
    rows = []
    for i in range(count):
        path = frames / f"{i}.png"
        Image.new("RGB", (640, 480), color).save(path)
        rows.append({"frame_id": i, "timestamp_us": 1000000 + i * 100000,
                     "path": f"frames/{i}.png", "width": 640, "height": 480})
    (frames / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    packet = directory / "packet.json"
    packet.write_text(json.dumps({"status": "complete", "files": {"frame_index": "frames/index.jsonl"}}))
    return packet


@pytest.mark.parametrize("case", ["empty", "out_of_order", "duplicate", "escape", "index_escape"])
def test_invalid_packet_evidence_is_rejected(packets, tmp_path, case):
    packet = make_packet(tmp_path, 2)
    index = tmp_path / "frames/index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if case == "empty":
        rows = []
    elif case == "out_of_order":
        rows.reverse()
    elif case == "duplicate":
        rows[1]["frame_id"] = rows[0]["frame_id"]
    elif case == "escape":
        rows[1]["path"] = "../outside.png"
    else:
        packet.write_text(json.dumps({"status": "complete", "files": {"frame_index": "../outside.jsonl"}}))
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        packets.capture_frames(packet)


@pytest.fixture
def runtime_models(monkeypatch):
    import xr_ai_models
    from xr_ai_models import ChatResponse, ToolCall

    async def chat(messages, **kwargs):
        if messages[-1].role == "tool":
            return ChatResponse("Acknowledged.", None, None, "stop", {})
        query = json.loads(messages[1].content)["request"]
        tool, arguments = {
            "Start guide": ("workflow__restart", {}),
            "Next": ("workflow__advance", {"skip": False}),
        }[query]
        return ChatResponse("", None, [ToolCall(id="packet", name=tool, arguments=json.dumps(arguments))],
                            "tool_calls", {})

    async def vision(images, query, **kwargs):
        assert len(images) == 1 and "rectangle blue" in query
        with Image.open(io.BytesIO(images[0])) as source:
            red, _green, blue = source.convert("RGB").getpixel((320, 240))
        return SimpleNamespace(content="YES" if blue > red else "NO")

    llm = Mock(chat=AsyncMock(side_effect=chat), close=AsyncMock())
    vlm = Mock(ask_images=AsyncMock(side_effect=vision), close=AsyncMock())
    monkeypatch.setattr(xr_ai_models, "make_llm", lambda *_: llm)
    monkeypatch.setattr(xr_ai_models, "make_vlm", lambda *_: vlm)
    return llm, vlm


@pytest.mark.parametrize("color,count,automatic,expected", [
    ("blue", 16, True, True),
    ("blue", 16, False, True),
    ("red", 16, True, False),
    ("blue", 1, True, False),
])
def test_packet_cli_uses_runtime_controls_and_unique_source_evidence(
    packets, runtime_models, tmp_path, monkeypatch, color, count, automatic, expected,
):
    llm, vlm = runtime_models
    packet = make_packet(tmp_path, count, color)
    guide = yaml.safe_load((_TEST_DATA / "test_procedure_indicator_guide.yaml").read_text())
    guide["task"]["status"] = "draft"
    guide["steps"][1]["trigger"]["arguments"]["duration_s"] = 1
    guide["steps"][1]["finish_on_complete"] = automatic
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(guide))
    original = path.read_bytes()
    original_packet = packet.read_bytes()
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["guide.py", "--packet", str(packet), "--guide", str(path),
                                    "--output", str(output), "--tail-s", "15" if expected else "2"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(str(_SCRIPTS / "evaluate_recorded_packet.py"), run_name="__main__")
    assert exit_status.value.code == (0 if expected else 1)
    result = json.loads((output / "result.json").read_text())
    assert result["passed"] is expected
    assert result["draft_test_only"]
    assert path.read_bytes() == original == (output / "guide.yaml").read_bytes()
    assert packet.read_bytes() == original_packet
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_status"] == "draft"
    assert len(manifest["frames"]) == count
    assert all(len(frame["sha256"]) == 64 for frame in manifest["frames"])
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    published = [e["source"]["frame_id"] for e in events if e["kind"] == "source_frame"]
    assert published == list(range(len(published)))
    assert len(published) <= count
    turns = [e["text"] for e in events if e["kind"] == "user"]
    assert turns == (["Start guide", "Next"] + ([] if automatic else ["Next"]) if expected else ["Start guide"])
    snapshots = [e for e in events if e["kind"] == "guidance_snapshot"]
    if expected:
        assert snapshots[-1]["context"]["active"] is False
        timer_entered = next(e for e in snapshots if e["context"]["active_trigger"] is not None)
        assert snapshots[-1]["elapsed_s"] - timer_entered["elapsed_s"] >= 0.9
        assert result["checks_not_scored"] == 0
    else:
        assert result["checks_not_scored"] > 0
    sources = json.loads((output / "runtime.json").read_text())
    assert {"evaluation/guide/evaluate_recorded_packet.py",
            "evaluation/guide/_recorded_camera.py",
            "evaluation/guide/_runtime_harness.py"} <= sources.keys()
    assert not any("example-guides/" in source or "tests/" in source for source in sources)
    assert (output / "relay.jsonl").is_file()
    llm.close.assert_awaited_once()
    vlm.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("runner", ["packet", "scenario"])
@pytest.mark.parametrize("wrong_foreground_call", [False, True])
async def test_evaluation_attributes_overlapping_model_calls_to_their_origin(
    packets, runtime_models, tmp_path, monkeypatch, runner, wrong_foreground_call,
):
    from procedure_capture_guide_worker.guide.agent import GuideAgent
    from xr_ai_models import ChatMessage, ChatResponse, ToolCall

    llm, _vlm = runtime_models
    foreground_chat = llm.chat.side_effect
    observed = asyncio.Event()
    observations = 0

    async def chat(messages, **kwargs):
        nonlocal observations
        payload = json.loads(messages[1].content)
        if payload.get("request") == "auxiliary":
            return ChatResponse("Auxiliary output.", None, None, "stop", {})
        if "request" not in payload:
            observations += 1
            # First exercise the real observation commit tool without completing
            # the step; completion follows after the start response is delivered.
            updates = {} if observations == 1 else {"visible": True}
            observed.set()
            return ChatResponse("", None, [ToolCall(
                id=f"observation-{observations}", name="workflow__commit",
                arguments=json.dumps({"updates": updates}),
            )], "tool_calls", {})
        response = await foreground_chat(messages, **kwargs)
        if wrong_foreground_call and payload["request"] == "Next":
            # A genuine wrong foreground call must still fail routing assertions.
            response.tool_calls[0] = ToolCall(id="wrong", name="workflow__status", arguments="{}")
        return response

    llm.chat.side_effect = chat
    route = GuideAgent._route

    async def overlapping_route(self, text, participant_id):
        response = await route(self, text, participant_id)
        if text == "Start guide":
            async with asyncio.timeout(5):
                await observed.wait()
            # Another component's task can inherit the foreground context without
            # belonging to its model/tool loop (for example, voice aggregation).
            await asyncio.create_task(self._llm.chat((
                ChatMessage(role="system", content="Auxiliary component"),
                ChatMessage(role="user", content=json.dumps({"request": "auxiliary"})),
            )))
        return response

    monkeypatch.setattr(GuideAgent, "_route", overlapping_route)
    packet = make_packet(tmp_path, count=30)
    guide = yaml.safe_load((_TEST_DATA / "test_procedure_indicator_guide.yaml").read_text())
    guide["steps"] = guide["steps"][:1]
    step = guide["steps"][0]
    step["next"] = None
    step["evidence"].pop("commit")
    step["evidence"]["consecutive"] = 1
    step["agent"] = {"prompt": "Commit visible only for a blue indicator.", "tools": []}
    guide_path = tmp_path / "guide.yaml"
    guide_path.write_text(yaml.safe_dump(guide))
    output = tmp_path / "result"
    if runner == "packet":
        spec = importlib.util.spec_from_file_location(
            "attribution_packet_eval", _SCRIPTS / "evaluate_recorded_packet.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        passed = await module.evaluate(packet, guide_path, output, tail_s=5)
    else:
        tea_eval = _SAMPLE / "example-guides/tea-making/evaluation"
        monkeypatch.syspath_prepend(str(tea_eval))
        spec = importlib.util.spec_from_file_location("attribution_scenario_eval", tea_eval / "run_scenario.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = tmp_path / "scenario.yaml"
        manifest.write_text(yaml.safe_dump({
            "guide_path": "guide.yaml", "fail_fast": True, "actions": [
                {"action": "image", "path": "frames/0.png"},
                {"action": "say", "text": "Start guide", "tools": ["workflow__restart"]},
                {"action": "wait_speech", "expect": "The indicator is blue", "timeout_s": 5},
                {"action": "say", "text": "Next", "tools": ["workflow__advance"],
                 "arguments": [{"skip": False}]},
            ],
        }))
        passed = await module.evaluate(manifest, output)

    assert passed is not wrong_foreground_call
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    start = next(event for event in events if event["kind"] == "user")
    delivered = next(index for index, event in enumerate(events)
                     if event["kind"] == "speech" and event["timestamp_us"] == start["timestamp_us"])
    background = [event for event in events[:delivered]
                  if event["kind"] == "model" and event["origin"] == "observation"]
    assert background and all(event["turn_id"] is None for event in background)
    assert background[0]["calls"][0]["name"] == "workflow__commit"
    auxiliary = next(event for event in events if event.get("text") == "Auxiliary output.")
    assert auxiliary["origin"] == "unscoped" and auxiliary["turn_id"] is None
    turns = [event for event in events if event["kind"] == "user"]
    assert len({event["turn_id"] for event in turns}) == len(turns)
    for turn in turns:
        calls = [call["name"] for event in events if event["kind"] == "model"
                 and event["origin"] == "foreground" and event["turn_id"] == turn["turn_id"]
                 for call in event["calls"]]
        expected = ["workflow__restart"] if turn["text"] == "Start guide" else ["workflow__advance"]
        if wrong_foreground_call and turn["text"] == "Next":
            expected = ["workflow__status"]
        assert calls == expected


@pytest.mark.parametrize("fault", ["step_completion", "automatic_completion", "final_inactive"])
def test_packet_cli_rejects_announcements_that_disagree_with_progress(
    packets, runtime_models, tmp_path, monkeypatch, fault,
):
    from procedure_capture_guide_worker.guide.guiding import _Guide

    packet = make_packet(tmp_path)
    guide = yaml.safe_load((_TEST_DATA / "test_procedure_indicator_guide.yaml").read_text())
    guide["task"]["status"] = "draft"
    guide["steps"] = guide["steps"][:1]
    guide["steps"][0]["next"] = None
    guide["steps"][0]["finish_on_complete"] = fault == "automatic_completion"
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(guide))
    guidance = _Guide.guidance
    was_active = False
    injected = 0

    async def inconsistent_progress(self, participant_id):
        nonlocal was_active, injected
        snapshot = await guidance(self, participant_id)
        context = snapshot.context
        was_active |= context["active"]
        # Leave execution and speech intact; simulate a contradictory progress report.
        if fault == "step_completion" and context["state"].get("visible"):
            context = {**context, "state": {**context["state"], "visible": False}}
        elif was_active and not context["active"] and (
            fault == "final_inactive" or (fault == "automatic_completion" and injected == 0)
        ):
            context = {**context, "active": True}
        else:
            return snapshot
        injected += 1
        return replace(snapshot, context=context)

    monkeypatch.setattr(_Guide, "guidance", inconsistent_progress)
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["guide.py", "--packet", str(packet), "--guide", str(path),
                                    "--output", str(output), "--tail-s", "15"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(str(_SCRIPTS / "evaluate_recorded_packet.py"), run_name="__main__")
    assert exit_status.value.code == 1
    assert injected > 0
    result = json.loads((output / "result.json").read_text())
    assert result["passed"] is False
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert not any(event["kind"] == "error" for event in events), "Must fail on progress, not a timeout"
    checks = [event for event in events if event["kind"] == "check"]
    if fault == "final_inactive":
        assert all(event["passed"] for event in checks)
        assert result["checks_not_scored"] == 0
    else:
        assert checks[-1]["action"] == "wait_speech" and not checks[-1]["passed"]
        assert [event["text"] for event in events if event["kind"] == "user"] == ["Start guide"]
    completion = (guide["task"]["complete_message"] if fault != "step_completion"
                  else guide["steps"][0]["messages"]["complete"])
    assert any(event["kind"] == "speech" and completion in event["text"] for event in events)


def test_runtime_provenance_excludes_environment_and_tracks_loaded_sdk(packets, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_SAMPLE / "evaluation/guide"))
    from _runtime_harness import _runtime_hashes as runtime_hashes
    sample = tmp_path / "sample"
    sources = ["yaml/models.json", "yaml/worker.yaml", "yaml/guide-agent-config.yaml",
               "worker/procedure_capture_guide_worker/guide/agent.py",
               "worker/procedure_capture_guide_worker/guide/prompts/guide_vision.txt"]
    excluded = ["worker/.venv/lib/dependency.py", "worker/.venv/lib/prompt.txt",
                "worker/unrelated.py", "worker/sibling_package/other.py"]
    for relative in sources + excluded:
        path = sample / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    sdk = tmp_path / "sdk.py"
    sdk.write_text("before")
    monkeypatch.setitem(sys.modules, "xr_ai_tools.provenance_test", SimpleNamespace(__file__=str(sdk)))
    monkeypatch.setitem(runtime_hashes.__globals__, "_SAMPLE", sample)
    before = runtime_hashes(sample / "yaml/models.json")
    assert set(sources) <= before.keys()
    assert not set(excluded) & before.keys()
    assert "../sdk.py" in before
    sdk.write_text("after")
    after = runtime_hashes(sample / "yaml/models.json")
    assert before["../sdk.py"] != after["../sdk.py"]
    assert all(before[path] == after[path] for path in sources)


def test_generation_cases_have_separate_inputs_and_review_criteria():
    cases = yaml.safe_load((_SAMPLE / "evaluation/guide/authoring_cases.yaml").read_text())
    assert len(cases) == 7
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert set(case) == {"id", "input", "checks"}
        assert set(case["input"]) == {"packet", "captions", "transcripts", "media_access"}
        assert case["input"]["packet"]["status"] == "complete"
        assert case["input"]["captions"] and case["input"]["transcripts"]
        assert case["checks"] and all(isinstance(check, str) and check for check in case["checks"])
        assert "checks" not in case["input"]
