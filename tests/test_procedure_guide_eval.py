# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test synthetic guide fixtures and evaluation scoring without GPU services."""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from PIL import Image
from xr_ai_models import ChatResponse, ToolCall

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide"


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(_SAMPLE / "worker"))
    monkeypatch.syspath_prepend(str(_SAMPLE / "eval"))
    spec = importlib.util.spec_from_file_location("tea_visual_eval", _SAMPLE / "eval/tea_visual.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenes_are_reproducible_and_distinct(evaluator, tmp_path):
    cases, guide = evaluator.prepare(_SAMPLE / "eval/tea_visual_cases.yaml", tmp_path / "images")
    assert len(cases) == 25
    assert guide.path == _SAMPLE / "guides/tea-making.guide.yaml"
    assert guide.workflow.steps["steep_timer"].trigger.arguments["duration_s"] == 300
    scenes = yaml.safe_load((_SAMPLE / "eval/tea_scenes.yaml").read_text())
    for scene in scenes.values():
        if scene["kind"] == "packet":
            assert set(scene) <= {"kind", "blend", "color", "covered"}
    identification = guide.workflow.steps["identify"]
    assert identification.evidence.pattern == r"(?i)^YES$"
    pixels = set()
    for case in cases:
        rendered = evaluator.render_scene(scenes[case["scene"]])
        with Image.open(tmp_path / "images" / f'{case["scene"]}.png') as saved:
            assert rendered.tobytes() == saved.tobytes()
            assert saved.size == (800, 600)
            pixels.add(saved.tobytes())
    assert len(pixels) == len(cases)
    provenance = json.loads((tmp_path / "images/provenance.json").read_text())
    assert len(provenance["images"]) == len(cases)
    assert provenance["guide_sha256"] == guide.sha256
    with pytest.raises(FileExistsError):
        evaluator.prepare(_SAMPLE / "eval/tea_visual_cases.yaml", tmp_path / "images")


@pytest.mark.parametrize("text,available,recognition,gate", [
    ("TARGET | 100 C", True, True, True),
    ("UNKNOWN", True, False, True),
    ("CURRENT | 100 C", True, False, False),
    ("model unavailable", False, None, None),
])
def test_negative_cases_separate_recognition_and_gate(evaluator, text, available, recognition, gate):
    case = {"answer": r"TARGET\s*\|\s*100 C", "accept": False}
    row = evaluator.score(case, r"CURRENT\s*\|\s*100 C", text, available)
    assert row["recognition"] is recognition
    assert row["gate_correct"] is gate


def test_empty_and_unavailable_runs_cannot_pass(evaluator):
    assert not evaluator.summarize([])["passed"]
    assert evaluator.summarize([{"status": "unavailable"}]) == {
        "attempts": 1, "scored": 0, "unavailable": 1,
        "recognition_correct": 0, "gate_correct": 0,
        "false_accepts": 0, "false_rejects": 0, "passed": False,
    }


def test_schema_and_renderer_fail_loudly(evaluator):
    with pytest.raises(ValueError, match="unknown scene kind"):
        evaluator.render_scene({"kind": "typo"})
    with pytest.raises(ValueError, match="unknown cup state"):
        evaluator.render_scene({"kind": "cup", "state": "typo"})


@pytest.mark.asyncio
@pytest.mark.parametrize("correct", [False, True])
async def test_visual_evaluator_uses_guide_question_not_answer(evaluator, tmp_path, monkeypatch, correct):
    requests = []
    fake_vlm = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(evaluator, "make_vlm", lambda *_args: fake_vlm)
    manifest = yaml.safe_load((_SAMPLE / "eval/tea_visual_cases.yaml").read_text())
    scenes = evaluator.load_scenes(_SAMPLE / "eval/tea_visual_cases.yaml", manifest)
    canonical = []
    for case in manifest["cases"]:
        if case["step"] == "heat_water" and case["answer"] != "UNKNOWN":
            reading = scenes[case["scene"]]["readings"][-1]
            canonical.append(f'{reading["label"]} | {reading["value"]}')
        else:
            canonical.append(case["answer"])

    class FakeTool:
        def __init__(self, *, images, vlm, system_prompt):
            self.images = images
            assert vlm is fake_vlm
            assert system_prompt == evaluator._PROMPT.read_text()

        async def execute(self, request):
            requests.append(request)
            assert self.images.resolve(request.image).startswith(b"\x89PNG")
            assert request.image.uri.startswith("xr-image://")
            text = canonical[(len(requests) - 1) % len(canonical)] if correct else "UNKNOWN"
            return SimpleNamespace(text=text, available=True)

    monkeypatch.setattr(evaluator, "ImageQueryTool", FakeTool)
    output = tmp_path / "visual"
    summary = await evaluator.evaluate(_SAMPLE / "eval/tea_visual_cases.yaml", output,
                                      _SAMPLE / "yaml/models.json", 2, 1)
    assert summary["attempts"] == 50
    assert summary["false_rejects"] == (0 if correct else 10)
    assert summary["gate_correct"] == (50 if correct else 40)
    assert summary["passed"] is correct
    if correct:
        assert summary["recognition_correct"] == 50
        assert all(step["passed"] for step in summary["by_step"].values())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    guide = evaluator.select_guide(_SAMPLE / "guides", "Tea Making")
    for row, request in zip(rows, requests, strict=True):
        assert request.query == guide.workflow.steps[row["step"]].trigger.arguments["question"]
    fake_vlm.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_timeout_is_not_a_correct_negative(evaluator, tmp_path, monkeypatch):
    fake_vlm = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(evaluator, "make_vlm", lambda *_args: fake_vlm)

    class TimeoutTool:
        def __init__(self, **_kwargs):
            pass

        async def execute(self, _request):
            raise TimeoutError("simulated")

    monkeypatch.setattr(evaluator, "ImageQueryTool", TimeoutTool)
    summary = await evaluator.evaluate(_SAMPLE / "eval/tea_visual_cases.yaml", tmp_path / "timeouts",
                                      _SAMPLE / "yaml/models.json", 1, 1)
    assert summary["unavailable"] == 25
    assert summary["scored"] == summary["gate_correct"] == 0
    assert not summary["passed"]
    fake_vlm.close.assert_awaited_once()


@pytest.mark.parametrize("filename", ["tea_guide.yaml", "tea_guide_complete.yaml"])
def test_flow_uses_shipped_guide_and_control_contract(evaluator, filename):
    path = _SAMPLE / "eval" / filename
    manifest = yaml.safe_load(path.read_text())
    guide = evaluator.select_guide(path.parent / manifest["guides_dir"], manifest["guide"])
    assert guide.path == _SAMPLE / "guides/tea-making.guide.yaml"
    scenes = evaluator.load_scenes(path, manifest)
    assert all(action["scene"] in scenes for action in manifest["actions"] if action["action"] == "image")
    utterances = [action for action in manifest["actions"] if action["action"] == "say"]
    timer = next(action for action in utterances if action["tools"] == ["clock__timer"])
    assert timer["arguments"] == [{}]
    assert any(action["text"] == "Stop guide" for action in utterances)
    assert sum(action["text"] == "Start guide" for action in utterances) == 2
    waits = [action for action in manifest["actions"] if action["action"] == "wait_speech"]
    completion = next(action for action in waits if "Five minutes" in action["expect"])
    assert completion["timeout_s"] > guide.workflow.steps["steep_timer"].trigger.arguments["duration_s"]
    for action in manifest["actions"]:
        if "expect" in action:
            re.compile(action["expect"])
    if filename == "tea_guide.yaml":
        cases = evaluator.visual_cases(yaml.safe_load((path.parent / manifest["visual_cases_file"]).read_text()))
        scene = None
        for action in manifest["actions"]:
            if action["action"] == "image":
                scene = action["scene"]
            elif action["action"] == "observe":
                assert "answers" not in action
                assert cases[action["case"]]["scene"] == scene
    assert all(action["duration_s"] >= 10 for action in manifest["actions"] if action["action"] == "quiet")


@pytest.mark.parametrize("filename,fail_fast", [
    ("tea_guide.yaml", True),
    ("tea_guide_complete.yaml", True),
    ("guide_cases.yaml", True),
    ("guide_routing.yaml", False),
])
def test_manifest_declares_dependent_or_independent_checks(filename, fail_fast):
    manifest = yaml.safe_load((_SAMPLE / "eval" / filename).read_text())
    assert manifest["fail_fast"] is fail_fast


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["false", 1, None])
async def test_manifest_rejects_non_boolean_fail_fast(evaluator, tmp_path, value):
    spec = importlib.util.spec_from_file_location("guide_eval", _SAMPLE / "eval/guide.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "scenario.yaml"
    manifest.write_text(yaml.safe_dump({"fail_fast": value}))
    with pytest.raises(ValueError, match="fail_fast must be a boolean"):
        await module.evaluate(manifest, tmp_path / "results")


def test_inline_scene_manifest_preserves_provenance(evaluator, tmp_path):
    manifest = yaml.safe_load((_SAMPLE / "eval/tea_visual_cases.yaml").read_text())
    manifest["scenes"] = evaluator.load_scenes(_SAMPLE / "eval/tea_visual_cases.yaml", manifest)
    del manifest["scenes_file"]
    manifest["guides_dir"] = str(_SAMPLE / "guides")
    path = tmp_path / "inline.yaml"
    path.write_text(yaml.safe_dump(manifest))
    cases, _guide = evaluator.prepare(path, tmp_path / "output")
    provenance = json.loads((tmp_path / "output/provenance.json").read_text())
    assert provenance["scenes"] == manifest["scenes"]
    assert str(path) in provenance["source_sha256"]
    assert len(cases) == len(provenance["images"]) == 25


@pytest.mark.parametrize("scene,text,recognition,accepted", [
    ("current100", "CURRENT | 100.0 C", True, True),
    ("current100", "CURRENT | 100℃", True, True),
    ("actual212", "ACTUAL | 212.00℉", True, True),
    ("current100", "CURRENT\n|\t100 C", True, True),
    ("current100", "CURRENT | 100.1 C", False, False),
    ("current100", "TARGET | 100 C", False, False),
    ("target100", "CURRENT | 100 C", False, True),
    ("chamomile", "NO CHAMOMILE", False, False),
    ("chamomile", "YES\n", True, True),
])
def test_recognition_and_runtime_gate_have_distinct_contracts(evaluator, scene, text, recognition, accepted):
    manifest = yaml.safe_load((_SAMPLE / "eval/tea_visual_cases.yaml").read_text())
    case = evaluator.visual_cases(manifest)[scene]
    guide = evaluator.select_guide(_SAMPLE / "eval" / manifest["guides_dir"], manifest["guide"])
    pattern = guide.workflow.steps[case["step"]].evidence.pattern
    row = evaluator.score(case, pattern, text, True)
    assert row["recognition"] is recognition
    assert row["accepted"] is accepted
    assert row["gate_correct"] is (accepted == case["accept"])


def test_recognition_normalization_does_not_relax_evidence_gate(evaluator):
    case = {"answer": "BREAKFAST BLACK", "accept": True}
    row = evaluator.score(case, "^BREAKFAST BLACK$", "BREAKFAST\nBLACK", True)
    assert row["recognition"]
    assert not row["accepted"]


@pytest.fixture
def runtime_evaluator(evaluator, monkeypatch):
    spec = importlib.util.spec_from_file_location("guide_eval", _SAMPLE / "eval/guide.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def chat(messages, **kwargs):
        if messages[-1].role == "tool":
            return ChatResponse("The ten-second wait has started. It is still running.", None, None, "stop", {})
        query = json.loads(messages[1].content)["request"]
        tool = {"Start guide": "workflow__restart", "Stop guide": "workflow__reset",
                "Timer?": "clock__timer"}[query]
        return ChatResponse("", None, [ToolCall(id="eval", name=tool, arguments="{}")], "tool_calls", {})

    llm = Mock(chat=AsyncMock(side_effect=chat), close=AsyncMock())
    vlm = Mock(ask_images=AsyncMock(return_value=SimpleNamespace(content="NO")), close=AsyncMock())
    monkeypatch.setattr(module, "make_llm", Mock(return_value=llm))
    monkeypatch.setattr(module, "make_vlm", Mock(return_value=vlm))
    return module


def timer_scenario(tmp_path, actions):
    guide = yaml.safe_load((_SAMPLE / "eval/fixtures/indicator.guide.yaml").read_text())
    guide["task"]["start_step"] = "wait"
    guide["steps"] = [guide["steps"][1]]
    guide["steps"][0]["trigger"]["arguments"]["duration_s"] = 1
    guide["steps"][0]["trigger"]["interval_s"] = 0.05
    (tmp_path / "test.guide.yaml").write_text(yaml.safe_dump(guide))
    manifest = tmp_path / "scenario.yaml"
    manifest.write_text(yaml.safe_dump({"guides_dir": ".", "guide": "Indicator Check",
                                      "fail_fast": True, "actions": actions}))
    return manifest


@pytest.mark.asyncio
@pytest.mark.parametrize("origin,minimum,expected", [
    ("The ten-second wait has started", 0.5, True),
    ("The ten-second wait has started", 30, False),
    ("This announcement never happened", 0.5, False),
])
async def test_runtime_timing_check_passes_and_fails(runtime_evaluator, tmp_path, origin, minimum, expected):
    manifest = timer_scenario(tmp_path, [
        {"action": "say", "text": "Start guide"},
        {"action": "say", "text": "Timer?"},
        {"action": "wait_speech", "expect": "Indicator check finished", "timeout_s": 3,
         "after_speech": origin, "minimum_elapsed_s": minimum},
    ])
    assert await runtime_evaluator.evaluate(manifest, tmp_path / "output") is expected
    result = json.loads((tmp_path / "output/result.json").read_text())
    assert result["checks_passed"] == (3 if expected else 2)


def test_timer_origin_is_first_announcement_in_current_run(runtime_evaluator):
    def speech(time, text):
        return {"kind": "speech", "elapsed_s": time, "text": text}

    restart = {"kind": "model", "calls": [{"name": "workflow__restart"}]}
    events = [restart, speech(0, "Timer started"), speech(300, "Finished"), restart,
              speech(400, "Timer started"), speech(450, "Timer started and is running"), speech(700, "Finished")]
    assert runtime_evaluator.announcement_elapsed(events, "Timer started", "Finished") == 300
    assert runtime_evaluator.announcement_elapsed(events + [restart], "Timer started", "Finished") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped,expected", [(True, True), (False, False)])
async def test_quiet_detects_runtime_announcement(runtime_evaluator, tmp_path, stopped, expected):
    actions = [{"action": "say", "text": "Start guide"}]
    if stopped:
        actions.append({"action": "say", "text": "Stop guide"})
    actions.append({"action": "quiet", "duration_s": 1.3 if not stopped else 0.05})
    manifest = timer_scenario(tmp_path, actions)
    assert await runtime_evaluator.evaluate(manifest, tmp_path / "output") is expected


@pytest.mark.asyncio
async def test_quiet_includes_speech_during_image_switch(runtime_evaluator, tmp_path, monkeypatch):
    module = runtime_evaluator
    runtime = None
    publish = module.AgentRuntime.publish

    async def remember_runtime(self, *args, **kwargs):
        nonlocal runtime
        runtime = self
        return await publish(self, *args, **kwargs)

    monkeypatch.setattr(module.AgentRuntime, "publish", remember_runtime)
    publish_frame = module.Camera.publish
    injected = False

    async def noisy_frame(self, **kwargs):
        nonlocal injected
        await publish_frame(self, **kwargs)
        if not injected:
            injected = True
            await runtime.publish(module.VOICE_OUTPUT_TOPIC, module.VoiceOutput(text="Obsolete completion"),
                                  participant_id=module._PID)

    monkeypatch.setattr(module.Camera, "publish", noisy_frame)
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "blue").save(image)
    manifest = timer_scenario(tmp_path, [
        {"action": "say", "text": "Start guide"}, {"action": "say", "text": "Stop guide"},
        {"action": "image", "path": "image.png"}, {"action": "quiet", "duration_s": 0.05},
    ])
    assert not await module.evaluate(manifest, tmp_path / "output")
    events = [json.loads(line) for line in (tmp_path / "output/events.jsonl").read_text().splitlines()]
    assert any(event["kind"] == "speech" and event["text"] == "Obsolete completion" for event in events)
    assert not next(event["passed"] for event in events if event["kind"] == "check" and event["action"] == "quiet")


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [
    {"action": "wait_speech", "expect": "done", "after_speech": "start"},
    {"action": "wait_speech", "expect": "done", "minimum_elapsed_s": 1},
    {"action": "wait_speech", "expect": "done", "after_speech": "start", "minimum_elapsed_s": -1},
    {"action": "wait_speech", "expect": "["},
    {"action": "wait_speech", "expect": "done", "timeout_s": float("inf")},
    {"action": "quiet"}, {"action": "quiet", "duration_s": True},
    {"action": "quiet", "duration_s": float("nan")}, {"action": "quiet", "duration_s": 0},
    {"action": "observe", "case": "missing"},
])
async def test_invalid_actions_fail_before_model_work(runtime_evaluator, tmp_path, action):
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text(yaml.safe_dump({"actions": [{"action": "say", "text": "Start guide"}, action]}))
    with pytest.raises(ValueError, match="action 1"):
        await runtime_evaluator.evaluate(manifest, tmp_path / "output")
    runtime_evaluator.make_llm.assert_not_called()
    runtime_evaluator.make_vlm.assert_not_called()
    assert not (tmp_path / "output").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [(" \nno\t", True), ("YES", False)])
async def test_runtime_observe_uses_shared_visual_case(runtime_evaluator, tmp_path, answer, expected):
    module = runtime_evaluator
    module.make_vlm.return_value.ask_images.return_value = SimpleNamespace(content=answer)
    guide = yaml.safe_load((_SAMPLE / "eval/fixtures/indicator.guide.yaml").read_text())
    guide["steps"] = guide["steps"][:1]
    guide["steps"][0]["next"] = None
    (tmp_path / "test.guide.yaml").write_text(yaml.safe_dump(guide))
    cases = {"cases": [{"scene": "red", "step": "indicator", "answer": "NO", "accept": False}]}
    (tmp_path / "cases.yaml").write_text(yaml.safe_dump(cases))
    manifest = tmp_path / "scenario.yaml"
    manifest.write_text(yaml.safe_dump({
        "guides_dir": ".", "guide": "Indicator Check", "scenes": {"red": {"rectangle": "red"}},
        "visual_cases_file": "cases.yaml", "actions": [
            {"action": "image", "scene": "red"}, {"action": "say", "text": "Start guide"},
            {"action": "observe", "case": "red", "count": 1},
        ],
    }))
    assert await module.evaluate(manifest, tmp_path / "output") is expected
    assert json.loads((tmp_path / "output/visual_cases.json").read_text()) == {"red": cases["cases"][0]}
