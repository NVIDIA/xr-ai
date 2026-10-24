# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test synthetic guide fixtures and evaluation scoring without GPU services."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from _procedure_guide_helpers import clef_response, stub_decisions
from PIL import Image
from xr_ai_models import ChatResponse, ToolCall

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide"
_TEST_DATA = Path(__file__).resolve().parent
_TEA_EVAL = _SAMPLE / "example-guides/tea-making/evaluation"


@pytest.mark.parametrize("runner,other_runner", [
    ("example-guides/tea-making/evaluation/run_scenario.py",
     "guide-generation-skills"),
    ("evaluation/guide/evaluate_recorded_packet.py",
     "example-guides/tea-making/evaluation"),
])
def test_evaluation_runners_import_shared_runtime_independently(runner, other_runner, tmp_path):
    # A fresh interpreter prevents earlier tests' sys.path additions or cached
    # modules from hiding an accidental dependency between the two runners.
    probe = """
import runpy
import sys
from pathlib import Path
sample, runner, other = map(Path, sys.argv[1:])
sys.path[:0] = [str(sample / 'worker'), str((sample / runner).parent)]
namespace = runpy.run_path(str(sample / runner))
import _runtime_harness
assert Path(_runtime_harness.__file__).resolve() == sample / 'evaluation/guide/_runtime_harness.py'
assert _runtime_harness._SAMPLE == sample
assert str(sample / other) not in sys.path
if 'runtime' in namespace:
    assert namespace['runtime'] is _runtime_harness
else:
    assert namespace['run_evaluation'] is _runtime_harness.run_evaluation
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(_SAMPLE), runner, other_runner],
        cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(_SAMPLE / "worker"))
    monkeypatch.syspath_prepend(str(_TEA_EVAL))
    spec = importlib.util.spec_from_file_location("visual_eval", _TEA_EVAL / "evaluate_visual_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "make_decision", lambda *_: stub_decisions())
    return module


def test_scenes_are_reproducible_and_distinct(evaluator, tmp_path):
    cases, guide = evaluator.prepare(_TEA_EVAL / "fixtures/visual_expectations.yaml", tmp_path / "images")
    assert len(cases) == 25
    assert guide.path == _SAMPLE / "example-guides/tea-making/guide.yaml"
    assert guide.workflow.steps["steep_timer"].trigger.arguments["duration_s"] == 300
    scenes = yaml.safe_load((_TEA_EVAL / "fixtures/image_scenes.yaml").read_text())
    for scene in scenes.values():
        if scene["kind"] == "packet":
            assert set(scene) <= {"kind", "blend", "color", "covered"}
    identification = guide.workflow.steps["identify"]
    assert identification.evidence.pattern is None and identification.evidence.condition
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
        evaluator.prepare(_TEA_EVAL / "fixtures/visual_expectations.yaml", tmp_path / "images")


def test_visual_entry_point_renders_tea_manifest_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(_SAMPLE / "worker"))
    output = tmp_path / "preview"
    result = subprocess.run(
        [sys.executable, "example-guides/tea-making/evaluation/evaluate_visual_checks.py",
         "--manifest", "example-guides/tea-making/evaluation/fixtures/visual_expectations.yaml",
         "--render-only", "--output", str(output)],
        cwd=_SAMPLE, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    provenance = json.loads((output / "provenance.json").read_text())
    assert len(provenance["images"]) == 25
    assert str(_SAMPLE / "example-guides/tea-making/guide.yaml") in provenance["source_sha256"]
    assert str(_TEA_EVAL / "_scene_rendering.py") in provenance["source_sha256"]
    assert str(_TEA_EVAL / "_visual_expectations.py") in provenance["source_sha256"]


@pytest.mark.parametrize("text,available,recognition,gate", [
    ("TARGET | 100 C", True, True, True),
    ("UNKNOWN", True, False, True),
    ("CURRENT | 100 C", True, False, False),
    ("model unavailable", False, None, None),
])
def test_negative_cases_separate_recognition_and_gate(evaluator, text, available, recognition, gate):
    case = {"answer": r"TARGET\s*\|\s*100 C", "accept": False}
    row = evaluator.score(case, text == "CURRENT | 100 C", text, available)
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
    manifest = yaml.safe_load((_TEA_EVAL / "fixtures/visual_expectations.yaml").read_text())
    manifest["cases"] = [
        {"scene": "chamomile", "step": "identify", "answer": ".*chamomile.*", "accept": True},
        {"scene": "green", "step": "identify", "answer": ".*green.*", "accept": False},
    ]
    manifest["guide_path"] = str(_SAMPLE / "example-guides/tea-making/guide.yaml")
    manifest["scenes_file"] = str(_TEA_EVAL / "fixtures/image_scenes.yaml")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    canonical = ["The packet reads chamomile.", "The packet reads green tea."]
    decisions = stub_decisions()
    async def decide(state, questions, **kwargs):
        assert state["condition"] and "condition_met" in questions
        return clef_response("yes" if "chamomile" in state["current_observation"] else "no")
    decisions.decide.side_effect = decide
    monkeypatch.setattr(evaluator, "make_decision", lambda *_: decisions)

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
    summary = await evaluator.evaluate(manifest_path, output,
                                      _SAMPLE / "yaml/models.json", 2, 1)
    assert summary["attempts"] == 4
    assert summary["false_rejects"] == (0 if correct else 2)
    assert summary["gate_correct"] == (4 if correct else 2)
    assert summary["passed"] is correct
    if correct:
        assert summary["recognition_correct"] == 4
        assert all(step["passed"] for step in summary["by_step"].values())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    guide = evaluator.load_guide(_SAMPLE / "example-guides/tea-making/guide.yaml")
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
    summary = await evaluator.evaluate(
        _TEA_EVAL / "fixtures/visual_expectations.yaml", tmp_path / "timeouts",
        _SAMPLE / "yaml/models.json", 1, 1,
    )
    assert summary["unavailable"] == 25
    assert summary["scored"] == summary["gate_correct"] == 0
    assert not summary["passed"]
    fake_vlm.close.assert_awaited_once()


@pytest.mark.parametrize("filename", ["scenarios/step_verification.yaml", "scenarios/full_run.yaml"])
def test_flow_uses_shipped_guide_and_control_contract(evaluator, filename):
    path = _SAMPLE / "example-guides/tea-making/evaluation" / filename
    manifest = yaml.safe_load(path.read_text())
    guide = evaluator.load_guide(path.parent / manifest["guide_path"])
    assert guide.path == _SAMPLE / "example-guides/tea-making/guide.yaml"
    scenes = evaluator.load_scenes(path, manifest)
    assert all(action["scene"] in scenes for action in manifest["actions"] if action["action"] == "image")
    utterances = [action for action in manifest["actions"] if action["action"] == "say"]
    assert any(action["text"] == "Stop guide" for action in utterances)
    waits = [action for action in manifest["actions"] if action["action"] == "wait_speech"]
    for action in manifest["actions"]:
        if "expect" in action:
            re.compile(action["expect"])
    if filename == "scenarios/step_verification.yaml":
        assert sum(action["text"] == "Start guide" for action in utterances) == 1
        assert not any("clock__timer" in action["tools"] for action in utterances)
        assert not any("Five minutes" in action["expect"] for action in waits)
        assert len(waits) == 3
        assert manifest["actions"][-3]["expect"] == "The tea bag is visibly in the water"
        assert manifest["actions"][-2]["text"] == "Stop guide"
        assert manifest["actions"][-1]["expect"] == "inactive"
        cases = evaluator.visual_cases(yaml.safe_load((path.parent / manifest["visual_cases_file"]).read_text()))
        scene = None
        for action in manifest["actions"]:
            if action["action"] == "image":
                scene = action["scene"]
            elif action["action"] == "observe":
                assert "answers" not in action
                assert cases[action["case"]]["scene"] == scene
    else:
        timer = next(action for action in utterances if action["tools"] == ["clock__timer"])
        assert timer["arguments"] == [{}]
        assert sum(action["text"] == "Start guide" for action in utterances) == 2
        completion = next(action for action in waits if "Five minutes" in action["expect"])
        assert completion["timeout_s"] > guide.workflow.steps["steep_timer"].trigger.arguments["duration_s"]
        assert completion["minimum_elapsed_s"] == 295
        quiet = [action for action in manifest["actions"] if action["action"] == "quiet"]
        assert len(quiet) == 1 and quiet[0]["duration_s"] >= 10


@pytest.mark.parametrize("filename,fail_fast", [
    ("example-guides/tea-making/evaluation/scenarios/step_verification.yaml", True),
    ("example-guides/tea-making/evaluation/scenarios/full_run.yaml", True),
    ("example-guides/tea-making/evaluation/scenarios/routing.yaml", False),
])
def test_manifest_declares_dependent_or_independent_checks(filename, fail_fast):
    manifest = yaml.safe_load((_SAMPLE / filename).read_text())
    assert manifest["fail_fast"] is fail_fast


def test_tea_routing_probes_are_independent_and_include_non_control_requests():
    manifest = yaml.safe_load((_TEA_EVAL / "scenarios/routing.yaml").read_text())
    assert manifest["guide_path"] == "../../guide.yaml"
    scenes = yaml.safe_load((_TEA_EVAL / "scenarios" / manifest["scenes_file"]).read_text())
    assert manifest["actions"][0]["scene"] in scenes
    utterances = [action for action in manifest["actions"] if action["action"] == "say"]
    probes = {action["text"]: action for action in utterances}
    for text in ("Don't advance", "My coworker said skip", "What does start guide do?", "Start capture"):
        assert probes[text]["tools"] == []
    for text in ("What color is the tea packet?", "How many tea packets are visible?"):
        assert probes[text]["tools"] == ["current_view"]
    # The first probe tests an inactive guide; each later probe starts afresh.
    assert utterances[0]["tools"] == []
    for restart in utterances[1:-1:2]:
        assert restart["text"] == "Start guide"
        assert restart["tools"] == ["workflow__restart"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["false", 1, None])
async def test_manifest_rejects_non_boolean_fail_fast(evaluator, tmp_path, value):
    spec = importlib.util.spec_from_file_location("guide_eval", _TEA_EVAL / "run_scenario.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "scenario.yaml"
    manifest.write_text(yaml.safe_dump({"fail_fast": value}))
    with pytest.raises(ValueError, match="fail_fast must be a boolean"):
        await module.evaluate(manifest, tmp_path / "results")


def test_inline_scene_manifest_preserves_provenance(evaluator, tmp_path):
    manifest = yaml.safe_load((_TEA_EVAL / "fixtures/visual_expectations.yaml").read_text())
    manifest["scenes"] = evaluator.load_scenes(_TEA_EVAL / "fixtures/visual_expectations.yaml", manifest)
    del manifest["scenes_file"]
    manifest["guide_path"] = str(_SAMPLE / "example-guides/tea-making/guide.yaml")
    path = tmp_path / "inline.yaml"
    path.write_text(yaml.safe_dump(manifest))
    cases, _guide = evaluator.prepare(path, tmp_path / "output")
    provenance = json.loads((tmp_path / "output/provenance.json").read_text())
    assert provenance["scenes"] == manifest["scenes"]
    assert str(path) in provenance["source_sha256"]
    assert len(cases) == len(provenance["images"]) == 25


@pytest.mark.parametrize("accepted", [False, True])
def test_recognition_and_clef_decision_are_scored_independently(evaluator, accepted):
    case = {"answer": ".*chamomile.*", "accept": True}
    row = evaluator.score(case, accepted, "The packet reads chamomile.", True)
    assert row["recognition"]
    assert row["accepted"] is accepted
    assert row["gate_correct"] is accepted




def test_recognition_normalization_does_not_relax_evidence_gate(evaluator):
    case = {"answer": "BREAKFAST BLACK", "accept": True}
    row = evaluator.score(case, False, "BREAKFAST\nBLACK", True)
    assert row["recognition"]
    assert not row["accepted"]


@pytest.fixture
def runtime_evaluator(evaluator, monkeypatch):
    spec = importlib.util.spec_from_file_location("guide_eval", _TEA_EVAL / "run_scenario.py")
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
    monkeypatch.setattr(module.runtime.xr_ai_models, "make_llm", Mock(return_value=llm))
    monkeypatch.setattr(module.runtime.xr_ai_models, "make_decision", lambda *_: stub_decisions())
    monkeypatch.setattr(module.runtime.xr_ai_models, "make_vlm", Mock(return_value=vlm))
    return module


def timer_scenario(tmp_path, actions):
    guide = yaml.safe_load((_TEST_DATA / "test_procedure_indicator_guide.yaml").read_text())
    guide["task"]["start_step"] = "wait"
    guide["steps"] = [guide["steps"][1]]
    guide["steps"][0]["trigger"]["arguments"]["duration_s"] = 1
    guide["steps"][0]["trigger"]["interval_s"] = 0.05
    (tmp_path / "test.guide.yaml").write_text(yaml.safe_dump(guide))
    manifest = tmp_path / "scenario.yaml"
    manifest.write_text(yaml.safe_dump({"guide_path": "test.guide.yaml",
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
    actions.append({"action": "quiet", "duration_s": 1.3})
    manifest = timer_scenario(tmp_path, actions)
    assert await runtime_evaluator.evaluate(manifest, tmp_path / "output") is expected


@pytest.mark.asyncio
async def test_quiet_includes_speech_during_image_switch(runtime_evaluator, tmp_path, monkeypatch):
    module = runtime_evaluator
    runtime = None
    publish = module.runtime.AgentRuntime.publish

    async def remember_runtime(self, *args, **kwargs):
        nonlocal runtime
        runtime = self
        return await publish(self, *args, **kwargs)

    monkeypatch.setattr(module.runtime.AgentRuntime, "publish", remember_runtime)
    publish_frame = module.runtime.Camera.publish
    injected = False

    async def noisy_frame(self, **kwargs):
        nonlocal injected
        await publish_frame(self, **kwargs)
        if not injected:
            injected = True
            await runtime.publish(
                module.runtime.VOICE_OUTPUT_TOPIC,
                module.runtime.ResponseOutput(kind="update", text="Obsolete completion"),
                participant_id=module.runtime._PID,
            )

    monkeypatch.setattr(module.runtime.Camera, "publish", noisy_frame)
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
@pytest.mark.parametrize("intervening", [
    {"action": "wait_speech", "expect": "done"},
    {"action": "observe", "answers": ["YES"]},
    {"action": "quiet", "duration_s": 0.1},
])
async def test_quiet_rejects_intervening_checks_before_model_work(runtime_evaluator, tmp_path, intervening):
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text(yaml.safe_dump({"actions": [
        {"action": "say", "text": "Stop guide"},
        {"action": "image", "scene": "blue"},
        intervening,
        {"action": "image", "scene": "blue"},
        {"action": "quiet", "duration_s": 1.3},
    ]}))
    with pytest.raises(ValueError, match="action 4: quiet requires.*only image actions between"):
        await runtime_evaluator.evaluate(manifest, tmp_path / "output")
    runtime_evaluator.runtime.xr_ai_models.make_llm.assert_not_called()
    runtime_evaluator.runtime.xr_ai_models.make_vlm.assert_not_called()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("images", [0, 1, 2])
def test_quiet_accepts_only_images_after_new_say(runtime_evaluator, images):
    runtime_evaluator.validate_actions({"actions": [
        {"action": "say", "text": "Start guide"},
        {"action": "wait_speech", "expect": "done"},
        {"action": "say", "text": "Stop guide"},
        *[{"action": "image", "scene": "blue"} for _ in range(images)],
        {"action": "quiet", "duration_s": 1.3},
    ]}, {})


def test_quiet_requires_a_say_even_after_images(runtime_evaluator):
    with pytest.raises(ValueError, match="action 1: quiet requires"):
        runtime_evaluator.validate_actions({"actions": [
            {"action": "image", "scene": "blue"},
            {"action": "quiet", "duration_s": 1.3},
        ]}, {})


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
    runtime_evaluator.runtime.xr_ai_models.make_llm.assert_not_called()
    runtime_evaluator.runtime.xr_ai_models.make_vlm.assert_not_called()
    assert not (tmp_path / "output").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [(" \nno\t", True), ("YES", False)])
async def test_runtime_observe_uses_shared_visual_case(runtime_evaluator, tmp_path, answer, expected):
    module = runtime_evaluator
    module.runtime.xr_ai_models.make_vlm.return_value.ask_images.return_value = SimpleNamespace(content=answer)
    guide = yaml.safe_load((_TEST_DATA / "test_procedure_indicator_guide.yaml").read_text())
    guide["steps"] = guide["steps"][:1]
    guide["steps"][0]["next"] = None
    (tmp_path / "test.guide.yaml").write_text(yaml.safe_dump(guide))
    cases = {"cases": [{"scene": "red", "step": "indicator", "answer": "NO", "accept": False}]}
    (tmp_path / "cases.yaml").write_text(yaml.safe_dump(cases))
    manifest = tmp_path / "scenario.yaml"
    manifest.write_text(yaml.safe_dump({
        "guide_path": "test.guide.yaml", "scenes": {"red": {"rectangle": "red"}},
        "visual_cases_file": "cases.yaml", "actions": [
            {"action": "image", "scene": "red"}, {"action": "say", "text": "Start guide"},
            {"action": "observe", "case": "red", "count": 1},
        ],
    }))
    assert await module.evaluate(manifest, tmp_path / "output") is expected
    assert json.loads((tmp_path / "output/visual_cases.json").read_text()) == {"red": cases["cases"][0]}
