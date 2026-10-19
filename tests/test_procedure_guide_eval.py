# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test synthetic guide fixtures and evaluation scoring without GPU services."""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from PIL import Image

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
    assert guide.workflow.steps["steep_timer"].trigger.arguments["duration_s"] == 300
    scenes = yaml.safe_load((_SAMPLE / "eval/tea_scenes.yaml").read_text())
    for scene in scenes.values():
        if scene["kind"] == "packet":
            assert set(scene) <= {"kind", "blend", "color", "covered"}
    identification = guide.workflow.steps["identify"]
    assert identification.evidence.pattern == r"(?i)^CHAMOMILE$"
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
async def test_visual_evaluator_uses_guide_question_not_answer(evaluator, tmp_path, monkeypatch):
    requests = []
    fake_vlm = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(evaluator, "make_vlm", lambda *_args: fake_vlm)

    class FakeTool:
        def __init__(self, *, images, vlm, system_prompt):
            self.images = images
            assert vlm is fake_vlm
            assert system_prompt == evaluator._PROMPT.read_text()

        async def execute(self, request):
            requests.append(request)
            assert self.images.resolve(request.image).startswith(b"\x89PNG")
            assert request.image.uri.startswith("xr-image://")
            return SimpleNamespace(text="UNKNOWN", available=True)

    monkeypatch.setattr(evaluator, "ImageQueryTool", FakeTool)
    output = tmp_path / "visual"
    summary = await evaluator.evaluate(_SAMPLE / "eval/tea_visual_cases.yaml", output,
                                      _SAMPLE / "yaml/models.json", 2, 1)
    assert summary["attempts"] == 50
    assert summary["false_rejects"] == 10
    assert summary["gate_correct"] == 40
    assert not summary["passed"]
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    guide = evaluator.select_guide(_SAMPLE / "eval/fixtures", "Chamomile Evaluation")
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
def test_flow_uses_brand_neutral_guide_and_control_contract(evaluator, filename):
    path = _SAMPLE / "eval" / filename
    manifest = yaml.safe_load(path.read_text())
    guide = evaluator.select_guide(path.parent / manifest["guides_dir"], manifest["guide"])
    assert guide.path == _SAMPLE / "eval/fixtures/chamomile.guide.yaml"
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
