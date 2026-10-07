# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline contract tests for direct, model-authored guide YAML, not generation."""

import copy
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/sop-sample"
_REFERENCES = _SAMPLE / "skills/recording-to-guide/references"
sys.path.insert(0, str(_SAMPLE / "worker"))

from sop_sample_worker._validate_guide import main  # noqa: E402
from sop_sample_worker._workflow_spec import load_workflow, parse_workflow  # noqa: E402


@pytest.fixture
def visual():
    return yaml.safe_load((_REFERENCES / "example.guide.yaml").read_text())


def test_visual_example_uses_same_writable_boolean_and_closed_answers():
    workflow = load_workflow(_REFERENCES / "example.guide.yaml")
    step = workflow.steps[workflow.start_step]
    assert workflow.status == "draft"
    assert not workflow.runnable
    assert not step.is_complete(workflow.initial_state())
    assert step.evidence.commit == step.complete_when == {step.writes[0]: True}
    assert step.is_complete({**workflow.initial_state(), **step.evidence.commit})
    assert step.evidence.consecutive >= 2
    assert re.fullmatch(step.evidence.pattern, "ready")
    for response in ("not ready", "unclear", "ready, but still held", "unavailable"):
        assert not re.fullmatch(step.evidence.pattern, response)
    assert step.next_step is None
    assert step.trigger.result_field == "text"


def test_timer_example_requires_live_setup_before_elapsed_only_wait():
    workflow = load_workflow(_REFERENCES / "timer.guide.yaml")
    setup = workflow.steps[workflow.start_step]
    wait = workflow.steps[setup.next_step]
    assert setup.trigger.function == "current_view"
    assert setup.agent.tools == ("clock__now",)
    assert not setup.evidence.commit
    assert setup.complete_on_skip
    assert not setup.state_on_skip
    assert set(setup.writes) == {"placement_done", "started_us"}
    assert setup.complete_when == {"placement_done": True}
    assert workflow.initial_state()["started_us"] == 0
    assert wait.trigger.function == "clock__timer"
    assert wait.trigger.arguments == {"started_at_us": "$state.started_us", "duration_s": 10}
    assert wait.reads == ("started_us",)
    assert wait.trigger.result_field == "expired"
    assert wait.agent.tools == ()
    assert wait.voice.tools == ("clock__timer",)
    assert wait.evidence.commit == wait.complete_when == {"wait_done": True}
    assert re.fullmatch(wait.evidence.pattern, "true")
    assert not re.fullmatch(wait.evidence.pattern, "false")


@pytest.mark.parametrize("mutation,diagnostic", [
    (lambda root: root.update(review_blockers=[]), "unknown fields"),
    (lambda root: root["steps"][0]["evidence"].update(commit={"done_field": True}), "writable"),
    (lambda root: root["steps"][0].update(complete_when={"workpiece_positioned": "true"}), "boolean"),
    (lambda root: root["steps"][0]["trigger"].update(function="sequence_view"), "unsupported trigger"),
    (lambda root: root["steps"][0]["trigger"].update(arguments={}), "question"),
    (lambda root: root["steps"][0]["trigger"]["arguments"].update(max_frames=4), "max_frames"),
    (lambda root: root["steps"][0]["trigger"]["arguments"].update(question="x" * 501), "500"),
    (lambda root: root["steps"][0]["trigger"].update(result_field="ready"), "result field"),
    (lambda root: root["steps"][0]["agent"].update(tools=["workflow__commit"]), "unsupported tools"),
    (lambda root: root["steps"][0].update(next="unknown"), "unknown next"),
    (lambda root: root["steps"][0].update(next="position_workpiece"), "cycle"),
    (lambda root: root["steps"].append({**root["steps"][0], "id": "unreachable"}), "unreachable"),
])
def test_validator_rejects_model_authoring_errors(visual, mutation, diagnostic):
    mutation(visual)
    with pytest.raises(ValueError, match=diagnostic):
        parse_workflow(yaml.safe_dump(visual))


@pytest.mark.parametrize("argument,value", [
    ("duration_s", 0), ("duration_s", True), ("duration_s", 1.5),
    ("duration_s", "15"), ("started_at_us", 0),
    ("started_at_us", "$state.undeclared"),
])
def test_timer_request_errors(argument, value):
    raw = yaml.safe_load((_REFERENCES / "timer.guide.yaml").read_text())
    raw["steps"][1]["trigger"]["arguments"][argument] = value
    with pytest.raises(ValueError):
        parse_workflow(yaml.safe_dump(raw))


def test_review_blocked_recipe_never_matches(visual):
    schema = (_REFERENCES / "packet-and-guide-schema.md").read_text()
    recipe = re.search(r"```yaml\n(.*?)\n```", schema, re.DOTALL).group(1)
    visual["steps"][0].update(yaml.safe_load(recipe))
    workflow = parse_workflow(yaml.safe_dump(visual))
    step = workflow.steps[workflow.start_step]
    assert not step.evidence.commit
    for observation in ("", "ready", "true", "verified", "review_required", "(?!)"):
        assert not re.fullmatch(step.evidence.pattern, observation)
    assert not step.is_complete(workflow.initial_state())


def test_snapshot_parsing_and_state_are_independent(visual):
    snapshot = yaml.safe_dump(visual).encode()
    workflow = parse_workflow(snapshot)
    visual["task"]["name"] = "Changed after capture"
    assert workflow.name == "Arrange Workpiece"
    state = workflow.initial_state()
    state["workpiece_positioned"] = True
    assert not workflow.initial_state()["workpiece_positioned"]


def test_validator_checks_all_inputs_without_modifying_any(tmp_path, visual, capsys):
    good = tmp_path / "draft.guide.yaml"
    good.write_text(yaml.safe_dump(visual))
    approved = tmp_path / "approved.guide.yaml"
    other = copy.deepcopy(visual)
    other["task"]["status"] = "approved"
    approved.write_text(yaml.safe_dump(other))
    bad = tmp_path / "invalid.guide.yaml"
    bad.write_text("schema_version: [")
    originals = {path: path.read_bytes() for path in (good, approved, bad)}
    assert main([str(good), str(bad), str(approved), str(tmp_path / "missing")]) == 1
    stdout, stderr = capsys.readouterr()
    assert str(good) in stdout and str(approved) in stdout
    assert str(bad) in stderr and "missing" in stderr
    assert {path: path.read_bytes() for path in originals} == originals
    assert main([str(good), str(approved)]) == 0


def test_module_cli_exit_code_and_read_only_behavior(tmp_path, visual):
    path = tmp_path / "guide.yaml"
    path.write_text(yaml.safe_dump(visual))
    before = path.read_bytes()
    env = {**os.environ, "PYTHONPATH": str(_SAMPLE / "worker")}
    command = [sys.executable, "-m", "sop_sample_worker._validate_guide", str(path)]
    success = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert success.returncode == 0, success.stderr
    assert "draft, 1 steps" in success.stdout
    assert path.read_bytes() == before
    path.write_text("not: a guide")
    failure = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert failure.returncode == 1
    assert "invalid:" in failure.stderr
    assert "Traceback" not in failure.stderr
    assert path.read_text() == "not: a guide"
