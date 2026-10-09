# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline contract tests for direct, model-authored guide YAML, not generation."""

import copy
import os
import re
import shutil
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


def test_installed_skill_pair_resolves_shared_contract_and_references(tmp_path):
    # Test the documented sibling installation, not links that only work from
    # the repository root. Evaluation shares the authoring contract and examples.
    for name in ("recording-to-guide", "evaluate-guide"):
        shutil.copytree(_SAMPLE / "skills" / name, tmp_path / name)
    for path in tmp_path.rglob("*.md"):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#", 1)[0]).is_file(), (path, target)
    contracts = list(tmp_path.rglob("packet-and-guide-schema.md"))
    assert len(contracts) == 1
    for name in ("recording-to-guide", "evaluate-guide"):
        skill = tmp_path / name
        metadata = yaml.safe_load((skill / "SKILL.md").read_text().split("---", 2)[1])
        assert metadata["name"] == name
        interface = yaml.safe_load((skill / "agents/openai.yaml").read_text())["interface"]
        assert f"${name}" in interface["default_prompt"]


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
    assert step.next_step == "attach_clips"
    assert step.trigger.result_field == "text"


def test_visual_example_requires_fresh_evidence_for_each_linked_step():
    workflow = load_workflow(_REFERENCES / "example.guide.yaml")
    state = workflow.initial_state()
    visited = []
    step_id = workflow.start_step
    while step_id is not None:
        step = workflow.steps[step_id]
        assert not step.is_complete(state), "Earlier completion must not satisfy a later step"
        assert step.trigger.function == "current_view"
        assert step.evidence.commit == step.complete_when == {step.writes[0]: True}
        for response in ("not ready", "unclear", "unavailable"):
            assert not re.fullmatch(step.evidence.pattern, response)
        state.update(step.evidence.commit)
        assert step.is_complete(state)
        visited.append(step_id)
        step_id = step.next_step
    assert visited == ["position_workpiece", "attach_clips"]
    assert len({tuple(step.writes) for step in workflow.steps.values()}) == len(visited)


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


def test_unresolved_identity_example_preserves_action_without_false_success():
    workflow = load_workflow(_REFERENCES / "review-blocked.guide.yaml")
    step = workflow.steps[workflow.start_step]
    assert workflow.status == "draft"
    assert not workflow.runnable
    assert not step.is_complete(workflow.initial_state())
    assert step.trigger.function == "current_view"
    assert step.agent.tools == ()
    assert step.voice.tools == ()
    assert not step.evidence.commit
    assert step.complete_on_skip
    assert not step.state_on_skip
    for observation in ("ready", "true", "unclear", "adapter fitted", "(?!)"):
        assert not re.fullmatch(step.evidence.pattern, observation)


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
    ("state", "workpiece_positioned", "description"),
])
@pytest.mark.parametrize("value", [["ready"], {"ready": True}, True, 23, 1.5, None])
def test_text_fields_reject_non_strings(visual, path, value):
    target = visual
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="must be a string"):
        parse_workflow(yaml.safe_dump(visual))


def test_actual_string_regex_and_whitespace_preserve_existing_meaning(visual):
    visual["steps"][0]["evidence"]["pattern"] = "  [ready]  "
    visual["steps"][0]["agent"]["prompt"] = "  Observe carefully.  "
    workflow = parse_workflow(yaml.safe_dump(visual))
    step = workflow.steps[workflow.start_step]
    assert step.agent.prompt == "Observe carefully."
    assert step.evidence.pattern == "[ready]"
    assert re.fullmatch(step.evidence.pattern, "a")
    assert not re.fullmatch(step.evidence.pattern, "ready")
    visual["steps"][0]["evidence"]["pattern"] = " \n "
    with pytest.raises(ValueError, match="must not be empty"):
        parse_workflow(yaml.safe_dump(visual))


@pytest.mark.parametrize("edit,key", [
    (lambda source: "steps: []\n" + source, "steps"),
    (lambda source: source.replace("    initial: false", "    initial: false\n    initial: true"), "initial"),
    (lambda source: source.replace(
        "    evidence:\n", "    evidence: {pattern: wrong, consecutive: 1}\n    evidence:\n"),
     "evidence"),
    (lambda source: source.replace("      pattern: '^ready$'", "      pattern: '^ready$'\n      pattern: '^ready$'"),
     "pattern"),
    (lambda source: source.replace("    initial: false", "    initial: false\n    \"initial\": true"), "initial"),
])
def test_duplicate_keys_rejected_at_every_level(edit, key):
    source = (_REFERENCES / "example.guide.yaml").read_text()
    changed = edit(source)
    assert changed != source
    with pytest.raises(yaml.constructor.ConstructorError, match=f"duplicate mapping key '{key}'") as error:
        parse_workflow(changed)
    assert error.value.problem_mark.line >= 0


def test_safe_aliases_and_non_overlapping_merges_remain_supported():
    source = (_REFERENCES / "example.guide.yaml").read_text()
    source = source.replace("    type: boolean\n    description:", "    <<: {type: boolean}\n    description:")
    source = source.replace("    agent:\n", "    agent: &policy\n", 1)
    # Sharing a policy is not a duplicate key in the receiving step.
    start = source.index("    voice:\n")
    end = source.index("    evidence:\n", start)
    source = source[:start] + "    voice: *policy\n" + source[end:]
    workflow = parse_workflow(source)
    step = workflow.steps[workflow.start_step]
    assert step.voice == step.agent
    assert workflow.state_fields["workpiece_positioned"].type == "boolean"


def test_merge_cannot_override_an_existing_requirement():
    source = (_REFERENCES / "example.guide.yaml").read_text()
    source = source.replace("    initial: false", "    <<: {initial: false}\n    initial: true")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate mapping key 'initial'"):
        parse_workflow(source)


def test_repeated_merge_keys_are_rejected_before_expansion():
    source = (_REFERENCES / "example.guide.yaml").read_text()
    source = source.replace("    type: boolean", "    <<: {type: boolean}\n    <<: {initial: false}")
    source = source.replace("    initial: false\n", "")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate mapping key '<<'"):
        parse_workflow(source)


def test_guide_loader_is_local_and_still_safe():
    # Guide strictness must not change YAML behavior for unrelated sample configs.
    assert yaml.safe_load("value: 1\nvalue: 2") == {"value": 2}
    with pytest.raises(yaml.constructor.ConstructorError):
        parse_workflow("!!python/object/apply:builtins.str [ready]")


def test_duplicate_diagnostic_is_reported_by_read_only_cli(tmp_path, capsys):
    source = (_REFERENCES / "example.guide.yaml").read_text()
    source = source.replace("    initial: false", "    initial: false\n    initial: true")
    path = tmp_path / "duplicate.guide.yaml"
    path.write_text(source)
    assert main([str(path)]) == 1
    stdout, stderr = capsys.readouterr()
    assert not stdout
    assert "duplicate mapping key 'initial'" in stderr and "line" in stderr
    assert path.read_text() == source


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
    assert "draft, 2 steps" in success.stdout
    assert path.read_bytes() == before
    path.write_text("not: a guide")
    failure = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert failure.returncode == 1
    assert "invalid:" in failure.stderr
    assert "Traceback" not in failure.stderr
    assert path.read_text() == "not: a guide"
