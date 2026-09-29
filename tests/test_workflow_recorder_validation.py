# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guide request contracts and single-snapshot catalog generations."""

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/workflow-recorder"
sys.path.insert(0, str(_SAMPLE / "worker"))

from workflow_recorder_worker._workflow_spec import load_workflow  # noqa: E402
from workflow_recorder_worker.catalog import GuideCatalog  # noqa: E402


@pytest.fixture
def guide():
    return yaml.safe_load((_SAMPLE / "skills/recording-to-guide/references/example.guide.yaml").read_text())


@pytest.mark.parametrize("trigger", [
    {"function": "current_view", "arguments": {}},
    {"function": "current_view", "arguments": {"question": ""}},
    {"function": "current_view", "arguments": {"question": "Visible?", "extra": 1}},
    {"function": "current_view", "arguments": {"question": "x" * 501}},
    {"function": "current_view", "arguments": {"question": "Visible?"}, "result_field": "unknown"},
    {"function": "clock__timer", "arguments": {}},
    {"function": "clock__timer", "arguments": {"started_at_us": 1}},
    {"function": "clock__timer", "arguments": {"started_at_us": 1, "duration_s": 10}, "result_field": "unknown"},
    {"function": "clock__timer", "arguments": {"started_at_us": -1, "duration_s": 10}},
    {"function": "clock__timer", "arguments": {"started_at_us": True, "duration_s": 10}},
    {"function": "clock__timer", "arguments": {"started_at_us": 1, "duration_s": 0}},
    {"function": "clock__timer", "arguments": {"started_at_us": 1, "duration_s": 1.5}},
    {"function": "clock__timer", "arguments": {"started_at_us": "$state.workspace_clear", "duration_s": 10}},
])
def test_invalid_trigger_is_rejected_at_load(tmp_path, guide, trigger):
    guide["steps"][0]["trigger"] = {"interval_s": 1, **trigger}
    guide["task"]["status"] = "approved"
    path = tmp_path / "invalid.guide.yaml"
    path.write_text(yaml.safe_dump(guide))
    with pytest.raises(ValueError):
        load_workflow(path)
    catalog = GuideCatalog(tmp_path, tmp_path / "index.json", interval_s=1)
    entry, = catalog._read_guides()
    assert not entry.runnable
    assert entry.error


@pytest.mark.parametrize("result_field", [None, "elapsed_s", "remaining_s", "expired"])
def test_valid_timer_state_references_load(tmp_path, guide, result_field):
    guide["state"]["timer_start"] = {"type": "integer", "description": "Timer start", "initial": 0}
    step = guide["steps"][0]
    step["reads"] = ["timer_start"]
    step["trigger"] = {
        "function": "clock__timer", "interval_s": 1,
        "arguments": {"started_at_us": "$state.timer_start", "duration_s": 10},
        "result_field": result_field,
    }
    path = tmp_path / "timer.guide.yaml"
    path.write_text(yaml.safe_dump(guide))
    workflow = load_workflow(path)
    assert workflow.steps[workflow.start_step].trigger.result_field == result_field


@pytest.mark.parametrize("result_field", [None, "text"])
def test_valid_current_view_state_reference_loads(tmp_path, guide, result_field):
    guide["state"]["question"] = {"type": "string", "description": "Question", "initial": "Visible?"}
    step = guide["steps"][0]
    step["reads"] = ["question"]
    step["trigger"]["arguments"] = {"question": "$state.question"}
    step["trigger"]["result_field"] = result_field
    path = tmp_path / "view.guide.yaml"
    path.write_text(yaml.safe_dump(guide))
    assert load_workflow(path).steps[step["id"]].trigger.result_field == result_field


@pytest.mark.parametrize("interval", [float("nan"), float("inf"), 0, -1])
def test_invalid_trigger_interval_is_rejected(tmp_path, guide, interval):
    guide["steps"][0]["trigger"]["interval_s"] = interval
    path = tmp_path / "interval.guide.yaml"
    path.write_text(yaml.safe_dump(guide))
    with pytest.raises(ValueError, match="interval_s"):
        load_workflow(path)


def test_catalog_hashes_exact_bytes_it_parses(tmp_path, guide, monkeypatch):
    path = tmp_path / "edit.guide.yaml"
    draft = yaml.safe_dump(guide).encode()
    guide["task"]["status"] = "approved"
    approved = yaml.safe_dump(guide).encode()
    path.write_bytes(draft)
    read_bytes = Path.read_bytes

    def concurrent_edit(file):
        content = read_bytes(file)
        if file == path:
            file.write_bytes(approved)
        return content

    monkeypatch.setattr(Path, "read_bytes", concurrent_edit)
    catalog = GuideCatalog(tmp_path, tmp_path / "index.json", interval_s=1)
    first, = catalog._read_guides()
    assert first.sha256 == hashlib.sha256(draft).hexdigest()
    assert first.workflow.status == "draft"
    assert not first.runnable
    second, = catalog._read_guides()
    assert second.sha256 == hashlib.sha256(approved).hexdigest()
    assert second.runnable
    assert first.workflow.status == "draft"
