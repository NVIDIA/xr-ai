# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input contracts for the recorded-camera runner, without starting model servers."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "agent-samples/procedure-capture-replay/skills/evaluate-guide/scripts/replay_capture.py"
)
_SPEC = importlib.util.spec_from_file_location("recorded_replay", _SCRIPT)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


@pytest.fixture
def packet(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames/a.jpg").write_bytes(b"first")
    (tmp_path / "frames/b.jpg").write_bytes(b"second")
    path = tmp_path / "packet.json"
    path.write_text(json.dumps({"files": {"frame_index": "frames/index.jsonl"}}))
    rows = [{"frame_id": 1, "timestamp_us": 10, "path": "frames/a.jpg"},
            {"frame_id": 2, "timestamp_us": 20, "path": "frames/b.jpg"}]
    (tmp_path / "frames/index.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return path, rows


def test_recorded_frames_preserve_source_order_and_identity(packet):
    path, expected = packet
    before = runner.digest(path)
    assert runner.capture_frames(path) == expected
    assert runner.digest(path) == before


@pytest.mark.parametrize("fault", ["empty", "reversed", "missing", "outside"])
def test_recorded_frames_reject_invalid_evidence(packet, fault):
    path, rows = packet
    if fault == "empty":
        rows = []
    elif fault == "reversed":
        rows.reverse()
    elif fault == "missing":
        rows[0]["path"] = "frames/missing.jpg"
    else:
        rows[0]["path"] = "../outside.jpg"
    (path.parent / "frames/index.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        runner.capture_frames(path)
