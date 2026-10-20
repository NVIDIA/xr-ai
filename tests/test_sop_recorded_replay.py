# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise recorded packet evaluation through the production guide runtime.

Only model responses are deterministic doubles. Camera selection, evidence
deduplication, conversation tools, timers and delivery run their normal code.
These regressions test the harness contract, not local-model perception.
"""

import importlib.util
import io
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from PIL import Image

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide"


@pytest.fixture
def packets(monkeypatch):
    monkeypatch.syspath_prepend(str(_SAMPLE / "worker"))
    monkeypatch.syspath_prepend(str(_SAMPLE / "eval"))
    spec = importlib.util.spec_from_file_location("recorded_camera_eval", _SAMPLE / "eval/_recorded_camera.py")
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


@pytest.mark.parametrize("color,count,automatic,expected", [
    ("blue", 16, True, True),
    ("blue", 16, False, True),
    ("red", 16, True, False),
    ("blue", 1, True, False),
])
def test_packet_cli_uses_runtime_controls_and_unique_source_evidence(
    packets, tmp_path, monkeypatch, color, count, automatic, expected,
):
    import xr_ai_models
    from xr_ai_models import ChatResponse, ToolCall

    packet = make_packet(tmp_path, count, color)
    guide = yaml.safe_load((_SAMPLE / "eval/fixtures/indicator.guide.yaml").read_text())
    guide["task"]["status"] = "draft"
    guide["steps"][1]["trigger"]["arguments"]["duration_s"] = 1
    guide["steps"][1]["finish_on_complete"] = automatic
    path = tmp_path / "draft.guide.yaml"
    path.write_text(yaml.safe_dump(guide))
    original = path.read_bytes()
    original_packet = packet.read_bytes()

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
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["guide.py", "--packet", str(packet), "--guide", str(path),
                                    "--output", str(output), "--tail-s", "2"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(str(_SAMPLE / "eval/guide.py"), run_name="__main__")
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
    assert (output / "runtime.json").is_file()
    assert (output / "relay.jsonl").is_file()
    llm.close.assert_awaited_once()
    vlm.close.assert_awaited_once()


def test_generation_cases_have_separate_inputs_and_review_criteria():
    cases = yaml.safe_load((_SAMPLE / "eval/guide_generation_cases.yaml").read_text())
    assert len(cases) == 7
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert set(case) == {"id", "input", "checks"}
        assert set(case["input"]) == {"packet", "captions", "transcripts", "media_access"}
        assert case["input"]["packet"]["status"] == "complete"
        assert case["input"]["captions"] and case["input"]["transcripts"]
        assert case["checks"] and all(isinstance(check, str) and check for check in case["checks"])
        assert "checks" not in case["input"]
