# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise evidence ownership, native measurement checks, and drained run traces."""

# Pytest discovers the shared fixtures imported below.
# ruff: noqa: F811, E402
import asyncio
import hashlib
import io
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from _procedure_guide_helpers import clef_response
from PIL import Image
from xr_ai_models import DecisionAnswer, DecisionResponse, load_models_config
from xr_ai_tools.vision import ImageQueryResult

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide"
sys.path.insert(0, str(_SAMPLE / "worker"))

from procedure_capture_guide_worker.guide.controls import classify_control
from procedure_capture_guide_worker.guide.measurements import (
    MeasurementPolicy,
    compare_measurements,
    make_measurement_vlm,
)
from procedure_capture_guide_worker.guide.run_trace import RunTrace
from procedure_capture_guide_worker.guide.verification import verdict
from test_sop_guide import connected_engine, control, document, engine, observation, selected  # noqa: F401


def policy(**kwargs):
    return MeasurementPolicy.model_validate({
        "ranges": [{"unit": "C", "minimum": 100, "maximum": 100},
                   {"unit": "F", "minimum": 212, "maximum": 212}], **kwargs,
    })


def reading(value, unit="C", role="unlabeled"):
    return {"value": value, "unit": unit, "role": role}


@pytest.mark.parametrize("readings,expected", [
    ([reading(100)], "accepted"), ([reading(212, "F")], "accepted"),
    ([reading(100, role="target")], "accepted"),
    ([reading(94, role="current"), reading(100, role="target")], "rejected"),
    ([reading(100, role="current"), reading(94, role="target")], "accepted"),
    ([reading(100, role="current"), reading(94, role="current")], "rejected"),
    ([reading(100, "")], "rejected"), ([reading(100, "F")], "rejected"),
    ([], "unknown"),
])
def test_measurement_rules_preserve_units_roles_and_unknown(readings, expected):
    assert compare_measurements(json.dumps({"readings": readings}), policy()).verdict == expected


@pytest.mark.parametrize("value", [True, "100", None, float("nan"), float("inf")])
def test_invalid_measurement_is_not_coerced(value):
    with pytest.raises(ValueError):
        compare_measurements(json.dumps({"readings": [reading(value)]}), policy())


@pytest.mark.parametrize("ranges", [
    [{"unit": "mm", "minimum": 21, "maximum": 20}],
    [{"unit": "mm", "minimum": 0, "maximum": float("inf")}],
    [{"unit": "mm", "minimum": False, "maximum": 20}],
    [{"unit": " mm", "minimum": 0, "maximum": 20}],
    [{"unit": "mm", "minimum": 0, "maximum": 20}] * 2,
    [],
])
def test_invalid_measurement_policy_rejected(ranges):
    with pytest.raises(ValueError):
        policy(ranges=ranges)


def test_measurement_schema_is_per_client_and_not_prompt_only(monkeypatch):
    from procedure_capture_guide_worker.guide import measurements
    sample = Path(__file__).resolve().parents[1] / "agent-samples/procedure-capture-guide"
    config = load_models_config(sample / "yaml/models.json")
    original = dict(config.vlm("vlm").adapter.default_extras)
    factory = Mock()
    monkeypatch.setattr(measurements, "make_vlm", factory)
    make_measurement_vlm(config)
    derived = factory.call_args.args[0]
    assert derived is not config
    assert config.vlm("vlm").adapter.default_extras == original
    output = derived.vlm("vlm").adapter.default_extras["response_format"]
    assert output["type"] == "json_schema" and output["json_schema"]["strict"]
    assert output["json_schema"]["schema"]["additionalProperties"] is False


def test_affirmative_plurality_cannot_verify():
    answer = DecisionAnswer("accepted", .4, {"accepted": .4, "rejected": .3, "unknown": .3})
    assert verdict(answer) == "unknown"
    assert verdict(clef_response("yes").answers["condition_met"]) == "accepted"


async def test_unknown_breaks_streak_and_context_expires_at_transition(engine):
    backend = engine._backend
    session = backend._sessions["alice"]
    session.evidence_hits = 1
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    backend._trigger = AsyncMock(return_value=replace(observation("unclear", 1), value=image.getvalue()))
    backend._image_query.execute.side_effect = None
    backend._image_query.execute.return_value = ImageQueryResult(text="The attachment is obscured.", available=True)
    backend._decisions.decide.side_effect = None
    backend._decisions.decide.return_value = clef_response("unknown", .95)
    await backend._tick(session)
    assert session.evidence_hits == 0 and not session.state["base"]
    guidance = await backend.guidance("alice")
    assert guidance.context["last_verification"]["verdict"] == "unknown"
    assert guidance.context["last_verification"]["age_s"] >= 0
    guidance.context["last_verification"]["verdict"] = "accepted"
    assert session.last_verification["verdict"] == "unknown"
    await control(backend, "advance", skip=True)
    assert (await backend.guidance("alice")).context["last_verification"] is None


@pytest.mark.parametrize("authorization", [
    DecisionAnswer("not_command", .9, {"command": .05, "not_command": .9, "unknown": .05}),
    DecisionAnswer("unknown", .9, {"command": .05, "not_command": .05, "unknown": .9}),
    DecisionAnswer("command", .4, {"command": .4, "not_command": .3, "unknown": .3}),
])
async def test_control_requires_independent_affirmative_majority(authorization):
    authorization = replace(authorization, probabilities={
        **authorization.probabilities, "reported_instruction": 0.0, "explanation": 0.0,
    })
    decisions = Mock(decide=AsyncMock(side_effect=[
        DecisionResponse(answers={"action": DecisionAnswer("advance", .9, {"advance": .9})},
                         model="test", usage={}, raw={}),
        DecisionResponse(answers={"authorization": authorization}, model="test", usage={}, raw={}),
    ]))
    binding, record = await classify_control(decisions, "request", {"active": True}, timeout_s=1)
    assert binding is None and record["authorization"]["choice"] == authorization.choice
    assert decisions.decide.await_count == 2


async def test_trace_writer_drains_after_close_cancellation(tmp_path, document):
    guide = selected(document)
    guide = replace(guide, content=b"pinned source")
    trace = RunTrace(tmp_path, guide)
    entered, release = threading.Event(), threading.Event()
    save = trace._save
    def blocked(*args):
        entered.set()
        release.wait(5)
        save(*args)
    trace._save = blocked
    image = b"original selected frame"
    trace.emit("run", "run_started", state={"done": False})
    trace.emit("run", "observation_started", image=image, frame={"sequence": 1})
    trace.emit("run", "run_ended", reason="disconnect")
    assert await asyncio.to_thread(entered.wait, 2)
    closing = asyncio.create_task(trace.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events] == ["run_started", "observation_started", "run_ended"]
    assert (tmp_path / "run" / events[1]["image"]["path"]).read_bytes() == image
    assert events[1]["image"]["sha256"] == hashlib.sha256(image).hexdigest()
    assert json.loads((tmp_path / "run/packet.json").read_text())["status"] == "ended"


async def test_trace_separates_runs_and_keeps_pinned_source(tmp_path, document):
    import yaml
    from procedure_capture_guide_worker.guide.guides import load_guide
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(document))
    guide = load_guide(path)
    trace = RunTrace(tmp_path / "runs", guide)
    path.write_text("source edited during run")
    for run in ("first", "second"):
        trace.emit(run, "run_started", state={})
        trace.emit(run, "run_ended", reason="completed")
    await trace.close()
    for run in ("first", "second"):
        assert (tmp_path / "runs" / run / "guide.yaml").read_bytes() == guide.content


async def test_trace_follows_stop_to_original_run(engine, tmp_path, document):
    import yaml
    backend = engine._backend
    source = yaml.safe_dump(document).encode()
    guide = replace(selected(document), content=source)
    backend._trace = RunTrace(tmp_path, guide)
    await engine._route("start guide", "alice")
    session = backend._sessions["alice"]
    run_id = session.run_id
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    backend._trigger = AsyncMock(return_value=replace(observation("YES", 1), value=image.getvalue()))
    backend._image_query.execute.side_effect = None
    backend._image_query.execute.return_value = ImageQueryResult(text="YES", available=True)
    await backend._tick(session)
    await engine._route("stop guide", "alice")
    assert not session.active and session.run_id is None
    await backend._trace.close()
    directory = tmp_path / run_id
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    observation_event = next(event for event in events if event["kind"] == "observation_started")
    assert (directory / observation_event["image"]["path"]).read_bytes() == image.getvalue()
    assert any(event["kind"] == "observation_finished" and event["verdict"] == "accepted" for event in events)
    assert any(event["kind"] == "user_control_result" and event["tool"] == "workflow__reset" for event in events)
    assert any(event["kind"] == "control_result" and event["action"] == "reset" for event in events)
    assert json.loads((directory / "packet.json").read_text())["status"] == "ended"
    assert (directory / "guide.yaml").read_bytes() == source


async def test_trace_failure_after_logical_end_marks_packet_incomplete(tmp_path, document, monkeypatch):
    trace = RunTrace(tmp_path, selected(document))
    save = trace._save
    def fail(run_id, event, image):
        if event["kind"] == "observation_finished":
            raise OSError("disk full")
        save(run_id, event, image)
    monkeypatch.setattr(trace, "_save", fail)
    trace.emit("run", "run_started")
    trace.emit("run", "run_ended", reason="reset")
    trace.emit("run", "observation_finished")
    await trace.close()
    assert json.loads((tmp_path / "run/packet.json").read_text())["status"] == "incomplete"


async def test_unavailable_camera_replaces_previous_context(engine):
    backend = engine._backend
    session = backend._sessions["alice"]
    session.last_verification = {"verdict": "accepted", "observed_monotonic": 0}
    backend._trigger = AsyncMock(side_effect=LookupError("frame expired"))
    with pytest.raises(LookupError):
        await backend._tick(session)
    context = (await backend.guidance("alice")).context["last_verification"]
    assert context["verdict"] is None and context["error"]["type"] == "LookupError"


async def test_informational_classification_keeps_evidence_answering(engine):
    from xr_ai_models import ChatResponse
    session = engine._backend._sessions["alice"]
    session.last_verification = {
        "verdict": "rejected", "caption": {"text": "The display reads 94 C."},
        "observed_monotonic": 0,
    }
    before = dict(session.state)
    engine._decisions.decide.side_effect = None
    engine._decisions.decide.return_value = DecisionResponse(
        answers={"action": DecisionAnswer("status", 1.0, {"status": 1.0})},
        model="test", usage={}, raw={},
    )
    async def answer(messages, *, tools, **kwargs):
        context = json.loads(messages[1].content)
        assert context["last_verification"]["caption"]["text"] == "The display reads 94 C."
        assert {tool.name for tool in tools} == {"current_view", "workflow__status"}
        return ChatResponse("The last check read 94 C, below the required temperature.", None, None, "stop", {})
    engine._llm.chat.side_effect = answer
    assert "94 C" in await engine._route("Why has the check not passed?", "alice")
    assert session.state == before
    assert engine._decisions.decide.await_count == 1
