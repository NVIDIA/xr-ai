# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOP capture across the real visual-tool, decision, and native state boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionQuestion, DecisionResponse
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker._capture import (  # noqa: E402
    _attempt,
    _RecordedDecisions,
    _RecordedLanguage,
    _RecordedVision,
    _SOPCapture,
)
from tea_making_worker.config import load_config  # noqa: E402
from tea_making_worker.spec import load_workflow  # noqa: E402
from tea_making_worker.workflow import GuidanceAgent  # noqa: E402


def _image():
    buffer = io.BytesIO()
    Image.new("RGB", (6, 4), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


def _records(capture):
    index = [json.loads(line) for line in (capture.directory / "attempts.jsonl").read_text().splitlines()]
    return [
        (entry, [json.loads(line) for line in (capture.directory / entry["path"]).read_text().splitlines()])
        for entry in index
    ]


async def _capture(tmp_path, *, enabled=True):
    config = load_config(_SAMPLE / "yaml/tea_making_worker.yaml")
    workflow = load_workflow(config.workflow_config)
    capture = _SOPCapture(tmp_path, enabled=enabled)
    await capture.start(config, workflow)
    return capture, workflow


def _decision(choice="accepted"):
    probabilities = {label: float(label == choice) for label in ("accepted", "rejected", "unknown")}
    return DecisionResponse(
        {"evidence": DecisionAnswer(choice, 1.0, probabilities)},
        "fixture-clef",
        {},
        {"model_revision": "fixture-revision"},
    )


@pytest.mark.asyncio
async def test_sop_capture_records_exact_images_decisions_and_confirmed_state(tmp_path):
    capture, workflow = await _capture(tmp_path)
    data = _image()
    registry = ImageRegistry()
    reference = registry.put(data)
    frame = ImageFrame(
        image=reference,
        participant_id="tester",
        track_id="source-camera",
        sequence=1,
        timestamp_us=100,
        width=6,
        height=4,
    )
    vision = SimpleNamespace(
        ask_images=AsyncMock(
            return_value=ChatResponse(
                '{"observable":true,"description":"A clear water surface is visible inside the pot."}',
                None, None, "stop", {"model": "fixture-omni"}
            )
        )
    )
    decisions = SimpleNamespace(decide=AsyncMock(return_value=_decision()))
    guidance = GuidanceAgent(
        workflow=workflow,
        llm=SimpleNamespace(chat=AsyncMock(side_effect=AssertionError("unexpected generation"))),
        decisions=_RecordedDecisions(decisions),
        current_frame=SimpleNamespace(execute=AsyncMock(return_value=frame)),
        image_query=ImageQueryTool(images=registry, vlm=_RecordedVision(vision)),
        rag=SimpleNamespace(),
    )
    session = guidance.store.get("tester")
    guidance.store.start(session)
    guidance.store.advance(session, skip=True)
    async with capture.activate():
        await guidance._tick(session)
        session.next_tick = 0
        guidance._current_frame.execute.return_value = frame.model_copy(update={"sequence": 2, "timestamp_us": 200})
        await guidance._tick(session)
    entries = _records(capture)
    assert len(entries) == 2
    first, second = entries[0][1], entries[1][1]
    assert entries[0][0]["step_id"] == "fill_water"
    assert "steps/fill_water" in entries[0][0]["path"]
    assert first[0]["before"]["state"]["water_filled"] is False
    assert first[-1]["after"]["evidence_hits"] == 1
    assert second[-1]["after"]["state"]["water_filled"] is True
    assert second[-1]["after"]["revision"] > first[0]["before"]["revision"]
    visual = next(row for row in first if row["kind"] == "vision.started")
    assert visual["frame"]["track_id"] == "source-camera"
    assert visual["frame"]["timestamp_us"] == 100
    image = visual["images"][0]
    assert (capture.directory / image["path"]).read_bytes() == data
    assert image["sha256"] == hashlib.sha256(data).hexdigest()
    assert len(list((capture.directory / "images").iterdir())) == 1
    vision.ask_images.assert_awaited()
    assert vision.ask_images.call_args.args[0] == [data]
    judged = next(row for row in first if row["kind"] == "decision.started")
    assert judged["state"]["observation"] == "A clear water surface is visible inside the pot."
    assert judged["questions"]["evidence"]["criteria"] == dict(workflow.step("fill_water").observation.criteria)
    decision = next(row for row in first if row["kind"] == "decision.finished")
    assert decision["response"]["answers"]["evidence"]["probabilities"]["accepted"] == 1
    assert decision["response"]["raw"]["model_revision"] == "fixture-revision"
    assert any(row["kind"] == "observation.judged" and row["evidence"] == "accepted" for row in second)
    assert json.loads((capture.directory / "manifest.json").read_text())["status"] == "completed"
    assert json.loads((capture.directory / "config/workflow.json").read_text())["steps"][1]["id"] == "fill_water"


@pytest.mark.asyncio
async def test_sop_capture_isolates_concurrent_participants_and_redacts_credentials(tmp_path):
    capture, _workflow = await _capture(tmp_path)
    all_entered = asyncio.Event()
    entered = 0
    service = _RecordedLanguage(
        SimpleNamespace(chat=AsyncMock(return_value=ChatResponse("answer", None, None, "stop", {})))
    )

    async def run(participant, step):
        nonlocal entered
        async with _attempt(participant, "foreground", step_id=step, request_id=participant):
            entered += 1
            if entered == 2:
                all_entered.set()
            await all_entered.wait()
            await service.chat([], headers={"Authorization": "must-not-be-recorded"}, max_tokens=8)

    async with capture.activate():
        await asyncio.gather(run("../person-a", "inspect"), run("person-b", "heat"))
    entries = _records(capture)
    assert {entry["participant_id"] for entry, _rows in entries} == {"../person-a", "person-b"}
    for entry, rows in entries:
        assert (capture.directory / entry["path"]).resolve().is_relative_to(capture.directory.resolve())
        assert all(row["participant_id"] == entry["participant_id"] for row in rows)
        assert all(row["request_id"] == entry["participant_id"] for row in rows)
        assert "must-not-be-recorded" not in json.dumps(rows)
        assert next(row for row in rows if row["kind"] == "language.started")["settings"]["max_tokens"] == 8


@pytest.mark.asyncio
async def test_sop_capture_stream_cancellation_retains_partial_answer_and_closes_source(tmp_path):
    capture, _workflow = await _capture(tmp_path)
    suspended = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    async def stream(*_args, **_kwargs):
        try:
            yield "Partial"
            suspended.set()
            await release.wait()
            yield " answer"
        finally:
            closed.set()

    service = _RecordedVision(SimpleNamespace(stream_images=stream))

    async def run():
        async with _attempt("person", "foreground", step_id="inspect"):
            async for _token in service.stream_images([_image()], "Inspect this."):
                pass

    async with capture.activate():
        task = asyncio.create_task(run())
        await suspended.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
    rows = _records(capture)[0][1]
    result = next(row for row in rows if row["kind"] == "vision.finished")
    assert result["content"] == "Partial" and result["complete"] is False
    assert rows[-1]["status"] == "cancelled"
    assert len(list((capture.directory / "images").iterdir())) == 1


@pytest.mark.asyncio
async def test_sop_capture_failed_inference_retains_input_and_error(tmp_path):
    capture, _workflow = await _capture(tmp_path)
    service = _RecordedVision(SimpleNamespace(ask_images=AsyncMock(side_effect=RuntimeError("model unavailable"))))
    async with capture.activate():
        with pytest.raises(RuntimeError, match="model unavailable"):
            async with _attempt("person", "observer", step_id="inspect"):
                await service.ask_images([_image()], "Inspect this.")
    rows = _records(capture)[0][1]
    assert any(row["kind"] == "vision.failed" and row["error_type"] == "RuntimeError" for row in rows)
    assert rows[-1]["status"] == "failed"
    assert len(list((capture.directory / "images").iterdir())) == 1


@pytest.mark.asyncio
async def test_sop_capture_drains_file_write_before_propagating_cancellation(tmp_path):
    capture, _workflow = await _capture(tmp_path)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    path = tmp_path / "write-finished"

    def write():
        loop.call_soon_threadsafe(started.set)
        release.wait()
        path.write_text("finished")

    task = asyncio.create_task(capture.io(write))
    await started.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert path.read_text() == "finished"


@pytest.mark.asyncio
async def test_disabled_sop_capture_passes_through_without_attempt_or_image_files(tmp_path):
    capture, _workflow = await _capture(tmp_path, enabled=False)
    underlying = SimpleNamespace(ask_images=AsyncMock(return_value=ChatResponse("answer", None, None, "stop", {})))
    async with capture.activate():
        async with _attempt("person", "observer", step_id="inspect"):
            await _RecordedVision(underlying).ask_images([_image()], "Inspect.")
    underlying.ask_images.assert_awaited_once()
    assert not (capture.directory / "images").exists()
    assert not (capture.directory / "attempts.jsonl").exists()
    assert json.loads((capture.directory / "manifest.json").read_text())["capture_enabled"] is False


@pytest.mark.asyncio
async def test_sop_capture_serialization_keeps_authorization_criteria_and_redacts_secrets(tmp_path):
    capture, _workflow = await _capture(tmp_path)
    question = DecisionQuestion("Check the requested action.", {"command": "Execute", "not_command": "Discussion"})
    service = _RecordedDecisions(
        SimpleNamespace(
            decide=AsyncMock(
                return_value=DecisionResponse(
                    {"authorization": DecisionAnswer("not_command", 1.0, {"command": 0.0, "not_command": 1.0})},
                    "fixture",
                    {},
                    {},
                )
            )
        )
    )
    async with capture.activate():
        async with _attempt("person", "foreground"):
            await service.decide({"request": "Explain", "api_key": "secret-value"}, {"authorization": question})
    rows = _records(capture)[0][1]
    request = next(row for row in rows if row["kind"] == "decision.started")
    assert request["questions"]["authorization"]["criteria"]["not_command"] == "Discussion"
    assert request["state"]["api_key"] == "<redacted>"
    assert "secret-value" not in json.dumps(rows)


@pytest.mark.parametrize("capture", ["invalid", {"enabled": "true"}, {"unknown": True}])
def test_sop_capture_settings_require_boolean(tmp_path, capture):
    path = tmp_path / "worker.yaml"
    path.write_text(json.dumps({"capture": capture}))
    with pytest.raises(ValueError, match="enabled boolean"):
        load_config(path)


@pytest.mark.asyncio
async def test_sop_capture_new_launch_keeps_previous_run_and_rejects_late_attempt(tmp_path):
    first, _workflow = await _capture(tmp_path)
    release = asyncio.Event()

    async def late():
        await release.wait()
        async with _attempt("late", "observer", step_id="inspect"):
            pass

    async with first.activate():
        task = asyncio.create_task(late())
    release.set()
    with pytest.raises(RuntimeError, match="closed"):
        await task
    original = (first.directory / "manifest.json").read_bytes()
    second, _workflow = await _capture(tmp_path)
    async with second.activate():
        pass
    assert first.directory != second.directory
    assert (first.directory / "manifest.json").read_bytes() == original
    assert not (first.directory / "attempts.jsonl").exists()


@pytest.mark.asyncio
async def test_sop_capture_is_active_in_real_worker_composition(tmp_path, monkeypatch):
    from dataclasses import replace

    import tea_making_worker.app as app
    from xr_ai_runtime import Agent

    config = replace(load_config(_SAMPLE / "yaml/tea_making_worker.yaml"), artifacts_dir=tmp_path, web_events_port=0)
    clients = SimpleNamespace(
        chat=AsyncMock(return_value=ChatResponse("answer", None, None, "stop", {})),
        health=AsyncMock(return_value=True),
        close=AsyncMock(),
    )
    for factory in ("make_llm", "make_decision", "make_vlm", "make_tts"):
        monkeypatch.setattr(app, factory, lambda *_args: clients)
    monkeypatch.setattr(app, "RAGTools", lambda *_args: SimpleNamespace(health=AsyncMock(), close=AsyncMock()))
    monkeypatch.setattr(app, "HubVoiceTransport", lambda: SimpleNamespace(endpoint=Mock()))
    monkeypatch.setattr(app, "setup_logging", lambda *_args: None)

    class Voice(Agent):
        async def run(self, runtime):
            async with _attempt("startup-check", "foreground", request_id="startup"):
                await _RecordedLanguage(clients).chat([])

    monkeypatch.setattr(app, "VoiceAgent", SimpleNamespace(from_models=lambda **_kwargs: Voice()))
    await app.run_app(config)
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    manifest = json.loads((runs[0] / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["workflow"]["steps"][0] == "identify"
    assert (runs[0] / "relay-events.jsonl").exists()
    index = [json.loads(line) for line in (runs[0] / "attempts.jsonl").read_text().splitlines()]
    assert index[0]["participant_id"] == "startup-check"
    records = (runs[0] / index[0]["path"]).read_text()
    assert "language.finished" in records


@pytest.mark.asyncio
async def test_sop_capture_failed_image_write_never_dispatches_or_leaves_valid_partial_asset(tmp_path, monkeypatch):
    capture, _workflow = await _capture(tmp_path)
    write_bytes = Path.write_bytes
    data = _image()

    def fail_after_partial_write(path, payload):
        write_bytes(path, payload[:10])
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", fail_after_partial_write)
    underlying = SimpleNamespace(ask_images=AsyncMock(return_value=ChatResponse("answer", None, None, "stop", {})))
    async with capture.activate():
        with pytest.raises(OSError, match="disk full"):
            async with _attempt("person", "observer", step_id="inspect"):
                await _RecordedVision(underlying).ask_images([data], "Inspect.")
        underlying.ask_images.assert_not_awaited()
        assert not list((capture.directory / "images").glob("*.jpg"))
        monkeypatch.setattr(Path, "write_bytes", write_bytes)
        saved = await capture.image(data)
        assert (capture.directory / saved["path"]).read_bytes() == data
    assert _records(capture)[0][1][-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_sop_capture_path_is_frozen_to_the_saved_bytes_before_dispatch(tmp_path):
    capture, _workflow = await _capture(tmp_path)
    original = _image()
    source = tmp_path / "source.jpg"
    source.write_bytes(original)

    async def model(images, _question, **_kwargs):
        source.write_bytes(b"changed after recording")
        assert images == [original]
        return ChatResponse("answer", None, None, "stop", {})

    async with capture.activate():
        async with _attempt("person", "observer", step_id="inspect"):
            await _RecordedVision(SimpleNamespace(ask_images=model)).ask_images([source], "Inspect.")
    rows = _records(capture)[0][1]
    image = next(row for row in rows if row["kind"] == "vision.started")["images"][0]
    assert (capture.directory / image["path"]).read_bytes() == original


@pytest.mark.asyncio
async def test_context_compaction_records_a_separate_attempt_after_the_scheduling_turn(tmp_path):
    from tea_making_worker._conversation_context import _CompactionLanguage, _context_owner
    from xr_ai_conversation import ContextManager

    capture, _ = await _capture(tmp_path)
    started = asyncio.Event()

    async def summarize(*_args, **_kwargs):
        started.set()
        return ChatResponse("Earlier discussion about preparing a drink.", None, None, "stop", {})

    service = SimpleNamespace(chat=AsyncMock(side_effect=summarize))
    context = ContextManager(
        _CompactionLanguage(_RecordedLanguage(service)), dialogue_budget_bytes=512,
        compaction_input_budget_bytes=1024,
    )
    async with capture.activate():
        async with _attempt("tester", "foreground"):
            with _context_owner("tester"):
                for index in range(8):
                    context.record_exchange(
                        "tester", "Explain the procedure. " * 10, "Use the current instructions. " * 10,
                        timestamp_us=index, exchange_id=index,
                    )
        await asyncio.wait_for(started.wait(), 2)
        await context.close()
    records = _records(capture)
    foreground = [events for entry, events in records if entry["channel"] == "foreground"]
    compactions = [events for entry, events in records if entry["channel"] == "context"]
    assert foreground and compactions
    assert not any(event["kind"] == "language.started" for event in foreground[0])
    assert all(events[-1]["kind"] == "attempt.finished" for events in compactions)
    assert any(event["kind"] == "language.started" for events in compactions for event in events)
