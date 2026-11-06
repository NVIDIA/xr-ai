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
import yaml
from PIL import Image
from xr_ai_models import ChatResponse, DecisionAnswer, DecisionQuestion, DecisionResponse
from xr_ai_tools.current_frame import ImageFrame
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool

_SAMPLE = Path(__file__).resolve().parents[1] / "agent-samples/tea-making-sample"
sys.path.insert(0, str(_SAMPLE / "worker"))

from tea_making_worker.application.application_config import load_config  # noqa: E402
from tea_making_worker.components import run_visual_verification_flow  # noqa: E402
from tea_making_worker.components.sop_run_recording import (  # noqa: E402  # noqa: E402
    _attempt,
    _event,
    _frame,
    _RecordedDecisions,
    _RecordedLanguage,
    _RecordedVision,
    _SOPCapture,
)


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
    capture = _SOPCapture(tmp_path, enabled=enabled)
    await capture.start(
        configuration_snapshots={
            "worker": config,
            "models": json.loads(config.models_config.read_text()),
            "voice_gate": yaml.safe_load(config.voice_gate_yaml.read_text()),
        },
        source_directory=_SAMPLE / "worker/tea_making_worker",
        workflow_name="tea making",
        steps=("identify", "fill_water", "heat_water", "start_steeping", "steep_timer"),
    )
    return capture, config.sop


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
    capture, config = await _capture(tmp_path)
    manifest = json.loads((capture.directory / "manifest.json").read_text())
    # Moving recording into components must still fingerprint the application
    # procedure and schemas, rather than only the recorder's own directory.
    for source_path in (
        "application/tea_sop_agent.py",
        "application/tea_data_models.py",
        "components/visual_verification_flow.py",
    ):
        source = _SAMPLE / "worker/tea_making_worker" / source_path
        assert manifest["source_sha256"][source_path] == hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot = json.loads((capture.directory / "config/worker.json").read_text())
    assert snapshot["sop"]["water"]["question"] == config.water.question
    assert not (capture.directory / "config/prompts").exists()
    data = _image()
    registry = ImageRegistry()
    image_reference = registry.put(data)
    frames = [
        ImageFrame(
            image=image_reference,
            participant_id="tester",
            track_id="camera",
            sequence=index,
            timestamp_us=index * 1000,
            width=6,
            height=4,
        )
        for index in (1, 2)
    ]
    vision = SimpleNamespace(
        ask_images=AsyncMock(
            return_value=ChatResponse(
                "A clear water surface is visible inside the pot.",
                None,
                None,
                "stop",
                {},
            )
        )
    )
    decision_service = SimpleNamespace(
        decide=AsyncMock(
            return_value=DecisionResponse(
                {"completion": DecisionAnswer("positive", 1, {"positive": 1, "negative": 0, "unknown": 0})},
                "fixture",
                {},
                {},
            )
        )
    )
    states = []

    async def acquire():
        frame = frames.pop(0)
        await _frame(frame)
        return frame

    async def receive(state):
        states.append(state)
        await _event("flow.state", state=state)

    async with capture.activate():
        async with _attempt("tester", "flow", step_id="fill_water"):
            await run_visual_verification_flow(
                frame_source=acquire,
                observer=ImageQueryTool(images=registry, vlm=_RecordedVision(vision)),
                decisions=_RecordedDecisions(decision_service),
                question=config.water.question,
                completion=config.water.completion,
                required_positives=2,
                window_size=4,
                poll_interval_seconds=0.001,
                history_limit=8,
                observation_text_limit=1000,
                timeout_seconds=1,
                on_state=receive,
            )
    entries = _records(capture)
    assert len(entries) == 1 and entries[0][0]["step_id"] == "fill_water"
    rows = entries[0][1]
    calls = [row for row in rows if row["kind"] == "vision.started"]
    assert [call["frame"]["sequence"] for call in calls] == [1, 2]
    assert (capture.directory / calls[0]["images"][0]["path"]).read_bytes() == data
    assert calls[0]["images"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert len(list((capture.directory / "images").iterdir())) == 1
    decisions = [row for row in rows if row["kind"] == "decision.started"]
    assert decisions[0]["questions"]["completion"]["criteria"] == dict(config.water.completion.criteria)
    assert [state.positives_in_window for state in states] == [1, 2]
    assert states[-1].status == "completed"
    assert [row["state"]["status"] for row in rows if row["kind"] == "flow.state"] == ["running", "completed"]
    assert json.loads((capture.directory / "manifest.json").read_text())["status"] == "completed"


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
        async with _attempt(participant, "flow", step_id=step, request_id=participant):
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

    import tea_making_worker.application.tea_application as app
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
            async with _attempt("startup-check", "flow", request_id="startup"):
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
