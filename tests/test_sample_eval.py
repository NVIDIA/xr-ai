# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixture integrity, timing attribution and native IPC lifecycle for speech replays."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _sample_eval._fixtures import _audio_path, _decode_audio, _load_audio, _prepare
from _sample_eval._report import _turns, _write_report
from _sample_eval._scenario import _SAMPLES, _builtin, _discover, _load_scenario, _Scenario
from xr_ai_interaction import UserQuery


def _wav(samples: int = 320, rate: int = 16000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(np.arange(samples, dtype="<i2").tobytes())
    return out.getvalue()


def _short_scenario() -> _Scenario:
    return _Scenario("conversation", 0.08, [{"at_s": 0, "say": "hello"}])


def _pin(directory: Path, scenario: _Scenario) -> None:
    directory.mkdir()
    files = {}
    for text in scenario.texts:
        raw = _wav()
        _audio_path(directory, text).write_bytes(raw)
        files[text] = hashlib.sha256(raw).hexdigest()
    (directory / "audio.json").write_text(json.dumps({"version": 1, "files": files}))


@pytest.mark.parametrize("name", _SAMPLES)
def test_discovered_replays_have_ordered_inputs_and_stable_identity(name):
    scenario = _builtin(name)
    assert scenario.duration_s > 0 and Path(scenario.source).is_file()
    assert scenario.texts
    times = [event["at_s"] for event in scenario.events]
    assert times == sorted(times) and all(0 <= at < scenario.duration_s for at in times)
    assert scenario.sha256 == _builtin(name).sha256


@pytest.mark.parametrize("folder", ["eval", "evaluation"])
def test_replay_discovery_adds_sample_owned_files_without_registry_edits(tmp_path, folder):
    import yaml

    path = tmp_path / "agent-samples/new-sample" / folder / "speech/new-replay.yaml"
    path.parent.mkdir(parents=True)
    config = path.parents[2] / "yaml/worker.yaml"
    config.parent.mkdir()
    config.write_text("{}")
    data = {
        "duration_s": 10,
        "worker": {"package": "new_worker", "config": "../../yaml/worker.yaml"},
        "events": [{"at_s": 0, "say": "hello"}],
    }
    path.write_text(yaml.safe_dump(data))
    discovered = _discover(tmp_path)
    assert list(discovered) == ["new-replay"]
    scenario = _load_scenario(discovered["new-replay"])
    assert scenario.sample == "new-replay" and scenario.worker == data["worker"]
    assert scenario.source == str(path.resolve()) and scenario.texts == ["hello"]
    duplicate = path.parents[3] / "another-sample/eval/speech/new-replay.yaml"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(path.read_text())
    with pytest.raises(ValueError, match="duplicate speech replay new-replay"):
        _discover(tmp_path)


@pytest.mark.parametrize(
    "invalid", [
        "unknown", "time", "unordered", "zero_duration", "bool_duration", "worker", "missing_package",
        "missing_config", "speech", "handler", "launch", "launch_no_mode", "empty_launch", "guide_path", "image",
    ]
)
def test_invalid_sample_owned_replay_is_rejected_before_model_calls(tmp_path, invalid):
    import yaml

    path = tmp_path / "replay.yaml"
    config = tmp_path / "worker.yaml"
    config.write_text("{}")
    data = {
        "duration_s": 10,
        "worker": {"package": "worker", "config": "worker.yaml"},
        "events": [{"at_s": 0, "say": "hello"}],
    }
    if invalid == "unknown":
        data["warmup_s"] = 5
    elif invalid == "time":
        data["events"][0]["at_s"] = float("nan")
    elif invalid == "unordered":
        data["events"] = [{"at_s": 2, "say": "later"}, {"at_s": 1, "say": "earlier"}]
    elif invalid in {"zero_duration", "bool_duration"}:
        data["duration_s"] = 0 if invalid == "zero_duration" else True
    elif invalid == "worker":
        data["worker"]["typo"] = True
    elif invalid == "missing_package":
        del data["worker"]["package"]
    elif invalid == "missing_config":
        data["worker"]["config"] = "absent.yaml"
    elif invalid == "speech":
        data["events"][0]["say"] = ""
    elif invalid == "launch":
        data["worker"]["launch"] = {"typo": "guide"}
    elif invalid in {"launch_no_mode", "empty_launch"}:
        data["worker"]["launch"] = {"guide_path": "worker.yaml"} if invalid == "launch_no_mode" else {}
    elif invalid == "guide_path":
        data["worker"]["launch"] = {"mode": "guide", "guide_path": "missing.yaml"}
    elif invalid == "image":
        data["events"][0]["scene"] = {"image": 123}
    else:
        data["events"][0]["handler"] = "false"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="replay.yaml"):
        _load_scenario(path)


@pytest.mark.asyncio
async def test_prepare_reuses_audio_and_run_rejects_changed_or_overlapping_input(tmp_path, monkeypatch):
    import xr_ai_models

    scenario = _short_scenario()
    calls = []

    class TTS:
        async def synthesize(self, text, **_kwargs):
            calls.append(text)
            return _wav()

        async def close(self):
            pass

    monkeypatch.setattr(xr_ai_models, "load_models_config", lambda _path: None)
    monkeypatch.setattr(xr_ai_models, "make_tts", lambda *_args: TTS())
    cache = tmp_path / "audio"
    await _prepare(scenario, cache, tmp_path / "models.json")
    await _prepare(scenario, cache, tmp_path / "models.json")
    assert calls == ["hello"]
    audio, hashes = _load_audio(scenario, cache)
    assert len(audio["hello"]) == 320 and hashes["hello"]
    _audio_path(cache, "hello").write_bytes(_wav(16000))
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_audio(scenario, cache)
    manifest = {"version": 1, "files": {"hello": hashlib.sha256(_wav(16000)).hexdigest()}}
    (cache / "audio.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlaps"):
        _load_audio(scenario, cache)
    assert (cache / ".gitignore").read_text() == "*\n"


def test_wav_resampling_preserves_duration_and_rejects_empty_audio():
    assert len(_decode_audio(_wav(2400, 24000))) == 1600
    with pytest.raises(ValueError, match="contain audio"):
        _decode_audio(_wav(0))


@pytest.mark.parametrize("name", _SAMPLES)
def test_real_worker_config_accepts_model_override(tmp_path, name):
    from _sample_eval._replay import _worker

    models = tmp_path / "models.json"
    voice = tmp_path / "voice.yaml"
    from dataclasses import replace

    scenario = replace(_builtin(name), sample="another-replay-name")
    app, cfg = _worker(scenario, tmp_path, models, voice, None)
    assert callable(app.run_app)
    assert cfg.models_config == models and cfg.voice_gate_yaml == voice
    if hasattr(cfg, "artifacts_dir"):
        assert cfg.artifacts_dir == tmp_path / "artifacts"
    if hasattr(cfg, "web_events_port"):
        assert cfg.web_events_port == 0
    if hasattr(cfg, "text_memory_dir"):
        assert cfg.text_memory_dir == str(tmp_path / "text-memory")
    if scenario.worker.get("launch", {}).get("mode") == "capture":
        assert cfg.capture.artifacts_dir == tmp_path / "captures"
    elif scenario.worker.get("launch", {}).get("mode") == "guide":
        assert cfg.guide.guide_path.name == "guide.yaml"


def test_reply_completion_uses_request_ids_across_new_input_and_ignores_background():
    rows = []

    def row(at, event, **values):
        rows.append({"event": event, "elapsed": at, **values})

    def reply(at, request, response, final=False):
        row(
            at,
            "runtime",
            topic="voice.output",
            source="agent",
            message={
                "request_id": request,
                "response_id": response,
                "final": final,
                "text": "answer",
            },
        )

    row(0, "speech_input", text="first", audio_start_us=0, audio_end_us=100000)
    row(0.1, "speech_input_end")
    row(1.25, "request", request_id="a", message={"text": "FIRST!", "timestamp_us": 50000})
    row(1, "speech_input", text="second", audio_start_us=1000000, audio_end_us=1100000)
    row(1.1, "speech_input_end")
    row(1.26, "request", request_id="b", message={"text": "second", "timestamp_us": 1050000})
    row(1.2, "audio_out", rms=0.1)
    reply(1.3, "a", "a-main", True)
    reply(1.4, None, None, True)
    reply(1.6, "b", "b-main")
    reply(1.8, "b", "b-main", True)
    turns = _turns(rows, 2)
    assert turns[0]["completed_reply_text_s"] == pytest.approx(1.2)
    assert turns[1]["completed_reply_text_s"] == pytest.approx(0.7)
    assert turns[1]["first_observed_audio_s"] == pytest.approx(0.1)
    assert turns[0]["query_delivery_s"] == pytest.approx(1.15)
    assert turns[1]["query_delivery_s"] == pytest.approx(0.16)
    row(
        1.9,
        "runtime",
        topic="voice.output",
        source="another-agent",
        message={"request_id": "a", "response_id": "a-main", "final": False, "text": "pending"},
    )
    assert _turns(rows, 2)[0]["completed_reply_text_s"] is None


def test_incomplete_replay_reports_failure_and_only_writes_json(tmp_path):
    _report_fixture(tmp_path, _short_scenario(), [{"event": "run_finished", "elapsed": None, "status": "completed"}])
    report = _write_report(tmp_path)
    assert report["execution_status"] == "failed"
    assert report["completed_reply_text_s"]["median"] == "unmeasured"
    assert report["speaker_drops"] == "not applicable"
    assert not (tmp_path / "review.json").exists()
    assert not (tmp_path / "report.csv").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _SAMPLES)
async def test_native_ipc_replay_cleans_up_worker_and_restores_private_hooks(tmp_path, monkeypatch, name):
    import xr_ai_voice._transport as transport
    from _sample_eval import _replay as replay
    from _sample_eval._recording import _Recording
    from loguru import logger
    from xr_ai_hub import ProcessorEndpoint
    from xr_ai_runtime import AgentRuntime

    audio, video, closed = [], [], []
    sinks = []
    observe_errors = _Recording.observe_errors

    def remember_sink(recording):
        observe_errors(recording)
        sinks.append(recording._error_sink)

    monkeypatch.setattr(_Recording, "observe_errors", remember_sink)
    original_publish, original_address, original_pub = AgentRuntime._publish, transport._HUB_PUSH, transport._HUB_PUB

    async def app(_cfg, *, ready_file):
        if hasattr(app_module, "setup_logging"):
            app_module.setup_logging()
        if name in ("capture", "guide"):
            push, pub = app_module._HUB_PUSH, app_module._HUB_PUB
        else:
            push, pub = transport._HUB_PUSH, transport._HUB_PUB
        endpoint = ProcessorEndpoint(sub_addr=pub, push_addr=push)

        async def on_audio(chunk):
            audio.append(chunk)
            if name == "lab-instrument-monitoring" and len(audio) == 1:
                logger.error("injected worker failure after native logging setup")

        async def on_frame(frame):
            video.append(frame)

        endpoint.on_audio(on_audio)
        endpoint.on_frame(on_frame)
        task = asyncio.create_task(endpoint.run())
        try:
            await endpoint.wait_until_running()
            assert await endpoint.wait_for_subscriptions(timeout=2)
            ready_file.touch()
            await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            endpoint.stop()
            endpoint.close()
            logger.error("teardown-only error")
            closed.append(True)

    app_module = SimpleNamespace(run_app=app, _HUB_PUSH="original-push", _HUB_PUB="original-pub")
    if name == "lab-instrument-monitoring":
        app_module.setup_logging = lambda: logger.remove(sinks[-1])
    snapshots = []

    def worker(_scenario, _output, snapshot, gate_snapshot, _rag, _settings):
        assert snapshot.read_bytes() == b'{"models": {}}'
        assert gate_snapshot.read_bytes() == b"{}\n"
        assert gate_snapshot.stat().st_mode & 0o777 == 0o600
        snapshots.append(gate_snapshot)
        assert snapshot.stat().st_mode & 0o777 == 0o600
        snapshots.append(snapshot)
        return app_module, None

    monkeypatch.setattr(replay, "_worker", worker)
    monkeypatch.setattr(replay, "_RESPONSE_DRAIN_S", 0.1)
    from dataclasses import replace

    scenario = replace(_short_scenario(), sample=name)
    cache = tmp_path / "audio"
    _pin(cache, scenario)
    models = tmp_path / "models.json"
    models.write_text('{"models": {}}')
    output = tmp_path / "run"
    output.mkdir()
    voice = tmp_path / "voice.yaml"
    voice.write_bytes(b"{}\n")
    inputs = replay._preflight(scenario, models, cache, {}, voice)
    models.write_text("source changed after validation")
    voice.write_text("voice source changed after validation")
    await replay._run(scenario, inputs, output, "baseline", {}, ready_timeout=5)
    assert snapshots and all(not path.exists() for path in snapshots)
    assert len(audio) == 4 and len(video) == 1
    assert sum(chunk.samples for chunk in audio) == 1280
    np.testing.assert_array_equal(
        np.frombuffer(b"".join(chunk.data for chunk in audio), dtype="<f4"),
        np.concatenate([_decode_audio(_wav()), np.zeros(960, dtype="<f4")]),
    )
    assert closed == [True]
    assert transport._HUB_PUB == original_pub
    assert (app_module._HUB_PUSH, app_module._HUB_PUB) == ("original-push", "original-pub")
    rows = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "speech_input_end" for row in rows)
    source = next(row for row in rows if row["event"] == "speech_input")
    assert source["audio_start_us"] == audio[0].pts_us
    assert source["audio_end_us"] - source["audio_start_us"] == 20000
    assert [chunk.pts_us - audio[0].pts_us for chunk in audio] == [0, 20000, 40000, 60000]
    assert AgentRuntime._publish is original_publish and transport._HUB_PUSH == original_address
    report = _write_report(output)
    assert report["execution_status"] == "completed"
    assert report["error_events"]["total"] == int(name == "lab-instrument-monitoring")
    assert report["timing_comparable"] is (name != "lab-instrument-monitoring")
    if name == "lab-instrument-monitoring":
        assert len(sinks) == 2


@pytest.mark.asyncio
async def test_failed_worker_restores_instrumentation_and_reports_failure(tmp_path, monkeypatch):
    from _sample_eval import _replay as replay
    from xr_ai_runtime import AgentRuntime

    original = AgentRuntime._publish

    async def fail(_cfg, *, ready_file):
        raise RuntimeError("test readiness failure")

    monkeypatch.setattr(replay, "_worker", lambda *_args: (SimpleNamespace(run_app=fail), None))
    scenario = _short_scenario()
    cache = tmp_path / "audio"
    _pin(cache, scenario)
    models = tmp_path / "models.json"
    models.write_text('{"models": {}}')
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(RuntimeError, match="readiness failure"):
        await replay._run(
            scenario, replay._preflight(scenario, models, cache, {}), output, "baseline", {}, ready_timeout=5
        )
    assert AgentRuntime._publish is original
    assert _write_report(output)["execution_status"] == "failed"


@pytest.mark.parametrize("name", _SAMPLES)
def test_builtin_scenes_render_with_native_renderer(name):
    from _sample_eval._fixtures import _renderer

    render = _renderer()
    for event in _builtin(name).events:
        scene = event.get("scene")
        if scene and "image" not in scene:
            image = render(scene)
            assert image.mode == "RGB" and min(image.size) > 0


def _report_fixture(directory, scenario, rows):
    from dataclasses import asdict

    (directory / "manifest.json").write_text(json.dumps({"label": "test", "error_observer": "loguru-error-events"}))
    (directory / "scenario.json").write_text(json.dumps(asdict(scenario)))
    if not any(row["event"] == "measurement_finished" for row in rows):
        rows = [*rows, {"event": "measurement_finished", "elapsed": max(
            (row["elapsed"] for row in rows if row.get("elapsed") is not None), default=0
        )}]
    (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.asyncio
async def test_recorder_links_native_request_metadata_to_late_transcripts(tmp_path):
    import time

    from _sample_eval._recording import _Recording
    from xr_ai_interaction import ResponseOutput
    from xr_ai_runtime import Agent, AgentRuntime, Topic, subscribe
    from xr_ai_voice import VOICE_OUTPUT_TOPIC

    topic = Topic("test.query", UserQuery)

    class Answer(Agent):
        @subscribe(topic)
        async def answer(self, message: UserQuery, ctx):
            await ctx.publish(
                VOICE_OUTPUT_TOPIC,
                ResponseOutput(
                    kind="reply", text="Answer", request_id=ctx.metadata.message_id, response_id="main", final=True
                ),
            )

    with _Recording(tmp_path) as recording:
        recording.started = time.monotonic() - 2
        recording.emit("speech_input", elapsed=0, text="first", scheduled_s=0, audio_start_us=0, audio_end_us=100000)
        recording.emit("speech_input_end", elapsed=0.1, text="first", scheduled_s=0.1)
        recording.emit(
            "speech_input", elapsed=1, text="second", scheduled_s=1, audio_start_us=1000000, audio_end_us=1100000
        )
        recording.emit("speech_input_end", elapsed=1.1, text="second", scheduled_s=1.1)
        runtime = AgentRuntime()
        runtime.register("answer", Answer())
        async with runtime:
            await runtime.publish(
                topic, UserQuery(text="FIRST!", timestamp_us=0), participant_id="fixture-speaker", source="test"
            )
            await runtime.publish(
                topic, UserQuery(text="second", timestamp_us=1000000), participant_id="fixture-speaker", source="test"
            )
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    turns = _turns(rows, 3)
    assert all(turn["request_ids"] and turn["queries"] for turn in turns)
    assert turns[0]["completed_reply_text_s"] > 1.9
    assert turns[1]["completed_reply_text_s"] > 0.9


def test_preflight_uses_resolved_yaml_models_and_leaves_output_unclaimed(tmp_path, monkeypatch):
    from _sample_eval._replay import _preflight

    models = tmp_path / "models.yaml"
    models.write_text(
        "llm:\n  adapter:\n    preset: nemotron_omni\n  endpoint:\n    base_url: http://user:password@example.test/v1?token=secret\n"
    )
    scenario = _short_scenario()
    cache = tmp_path / "audio"
    _pin(cache, scenario)
    inputs = _preflight(scenario, models, cache, {})
    assert inputs["models"]["llm"]["served_alias"]
    assert inputs["models"]["llm"]["endpoint_host"] == "example.test"
    assert "password" not in json.dumps(inputs["models"])
    with pytest.raises(ValueError, match="image bindings"):
        _preflight(scenario, models, cache, {"unexpected": tmp_path / "missing.jpg"})
    import sys

    from _sample_eval.__main__ import _main

    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sample_eval.py",
            "run",
            "--sample",
            "conversation",
            "--models-config",
            str(models),
            "--audio-dir",
            str(cache),
            "--label",
            "test",
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="missing pinned audio"):
        _main()
    assert not output.exists()


def test_cli_prepare_authenticates_hosted_tts_from_cached_credentials(tmp_path, monkeypatch):
    import sys

    import httpx
    import xr_ai_launcher._credentials as credentials
    from _sample_eval import __main__ as cli

    cached = tmp_path / "credentials.json"
    cached.write_text(json.dumps({"XR_EVAL_TEST_API_KEY": "cached-test-token"}))
    monkeypatch.setattr(credentials, "_CREDS_FILE", cached)
    monkeypatch.setattr(credentials, "_HF_TOKEN_FILE", tmp_path / "missing-hf-token")
    monkeypatch.delenv("XR_EVAL_TEST_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({"tts": {
            "category": "tts",
            "adapter": {"kind": "openai_compat"},
            "endpoint": {"base_url": "https://speech.example.test", "api_key_env": "XR_EVAL_TEST_API_KEY"},
        }})
    )
    requests = []

    def respond(request):
        requests.append(request)
        assert str(request.url) == "https://speech.example.test/v1/audio/speech"
        assert request.headers["authorization"] == "Bearer cached-test-token"
        assert json.loads(request.content)["input"] == "hello"
        return httpx.Response(200, content=_wav())

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(respond))
    )
    monkeypatch.setattr(cli, "_builtin", lambda name: _short_scenario())
    audio = tmp_path / "audio"
    monkeypatch.setattr(
        sys, "argv", [
            "run_sample_eval.py", "prepare", "--sample", "conversation",
            "--models-config", str(models), "--audio-dir", str(audio),
        ]
    )
    cli._main()
    assert len(requests) == 1
    assert _load_audio(_short_scenario(), audio)[0]["hello"].size


def test_split_transcripts_and_delayed_replies_share_their_original_input():
    rows = [
        {
            "event": "speech_input",
            "elapsed": 0,
            "text": "Blue rectangle. First step.",
            "audio_start_us": 0,
            "audio_end_us": 500000,
        },
        {"event": "speech_input_end", "elapsed": 0.5},
        {"event": "speech_input", "elapsed": 1, "text": "Next.", "audio_start_us": 1000000, "audio_end_us": 1100000},
        {"event": "speech_input_end", "elapsed": 1.1},
        {
            "event": "request",
            "elapsed": 2,
            "request_id": "a",
            "message": {"text": "Blue rectangle.", "timestamp_us": 100000},
        },
        {
            "event": "request",
            "elapsed": 2.1,
            "request_id": "b",
            "message": {"text": "First step.", "timestamp_us": 300000},
        },
        {"event": "request", "elapsed": 2.2, "request_id": "c", "message": {"text": "Next.", "timestamp_us": 1050000}},
    ]
    for request, at in [("a", 3), ("b", 4), ("c", 5)]:
        rows.append(
            {
                "event": "runtime",
                "elapsed": at,
                "topic": "voice.output",
                "source": "agent",
                "message": {"request_id": request, "response_id": request, "text": "answer", "final": True},
            }
        )
    turns = _turns(rows, 10)
    assert turns[0]["request_ids"] == ["a", "b"] and len(turns[0]["queries"]) == 2
    assert turns[0]["completed_reply_text_s"] == 3.5
    assert turns[1]["completed_reply_text_s"] == 3.9
    rows[-2]["message"]["final"] = False
    turns = _turns(rows, 10)
    assert turns[0]["completed_reply_text_s"] is None


@pytest.mark.parametrize("transcript", ["Next.", "7 times 8", "Task 2"])
def test_source_timestamps_attribute_later_repeated_input_without_spelling_dependency(transcript):
    rows = [
        {"event": "speech_input", "elapsed": 90, "text": "Next.", "audio_start_us": 90000000, "audio_end_us": 90500000},
        {"event": "speech_input_end", "elapsed": 90.5},
        {
            "event": "speech_input",
            "elapsed": 180,
            "text": "Next.",
            "audio_start_us": 180000000,
            "audio_end_us": 180500000,
        },
        {"event": "speech_input_end", "elapsed": 180.5},
        {
            "event": "request",
            "elapsed": 200,
            "request_id": "b",
            "message": {"text": transcript, "timestamp_us": 180100000},
        },
        {
            "event": "runtime",
            "elapsed": 201,
            "topic": "voice.output",
            "source": "agent",
            "message": {"request_id": "b", "response_id": "b", "text": "answer", "final": True},
        },
    ]
    turns = _turns(rows, 300)
    assert turns[0]["request_ids"] == [] and turns[0]["completed_reply_text_s"] is None
    assert turns[1]["request_ids"] == ["b"] and turns[1]["completed_reply_text_s"] == 20.5
    assert turns[1]["queries"][0]["text"] == transcript
    rows[-2]["message"]["timestamp_us"] = 120000000
    turns = _turns(rows, 300)
    assert not any(turn["request_ids"] for turn in turns)


@pytest.mark.parametrize("lead_us", [0, 80000, 160000, 160001])
def test_speaker_hop_timestamp_lead_is_bounded(lead_us):
    rows = [
        {"event": "speech_input", "elapsed": 90, "text": "Next.", "audio_start_us": 90000000, "audio_end_us": 90500000},
        {"event": "speech_input_end", "elapsed": 90.5},
        {
            "event": "request",
            "elapsed": 100,
            "request_id": "speaker",
            "message": {"text": "Next.", "timestamp_us": 90000000 - lead_us},
        },
        {
            "event": "runtime",
            "elapsed": 101,
            "topic": "voice.output",
            "source": "agent",
            "message": {"request_id": "speaker", "response_id": "speaker", "text": "answer", "final": True},
        },
    ]
    turns = _turns(rows, 300)
    assert bool(turns[0]["request_ids"]) == (lead_us <= 160000)
    assert turns[0]["completed_reply_text_s"] == (10.5 if lead_us <= 160000 else None)


def test_overlapping_source_windows_remain_unattributed(tmp_path):
    rows = [
        {
            "event": "speech_input",
            "elapsed": 90,
            "text": "First",
            "duration_s": 0.5,
            "audio_start_us": 90000000,
            "audio_end_us": 90500000,
        },
        {"event": "speech_input_end", "elapsed": 90.5},
        {
            "event": "speech_input",
            "elapsed": 90.6,
            "text": "Second",
            "duration_s": 0.5,
            "audio_start_us": 90600000,
            "audio_end_us": 91100000,
        },
        {"event": "speech_input_end", "elapsed": 91.1},
        {
            "event": "request",
            "elapsed": 95,
            "request_id": "ambiguous",
            "message": {"text": "Second", "timestamp_us": 90460000},
        },
    ]
    turns = _turns(rows, 300)
    assert not any(turn["request_ids"] for turn in turns)
    rows.extend(
        [
            {
                "event": "request",
                "elapsed": 96,
                "request_id": "outside",
                "message": {"text": "Next.", "timestamp_us": 89000000},
            },
            {"event": "session_end", "elapsed": 300, "max_sender_lateness_s": 0.01},
            {"event": "run_finished", "elapsed": 310, "status": "completed"},
        ]
    )
    _report_fixture(tmp_path, _builtin("conversation"), rows)
    assert _write_report(tmp_path)["unattributed_requests"] == 2


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("mode", ["blocked_worker", "source_dies", "worker_dies"])
async def test_source_keeps_real_time_and_both_failure_directions_clean_up(tmp_path, monkeypatch, mode):
    import os
    import signal
    import time
    from dataclasses import replace
    from multiprocessing.shared_memory import SharedMemory

    import xr_ai_voice._transport as transport
    from _sample_eval import _replay as replay
    from xr_ai_hub import ProcessorEndpoint

    scenario = replace(_short_scenario(), duration_s=0.2)
    cache = tmp_path / "audio"
    _pin(cache, scenario)
    models = tmp_path / "models.json"
    models.write_text('{"models":{}}')
    output = tmp_path / "run"
    output.mkdir()
    closed, chunks, blocked, source_names = [], [], [], []

    async def app(_cfg, *, ready_file):
        source_names.append(Path(transport._HUB_PUSH.removeprefix("ipc://")).parent.name.replace("-", "_"))
        endpoint = ProcessorEndpoint(push_addr=transport._HUB_PUSH, sub_addr=transport._HUB_PUB)
        failure = asyncio.Event()

        async def audio(chunk):
            first = not chunks
            chunks.append(chunk)
            if first:
                if mode == "blocked_worker":
                    time.sleep(0.8)
                    blocked.append(time.monotonic())
                elif mode == "source_dies":
                    source_pid = json.loads((output / "source-process.json").read_text())["pid"]
                    os.kill(source_pid, signal.SIGKILL)
                else:
                    failure.set()

        endpoint.on_audio(audio)
        task = asyncio.create_task(endpoint.run())
        try:
            await endpoint.wait_until_running()
            assert await endpoint.wait_for_subscriptions(timeout=2)
            ready_file.touch()
            if mode == "worker_dies":
                await failure.wait()
                raise RuntimeError("worker failed during replay")
            await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            endpoint.stop()
            endpoint.close()
            closed.append(True)

    monkeypatch.setattr(replay, "_worker", lambda *_: (SimpleNamespace(run_app=app), None))
    monkeypatch.setattr(replay, "_RESPONSE_DRAIN_S", 1)
    args = (scenario, replay._preflight(scenario, models, cache, {}), output, "test", {})
    unrelated = SharedMemory(create=True, size=1)
    try:
        unrelated.buf[0] = 42
        if mode == "blocked_worker":
            await replay._run(*args, ready_timeout=5)
            rows = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            end = next(row for row in rows if row["event"] == "session_end")
            assert end["monotonic"] < blocked[0]
            assert end["max_sender_lateness_s"] <= 0.1
            assert len(chunks) == 10 and sum(chunk.samples for chunk in chunks) == 3200
            assert _write_report(output)["execution_status"] == "completed"
        else:
            with pytest.raises(RuntimeError, match="disconnected|worker failed"):
                await replay._run(*args, ready_timeout=5)
            assert _write_report(output)["execution_status"] == "failed"
        assert closed == [True]
        source_pid = json.loads((output / "source-process.json").read_text())["pid"]
        with pytest.raises(ProcessLookupError):
            os.kill(source_pid, 0)
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=source_names[0], create=False)
        reopened = SharedMemory(name=unrelated.name, create=False)
        try:
            assert reopened.buf[0] == 42
        finally:
            reopened.close()
    finally:
        unrelated.close()
        unrelated.unlink()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("phase", ["before_join", "during_replay"])
async def test_source_process_exits_when_owner_control_closes(tmp_path, phase):
    import multiprocessing
    import time

    from _sample_eval._fixtures import _load_images, _renderer
    from _sample_eval._source import _main, _receive

    scenario = _short_scenario()
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    render = _renderer()
    source = ctx.Process(
        target=_main,
        args=(
            scenario,
            {"hello": _decode_audio(_wav())},
            _load_images(scenario, {})[0],
            {json.dumps({"rectangle": "blue"}, sort_keys=True): render({"rectangle": "blue"})},
            tmp_path,
            f"ipc://{tmp_path}/push",
            f"ipc://{tmp_path}/pub",
            tmp_path.name[-30:],
            child,
            1,
        ),
    )
    try:
        source.start()
        child.close()
        assert await asyncio.wait_for(_receive(parent), 5) == "ready"
        if phase == "during_replay":
            parent.send({"command": "join"})
            assert await asyncio.wait_for(_receive(parent), 5) == "joined"
            parent.send(
                {"command": "start", "monotonic": time.monotonic() + 0.1, "epoch_us": time.time_ns() // 1000 + 100000}
            )
            assert await asyncio.wait_for(_receive(parent), 5) == "started"
        parent.close()
        async with asyncio.timeout(5):
            while source.is_alive():
                await asyncio.sleep(0.05)
        source.join()
        assert source.exitcode != 0
    finally:
        parent.close()
        child.close()
        if source.is_alive():
            source.kill()
            source.join()
        source.close()


def test_lab_speech_camera_fixtures_have_detectable_device_markers(monkeypatch):
    import importlib.util

    from PIL import Image
    from xr_ai_tools.marker_tracking import _extract_qr_markers, _full_frame_scans, _image_pixels

    sample = Path(__file__).resolve().parents[1] / "agent-samples/lab-instrument-monitoring"
    monkeypatch.syspath_prepend(str(sample / "worker"))
    spec = importlib.util.spec_from_file_location("lab_visual_fixtures", sample / "eval/visual_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = []
    for reading in ("99.0 A", "25.0 A"):
        raw = module._instrument_scene(left_reading="12.0 V", right_reading=reading, annotate=False)
        frames.append(raw)
        pixels = _image_pixels(Image.open(io.BytesIO(raw)))
        markers = _extract_qr_markers(_full_frame_scans(pixels))
        assert {marker.value for marker in markers} == {"device-1", "device-2"}
    assert frames[0] != frames[1]


@pytest.mark.parametrize("scene", [
    {"rectangel": "blue"}, {"kind": "packet", "blend": "Tea", "colour": "red"},
    {"kind": "kettle", "readings": [{"label": "water", "value": "20 C", "typo": True}]},
])
def test_renderer_rejects_unknown_scene_keys(scene):
    from _sample_eval._fixtures import _renderer

    with pytest.raises(ValueError, match="scene|readings"):
        _renderer()(scene)


def test_unknown_replay_name_is_actionable():
    with pytest.raises(ValueError, match="unknown replay.*choose one"):
        _builtin("missing-replay")


def test_fast_error_reply_is_not_a_comparable_result(tmp_path, monkeypatch, capsys):
    import sys
    import time

    from _sample_eval import __main__ as cli
    from _sample_eval import _replay as replay
    from _sample_eval._recording import _Recording
    from loguru import logger

    with _Recording(tmp_path, filename="observed.jsonl") as recording:
        logger.error("startup error outside the measurement")
        recording.started = time.monotonic()
        logger.remove(recording._error_sink)
        recording.observe_errors()
        logger.patch(lambda record: record.update(name="xr_ai_models._openai_compat")).error("private HTTP body")
        logger.opt(exception=ValueError("private exception detail")).error("handler fallback")
    observed = [json.loads(line) for line in (tmp_path / "observed.jsonl").read_text().splitlines()]
    assert "private" not in (tmp_path / "observed.jsonl").read_text()
    rows = observed + [
        {"event": "speech_input", "elapsed": 0, "text": "hello", "audio_start_us": 0, "audio_end_us": 20000,
         "duration_s": 0.02},
        {"event": "speech_input_end", "elapsed": 0.02},
        {"event": "request", "elapsed": 0.03, "request_id": "r", "message": {"timestamp_us": 10000}},
        {"event": "runtime", "elapsed": 0.04, "topic": "voice.output", "message": {
            "request_id": "r", "response_id": "reply", "final": True, "text": "I couldn't complete that request.",
        }},
        {"event": "session_end", "elapsed": 0.08, "max_sender_lateness_s": 0.001},
        {"event": "run_finished", "elapsed": 0.09, "status": "completed"},
    ]
    output = tmp_path / "run"

    async def run(scenario, _inputs, directory, *_args):
        _report_fixture(directory, scenario, rows)

    monkeypatch.setattr(cli, "load_credentials", lambda: None)
    monkeypatch.setattr(cli, "_builtin", lambda _name: _short_scenario())
    monkeypatch.setattr(replay, "_preflight", lambda *_args: {"worker_settings": None})
    monkeypatch.setattr(replay, "_run", run)
    monkeypatch.setattr(sys, "argv", [
        "run_sample_eval.py", "run", "--sample", "conversation", "--models-config", "unused.json",
        "--audio-dir", str(tmp_path), "--output-dir", str(output), "--label", "broken",
    ])
    cli._main()
    result = json.loads((output / "report.json").read_text())
    assert result["execution_status"] == "completed" and result["timing_comparable"] is False
    assert result["completed_reply_text_s"]["median"] == pytest.approx(0.02)
    summary = json.loads(capsys.readouterr().out)
    assert summary["error_events"] == {"model": 1, "runtime": 1, "total": 2}
    assert summary["handler_coverage"] == {"received": 1, "expected": 1}


def test_preflight_hashes_worker_guide_and_effective_voice_config(tmp_path):
    from dataclasses import replace

    import yaml
    from _sample_eval._replay import _preflight, _worker

    scenario = _builtin("guide")
    source = Path(scenario.source)
    native = (source.parent / scenario.worker["config"]).resolve()
    raw = yaml.safe_load(native.read_text())
    raw["guide_config"] = str(native.parent / "guide-agent-config.yaml")
    worker = tmp_path / "worker.yaml"
    worker.write_text(yaml.safe_dump(raw))
    guide = tmp_path / "guide.yaml"
    guide.write_bytes((source.parent / scenario.worker["launch"]["guide_path"]).read_bytes())
    scenario = replace(scenario, worker={**scenario.worker, "launch": {"mode": "guide", "guide_path": str(guide)}})
    cache = tmp_path / "audio"
    _pin(cache, scenario)
    models = tmp_path / "models.json"
    models.write_text("{}")
    voice = native.parent / "voice_gate.yaml"
    before = _preflight(scenario, models, cache, {}, voice, worker)
    raw["silence_duration"] = before["worker_settings"].silence_duration + 0.1
    worker.write_text(yaml.safe_dump(raw))
    guide.write_text(guide.read_text() + "\n# changed guide bytes\n")
    after = _preflight(scenario, models, cache, {}, voice, worker)
    assert before["worker_config_sha256"] != after["worker_config_sha256"]
    assert before["guide_sha256"] != after["guide_sha256"]
    assert before["voice_gate_sha256"] == after["voice_gate_sha256"]
    worker.write_text("invalid: [")
    _app, cfg = _worker(scenario, tmp_path, models, voice, None, before["worker_settings"])
    assert cfg.silence_duration == before["worker_settings"].silence_duration


@pytest.mark.parametrize("sample", ["lab-instrument-monitoring", "capture"])
def test_native_config_failure_does_not_claim_output(tmp_path, monkeypatch, sample):
    import sys
    from dataclasses import replace

    from _sample_eval import __main__ as cli

    scenario = _builtin(sample)
    args = []
    if sample == "capture":
        scenario = replace(scenario, worker={**scenario.worker, "launch": {"mode": "invalid"}})
        error = ValueError
    else:
        invalid = tmp_path / "lab.yaml"
        invalid.write_text("device_map_yaml: absent-device-map.yaml\n")
        args = ["--worker-config", str(invalid)]
        error = FileNotFoundError
    cache = tmp_path / "audio"
    _pin(cache, scenario)
    models = tmp_path / "models.json"
    models.write_text("{}")
    output = tmp_path / "run"
    monkeypatch.setattr(cli, "load_credentials", lambda: None)
    monkeypatch.setattr(cli, "_builtin", lambda _name: scenario)
    monkeypatch.setattr(sys, "argv", [
        "run_sample_eval.py", "run", "--sample", sample, "--models-config", str(models),
        "--audio-dir", str(cache), "--output-dir", str(output), "--label", "invalid", *args,
    ])
    with pytest.raises(error):
        cli._main()
    assert not output.exists()


def test_scene_replay_requires_explicit_native_endpoints(tmp_path):
    import yaml
    from _sample_eval._replay import _preflight

    scenario = _builtin("xr-render-demo")
    native = (Path(scenario.source).parent / scenario.worker["config"]).resolve()
    cache = tmp_path / "audio"
    _pin(cache, scenario)
    models = tmp_path / "models.json"
    models.write_text("{}")
    with pytest.raises(ValueError, match="dedicated scene_endpoint"):
        _preflight(scenario, models, cache, {})
    raw = yaml.safe_load(native.read_text())
    raw["voice_gate_yaml"] = str(native.parent / "voice_gate.yaml")
    worker = tmp_path / "worker.yaml"
    for host in ("127.0.0.1", "127.0.0.2", "localhost", "[::1]", "127.1", "0",
                 "[::ffff:127.0.0.1]", "2130706433", "0x7f000001", "my-spark", "192.168.1.42"):
        raw["scene_endpoint"] = f"tcp://{host}:8320"
        worker.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="not default ports"):
            _preflight(scenario, models, cache, {}, worker_config=worker)
    raw["scene_endpoint"] = "tcp://127.0.0.1:18320"
    worker.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="openxr_endpoint.*8330.*not default ports"):
        _preflight(scenario, models, cache, {}, worker_config=worker)
    for key in ("scene_endpoint", "openxr_endpoint"):
        invalid = {**raw, "openxr_endpoint": "tcp://127.0.0.1:18330", key: "tcp://127.0.0.1:8320abc"}
        worker.write_text(yaml.safe_dump(invalid))
        with pytest.raises(ValueError, match=key + ".*8320abc"):
            _preflight(scenario, models, cache, {}, worker_config=worker)
    raw.update(scene_endpoint="tcp://127.0.0.1:18320", openxr_endpoint="tcp://127.0.0.1:18330")
    worker.write_text(yaml.safe_dump(raw))
    inputs = _preflight(scenario, models, cache, {}, worker_config=worker)
    assert inputs["worker_settings"].scene_endpoint == raw["scene_endpoint"]
    del raw["openxr_endpoint"]
    worker.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="dedicated scene_endpoint"):
        _preflight(scenario, models, cache, {}, worker_config=worker)
    worker.write_text("")
    with pytest.raises(ValueError, match="dedicated scene_endpoint"):
        _preflight(scenario, models, cache, {}, worker_config=worker)


@pytest.mark.asyncio
@pytest.mark.parametrize("objects", [[], [{"id": "old-object"}]])
async def test_scene_preflight_checks_real_rpc_state(objects):
    import uuid

    from _sample_eval._replay import _check_scene
    from xr_ai_tools.rpc import RPCServer

    endpoint = f"ipc:///tmp/eval-scene-{uuid.uuid4().hex}"
    ready = asyncio.Event()
    server = RPCServer(endpoint, lambda _op, _args: {"objects": objects})
    task = asyncio.create_task(server.serve(ready=ready.set))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        inputs = {"worker_settings": SimpleNamespace(scene_endpoint=endpoint)}
        if objects:
            with pytest.raises(ValueError, match="empty dedicated scene"):
                await _check_scene(inputs)
        else:
            await _check_scene(inputs)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_cannot_connect_to_a_foreign_hub(tmp_path, monkeypatch):
    from _sample_eval import _replay as replay
    from xr_ai_hub import ProcessorEndpoint

    attempts = []

    def unexpected_connect(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("foreign endpoint reached SDK constructor")

    monkeypatch.setattr(ProcessorEndpoint, "__init__", unexpected_connect)

    async def app(_cfg, **_kwargs):
        ProcessorEndpoint(sub_addr="ipc:///tmp/xr_hub_pub", push_addr="ipc:///tmp/xr_hub_in")

    monkeypatch.setattr(replay, "_worker", lambda *_: (SimpleNamespace(run_app=app), None))
    scenario = _short_scenario()
    cache = tmp_path / "audio"
    _pin(cache, scenario)
    models = tmp_path / "models.json"
    models.write_text("{}")
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="private run hub"):
        await replay._run(scenario, replay._preflight(scenario, models, cache, {}), output, "test", {}, ready_timeout=5)
    assert attempts == [] and ProcessorEndpoint.__init__ is unexpected_connect


@pytest.mark.parametrize("readings", ["water", [["label", "value"]], [{}], [None]])
def test_kettle_readings_require_mappings(readings):
    from _sample_eval._fixtures import _renderer

    with pytest.raises(ValueError, match="readings must be a list of mappings"):
        _renderer()({"kind": "kettle", "readings": readings})


def test_observer_catches_quiet_tool_fallbacks_and_exception_warnings(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from _sample_eval._recording import _Recording
    from loguru import logger
    from xr_ai_tools.tool_calling import ToolLoopIterationLimitError

    original = ToolLoopIterationLimitError.__init__
    with _Recording(tmp_path) as recording:
        recording.started = time.monotonic()
        # Simulates the quiet typed fallback used by XR, Lab and Tea handlers.
        try:
            raise ToolLoopIterationLimitError("private response", messages=[], tool_calls=[], iterations=4)
        except ToolLoopIterationLimitError:
            pass
        logger.warning("ordinary warning is not a failure")
        logger.opt(exception=True).warning("no exception warning is not a failure")
        logger.exception("private error without active exception")
        logger.opt(exception=ValueError("private scene detail")).warning("caption failed")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda n: recording.emit("thread", number=n), range(400)))
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert ToolLoopIterationLimitError.__init__ is original
    assert len([row for row in rows if row["event"] == "thread"]) == 400
    assert [row["exception_type"] for row in rows if row["event"] == "error"] == [
        "ToolLoopIterationLimitError", None, "ValueError"
    ]
    assert "private" not in (tmp_path / "events.jsonl").read_text()


def test_loaded_settings_hash_nested_configs_and_path_contents(tmp_path):
    from dataclasses import replace

    from _sample_eval._replay import _load_worker_config, _settings_identity

    capture = _load_worker_config(_builtin("capture"))[0]
    before = _settings_identity(capture)
    changed = replace(capture, capture=replace(capture.capture, caption_interval_s=99))
    assert before["worker_settings_sha256"] != _settings_identity(changed)["worker_settings_sha256"]
    tea = _load_worker_config(_builtin("tea"))[0]
    workflow = tmp_path / "workflow.yaml"
    workflow.write_bytes(tea.workflow_config.read_bytes())
    tea = replace(tea, workflow_config=workflow)
    before = _settings_identity(tea)
    workflow.write_text(workflow.read_text() + "\n# changed workflow\n")
    assert before["worker_settings_sha256"] != _settings_identity(tea)["worker_settings_sha256"]
    lab = _builtin("lab-instrument-monitoring")
    before = _settings_identity(_load_worker_config(lab)[0])
    assert before == _settings_identity(_load_worker_config(lab)[0])
    assert any("DeviceMap" in name for name in before["unserialized_worker_settings"])


@pytest.mark.asyncio
async def test_run_rejects_concurrent_runs_and_releases_guard(monkeypatch):
    from _sample_eval import _replay as replay

    started, release = asyncio.Event(), asyncio.Event()

    async def run():
        started.set()
        await release.wait()

    monkeypatch.setattr(replay, "_run_isolated", run)
    first = asyncio.create_task(replay._run())
    await started.wait()
    with pytest.raises(RuntimeError, match="only one sample replay"):
        await replay._run()
    release.set()
    await first
    await replay._run()


@pytest.mark.parametrize("slow_server", [False, True])
def test_unavailable_scene_fails_before_cli_claims_output(tmp_path, monkeypatch, slow_server):
    import sys
    import uuid

    from _sample_eval import __main__ as cli
    from _sample_eval import _replay as replay
    from xr_ai_tools.rpc import RPCError, RPCServer

    endpoint = f"ipc:///tmp/eval-missing-{uuid.uuid4().hex}"
    output = tmp_path / "run"
    check = replay._check_scene

    async def checked(inputs):
        async def slow(_op, _args):
            await asyncio.sleep(10)
            return {"objects": []}

        task = None
        if slow_server:
            ready = asyncio.Event()
            task = asyncio.create_task(RPCServer(endpoint, slow).serve(ready=ready.set))
            await asyncio.wait_for(ready.wait(), 5)
        try:
            await check(inputs, timeout_s=0.05)
        finally:
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(cli, "load_credentials", lambda: None)
    monkeypatch.setattr(replay, "_preflight", lambda *_: {
        "worker_settings": SimpleNamespace(scene_endpoint=endpoint)
    })
    monkeypatch.setattr(replay, "_check_scene", checked)
    monkeypatch.setattr(sys, "argv", [
        "run_sample_eval.py", "run", "--sample", "xr-render-demo", "--models-config", "unused.json",
        "--audio-dir", str(tmp_path), "--output-dir", str(output), "--label", "invalid",
    ])
    with pytest.raises(RPCError) as exc:
        cli._main()
    assert exc.value.code == "timeout"
    assert not output.exists()


def test_partial_recorder_setup_restores_hooks_and_closes_file(tmp_path, monkeypatch):
    from _sample_eval._recording import _Recording
    from loguru import logger
    from xr_ai_runtime import AgentRuntime
    from xr_ai_tools.tool_calling import ToolLoopError

    original = ToolLoopError.__init__
    recording = _Recording(tmp_path)
    patch = recording.patch

    def fail(owner, name, value):
        if owner is AgentRuntime:
            raise ImportError("injected setup failure")
        patch(owner, name, value)

    monkeypatch.setattr(recording, "patch", fail)
    with pytest.raises(ImportError, match="injected setup failure"), recording:
        pass
    assert ToolLoopError.__init__ is original
    assert recording._file.closed
    with pytest.raises(ValueError):
        logger.remove(recording._error_sink)


def test_lab_device_map_file_is_fingerprinted(tmp_path):
    import yaml
    from _sample_eval._replay import _load_worker_config

    scenario = _builtin("lab-instrument-monitoring")
    native = (Path(scenario.source).parent / scenario.worker["config"]).resolve()
    raw = yaml.safe_load(native.read_text())
    mapping = tmp_path / "device_map.yaml"
    mapping.write_bytes((native.parent / raw["device_map_yaml"]).read_bytes())
    raw["device_map_yaml"] = str(mapping)
    worker = tmp_path / "worker.yaml"
    worker.write_text(yaml.safe_dump(raw))
    before = _load_worker_config(scenario, worker)
    mapping.write_text(mapping.read_text() + "\n# changed device map bytes\n")
    after = _load_worker_config(scenario, worker)
    assert before[1] == after[1]
    assert before[3] != after[3]
