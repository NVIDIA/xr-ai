# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run native sample workers against private test IPC and caller-owned services."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import hashlib
import importlib
import json
import multiprocessing
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import yaml

from ._fixtures import _load_audio, _load_images, _renderer
from ._recording import _Recording
from ._scenario import _ROOT, _Scenario

_RESPONSE_DRAIN_S = 10
_RUN_LOCK = threading.Lock()


def _model_description(config) -> dict:
    return {
        name: {"served_alias": spec.adapter.model_name, "endpoint_host": urlsplit(spec.endpoint.base_url).hostname}
        for name, spec in config.entries.items()
    }


def _settings_identity(settings) -> dict:
    opaque = set()

    def encode(value):
        if isinstance(value, Path):
            path = value.resolve()
            return {
                "path": str(path.relative_to(_ROOT)) if path.is_relative_to(_ROOT) else str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            }
        name = type(value).__module__ + "." + type(value).__qualname__
        opaque.add(name)
        return {"opaque_type": name}

    payload = json.dumps(
        dataclasses.asdict(settings) if settings is not None else None, default=encode, sort_keys=True
    )
    return {
        "worker_settings_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "unserialized_worker_settings": sorted(opaque),
    }


def _preflight(scenario, models, audio_dir, images, voice_gate=None, worker_config=None) -> dict:
    from xr_ai_models import load_models_config_from_dict

    raw = models.read_bytes()
    settings, worker_hash, guide_hash, device_map_hash = (
        _load_worker_config(scenario, worker_config) if scenario.worker else (None, None, None, None)
    )
    if settings is not None and hasattr(settings, "scene_endpoint"):
        explicit = (yaml.safe_load(worker_config.read_bytes()) or {}) if worker_config else {}
        if worker_config is None or not all(explicit.get(key) for key in ("scene_endpoint", "openxr_endpoint")):
            raise ValueError(
                "scene replay requires --worker-config with explicit dedicated scene_endpoint and openxr_endpoint"
            )
        for key, default_port in (("scene_endpoint", 8320), ("openxr_endpoint", 8330)):
            value = getattr(settings, key)
            try:
                port = urlsplit(value).port
            except ValueError as exc:
                raise ValueError(f"invalid {key} endpoint {value!r}: {exc}") from exc
            if port == default_port:
                raise ValueError(f"{key} endpoint {value!r} must use dedicated service ports, not default ports")
    effective_gate = voice_gate or getattr(settings, "voice_gate_yaml", None)
    gate_bytes = effective_gate.read_bytes() if effective_gate else None
    config = load_models_config_from_dict(yaml.safe_load(raw) or {}, source=str(models))
    audio, audio_hashes = _load_audio(scenario, audio_dir)
    pictures, image_hashes = _load_images(scenario, images)
    render = _renderer()
    scenes = {json.dumps({"rectangle": "blue"}, sort_keys=True): render({"rectangle": "blue"})}
    for event in scenario.events:
        if "scene" in event and "image" not in event["scene"]:
            scenes[json.dumps(event["scene"], sort_keys=True)] = render(event["scene"])
    return dict(
        models_bytes=raw,
        voice_gate_bytes=gate_bytes,
        audio=audio,
        audio_hashes=audio_hashes,
        pictures=pictures,
        image_hashes=image_hashes,
        scenes=scenes,
        models=_model_description(config),
        models_config_sha256=hashlib.sha256(raw).hexdigest(),
        voice_gate_override_sha256=hashlib.sha256(gate_bytes).hexdigest() if voice_gate else None,
        voice_gate_sha256=hashlib.sha256(gate_bytes).hexdigest() if gate_bytes is not None else None,
        worker_config_sha256=worker_hash,
        guide_sha256=guide_hash,
        device_map_sha256=device_map_hash,
        worker_settings=settings,
        **_settings_identity(settings),
    )


def _load_worker_config(scenario: _Scenario, override: Path | None = None):
    source = Path(scenario.source)
    package = scenario.worker["package"]
    worker_path = str(source.parents[2] / "worker")
    if worker_path not in sys.path:
        sys.path.insert(0, worker_path)
    module = importlib.import_module(package + ".config")
    path = override or (source.parent / scenario.worker["config"]).resolve()
    worker_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if launch := scenario.worker.get("launch"):
        request = {**launch, "worker_config": str(path)}
        if "guide_path" in request:
            request["guide_path"] = str((source.parent / request["guide_path"]).resolve())
        with tempfile.TemporaryDirectory(prefix="xr-eval-config-") as directory:
            launch_path = Path(directory) / "launch.yaml"
            launch_path.write_text(yaml.safe_dump(request))
            cfg = module.load_config(launch_path)
    else:
        cfg = module.load_config(path)
    guide = getattr(getattr(cfg, "guide", None), "guide_path", None)
    device_map_hash = None
    if hasattr(cfg, "device_map"):
        raw = yaml.safe_load(path.read_bytes()) or {}
        device_map = Path(raw.get("device_map_yaml", "device_map.yaml"))
        if not device_map.is_absolute():
            device_map = path.parent / device_map
        device_map_hash = hashlib.sha256(device_map.read_bytes()).hexdigest()
    return cfg, worker_hash, hashlib.sha256(guide.read_bytes()).hexdigest() if guide else None, device_map_hash


async def _check_scene(inputs: dict, *, timeout_s: float = 5) -> None:
    cfg = inputs["worker_settings"]
    if cfg is None or not hasattr(cfg, "scene_endpoint"):
        return
    from xr_ai_tools.rpc import RPCClient

    async with RPCClient(cfg.scene_endpoint, timeout_s=timeout_s) as client:
        state = await client.call("get_scene_state", {})
    if state.get("objects") != []:
        raise ValueError("scene replay requires an empty dedicated scene; get_scene_state did not return objects: []")


def _worker(scenario: _Scenario, output: Path, models: Path, voice_gate: Path | None, rag: str | None, settings=None):
    cfg = settings if settings is not None else _load_worker_config(scenario)[0]
    app = importlib.import_module(scenario.worker["package"] + ".app")
    overrides = {"models_config": models}
    capture = getattr(cfg, "capture", None)
    if capture is not None:
        overrides["capture"] = dataclasses.replace(capture, artifacts_dir=output / "captures")
    if voice_gate is not None:
        overrides["voice_gate_yaml"] = voice_gate
    fields = {field.name for field in dataclasses.fields(cfg)}
    for name, value in {
        "artifacts_dir": output / "artifacts",
        "web_events_port": 0,
        "text_memory_dir": str(output / "text-memory"),
    }.items():
        if name in fields:
            overrides[name] = value
    if rag is not None:
        overrides["rag_endpoint"] = rag
    return app, dataclasses.replace(cfg, **overrides)


def _source_identity() -> dict:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "HEAD", "--"],
        cwd=_ROOT,
        capture_output=True,
        check=False,
    )
    files = sorted(Path(__file__).parent.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    return {
        "revision": result.stdout.strip() if result.returncode == 0 else None,
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
        "harness_sha256": digest.hexdigest(),
    }


async def _replay(
    scenario, audio, images, renderer, connector, worker, recording, *, started=None, audio_epoch_us=None
) -> None:
    from xr_ai_hub import AudioChunk, PixelFormat

    audio_epoch_us = time.time_ns() // 1000 if audio_epoch_us is None else audio_epoch_us
    recording.started = time.monotonic() if started is None else started
    await asyncio.sleep(max(0, recording.started - time.monotonic()))
    recording.emit("session_start")
    current = renderer({"rectangle": "blue"})
    event_index, speech = 0, None
    next_frame, maximum_late = 0.0, 0.0
    for frame in range(round(scenario.duration_s * 50)):
        target = recording.started + frame * 0.02
        await asyncio.sleep(max(0, target - time.monotonic()))
        maximum_late = max(maximum_late, time.monotonic() - target)
        elapsed = frame * 0.02
        while event_index < len(scenario.events) and scenario.events[event_index]["at_s"] <= elapsed:
            event = scenario.events[event_index]
            if "scene" in event:
                scene = event["scene"]
                current = images[scene["image"]] if "image" in scene else renderer(scene)
                recording.emit("scene", scene=scene)
            if "say" in event:
                text = event["say"]
                speech = (frame * 320, text, audio[text])
                recording.emit(
                    "speech_input",
                    text=text,
                    duration_s=len(audio[text]) / 16000,
                    scheduled_s=elapsed,
                    audio_start_us=audio_epoch_us + frame * 20000,
                    audio_end_us=audio_epoch_us + frame * 20000 + round(len(audio[text]) * 1000000 / 16000),
                )
            event_index += 1
        if elapsed >= next_frame:
            await connector.push_frame(
                current.tobytes(),
                current.width,
                current.height,
                PixelFormat.RGB24,
                time.time_ns() // 1000,
                "fixture-speaker",
                "camera",
            )
            next_frame += 1
        chunk = np.zeros(320, dtype="<f4")
        if speech is not None:
            offset = frame * 320 - speech[0]
            part = speech[2][offset : offset + 320]
            chunk[: len(part)] = part
            if offset + 320 >= len(speech[2]):
                recording.emit("speech_input_end", text=speech[1], scheduled_s=elapsed)
                speech = None
        await connector.push_audio(
            AudioChunk(
                pts_us=audio_epoch_us + frame * 20000,
                sample_rate=16000,
                channels=1,
                samples=320,
                data=chunk.tobytes(),
                participant_id="fixture-speaker",
                track_id="microphone",
            )
        )
        if worker.done():
            await worker
            raise RuntimeError("sample worker exited during replay")
    recording.emit("session_end", duration_s=scenario.duration_s, max_sender_lateness_s=maximum_late)


async def _run(*args, **kwargs) -> None:
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("only one sample replay may run per process")
    try:
        await _run_isolated(*args, **kwargs)
    finally:
        _RUN_LOCK.release()


async def _run_isolated(
    scenario: _Scenario,
    inputs: dict,
    output: Path,
    label: str,
    metadata: dict,
    rag: str | None = None,
    ready_timeout: float = 600,
) -> None:
    import xr_ai_voice._transport as transport
    from xr_ai_hub import ProcessorEndpoint

    from ._source import _main as source_main
    from ._source import _receive

    manifest = {
        "label": label,
        "sample": scenario.sample,
        "duration_s": scenario.duration_s,
        "scenario_sha256": scenario.sha256,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "audio_sha256": inputs["audio_hashes"],
        "image_sha256": inputs["image_hashes"],
        "models_config_sha256": inputs["models_config_sha256"],
        "voice_gate_override_sha256": inputs["voice_gate_override_sha256"],
        "voice_gate_sha256": inputs["voice_gate_sha256"],
        "worker_config_sha256": inputs["worker_config_sha256"],
        "guide_sha256": inputs["guide_sha256"],
        "device_map_sha256": inputs["device_map_sha256"],
        "worker_settings_sha256": inputs["worker_settings_sha256"],
        "unserialized_worker_settings": inputs["unserialized_worker_settings"],
        "error_observer": "loguru-and-tool-loop-events",
        "models": inputs["models"],
        "operator_metadata": metadata,
        **_source_identity(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    serialized = dataclasses.asdict(scenario)
    if scenario.source:
        serialized["source"] = str(Path(scenario.source).relative_to(_ROOT))
    (output / "scenario.json").write_text(json.dumps(serialized, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="xr-eval-") as ipc:
        process, worker, connection = None, None, None
        completed = False
        try:
            with _Recording(output, filename="worker-events.jsonl") as recording:
                push, pub = f"ipc://{ipc}/in", f"ipc://{ipc}/pub"
                recording.patch(transport, "_HUB_PUSH", push)
                recording.patch(transport, "_HUB_PUB", pub)
                endpoint_init = ProcessorEndpoint.__init__

                def isolated_endpoint(endpoint, sub_addr, push_addr, **kwargs):
                    if (sub_addr, push_addr) != (pub, push) or kwargs.get("file_sub_addr") is not None:
                        raise ValueError("evaluation worker endpoint must use only the private run hub addresses")
                    return endpoint_init(endpoint, sub_addr=sub_addr, push_addr=push_addr, **kwargs)

                recording.patch(ProcessorEndpoint, "__init__", isolated_endpoint)
                context = multiprocessing.get_context("spawn")
                connection, child = context.Pipe()
                process = context.Process(
                    target=source_main,
                    args=(
                        scenario,
                        inputs["audio"],
                        inputs["pictures"],
                        inputs["scenes"],
                        output,
                        push,
                        pub,
                        Path(ipc).name,
                        child,
                        _RESPONSE_DRAIN_S,
                    ),
                )
                try:
                    process.start()
                    child.close()
                    (output / "source-process.json").write_text(json.dumps({"pid": process.pid}) + "\n")

                    async def signal(expected):
                        received = asyncio.create_task(_receive(connection))
                        try:
                            waiting = [received] + ([worker] if worker is not None else [])
                            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                            if worker is not None and worker in done:
                                await worker
                                raise RuntimeError("sample worker exited during source replay")
                            value = await received
                            if value != expected:
                                raise RuntimeError(f"source expected {expected!r}, received {value!r}")
                        finally:
                            received.cancel()
                            await asyncio.gather(received, return_exceptions=True)

                    async with asyncio.timeout(ready_timeout):
                        await signal("ready")
                    models_snapshot = Path(ipc) / "models.yaml"
                    models_snapshot.touch(mode=0o600)
                    models_snapshot.write_bytes(inputs["models_bytes"])
                    gate_snapshot = None
                    if inputs["voice_gate_bytes"] is not None:
                        gate_snapshot = Path(ipc) / "voice_gate.yaml"
                        gate_snapshot.touch(mode=0o600)
                        gate_snapshot.write_bytes(inputs["voice_gate_bytes"])
                    app, cfg = _worker(scenario, output, models_snapshot, gate_snapshot, rag, inputs["worker_settings"])
                    if hasattr(app, "setup_logging"):
                        setup = app.setup_logging

                        def observed_setup(*args, **kwargs):
                            result = setup(*args, **kwargs)
                            recording.observe_errors()
                            return result

                        recording.patch(app, "setup_logging", observed_setup)
                    for name, address in (("_HUB_PUSH", push), ("_HUB_PUB", pub)):
                        if hasattr(app, name):
                            recording.patch(app, name, address)
                    ready = output / "worker.ready"
                    worker = asyncio.create_task(app.run_app(cfg, ready_file=ready))
                    async with asyncio.timeout(ready_timeout):
                        while not ready.exists():
                            if worker.done():
                                await worker
                                raise RuntimeError("sample worker exited before readiness")
                            if not process.is_alive():
                                raise RuntimeError("source process exited before worker readiness")
                            await asyncio.sleep(0.1)
                    await asyncio.sleep(1)
                    connection.send({"command": "join"})
                    async with asyncio.timeout(ready_timeout):
                        await signal("joined")
                    await asyncio.sleep(1)
                    recording.started = time.monotonic() + 0.25
                    epoch = time.time_ns() // 1000 + round((recording.started - time.monotonic()) * 1000000)
                    connection.send({"command": "start", "monotonic": recording.started, "epoch_us": epoch})
                    await signal("started")

                    await signal("done")
                    completed = True
                finally:
                    recording.emit("measurement_finished")
                    child.close()
                    if worker is not None:
                        worker.cancel()
                        await asyncio.gather(worker, return_exceptions=True)
                    if connection is not None:
                        try:
                            connection.send({"command": "stop"})
                        except (BrokenPipeError, EOFError):
                            pass
                        connection.close()
                    if process.pid is not None:
                        deadline = time.monotonic() + 15
                        while process.is_alive() and time.monotonic() < deadline:
                            await asyncio.sleep(0.05)
                        if process.is_alive():
                            process.kill()
                            completed = False
                        process.join()
                        exitcode = process.exitcode
                        process.close()
                        if exitcode != 0:
                            from multiprocessing.shared_memory import SharedMemory

                            try:
                                orphan = SharedMemory(name=Path(ipc).name.replace("-", "_"), create=False)
                            except FileNotFoundError:
                                pass
                            else:
                                orphan.close()
                                orphan.unlink()
                        if completed and exitcode != 0:
                            completed = False
                            raise RuntimeError(f"source process exited with status {exitcode}")
                    recording.emit("run_finished", status="completed" if completed else "failed")
        finally:
            events = []
            for name in ("source-events.jsonl", "worker-events.jsonl"):
                path = output / name
                if path.exists():
                    events.extend(json.loads(line) for line in path.read_text().splitlines())
            events.sort(key=lambda row: row["monotonic"])
            (output / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
