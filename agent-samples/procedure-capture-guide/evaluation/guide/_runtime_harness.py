# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide a shared evaluation harness around the production guide agents.

Provide camera and speech boundaries plus trace collection shared by recorded
packet evaluation and example scenarios. Callers own their inputs and checks;
this module never generates guides or manages external model servers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import nemo_relay
import xr_ai_models
from loguru import logger
from PIL import Image
from procedure_capture_guide_worker.config import load_config
from procedure_capture_guide_worker.guide.agent import GuideAgent
from procedure_capture_guide_worker.guide.events import PARTICIPANT_JOINED_TOPIC, USER_QUERY_TOPIC
from procedure_capture_guide_worker.guide.guides import SelectedGuide
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
from xr_ai_interaction import ParticipantJoined, ResponseOutput, UserQuery
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC

_SAMPLE = Path(__file__).resolve().parents[2]
_PID = "guide-eval"


def _runtime_hashes(models_config: Path, sources: tuple[Path, ...] = ()) -> dict[str, str]:
    """Identify sample sources and loaded SDK code without scanning environments."""
    paths = {models_config.resolve(), _SAMPLE / "yaml/worker.yaml",
             _SAMPLE / "yaml/guide-agent-config.yaml", Path(__file__).resolve()}
    paths.update(path.resolve() for path in sources)
    package = _SAMPLE / "worker/procedure_capture_guide_worker"
    for pattern in ("*.py", "*.txt"):
        paths.update(package.rglob(pattern))
    sdk_packages = {"xr_ai_runtime", "xr_ai_hub", "xr_ai_models", "xr_ai_tools", "xr_ai_voice", "xr_ai_interaction"}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if name.split(".", 1)[0] in sdk_packages and source and Path(source).suffix == ".py":
            paths.add(Path(source).resolve())
    return {os.path.relpath(path, _SAMPLE): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


class Camera:
    """Supply decoded fixture frames at the hub endpoint boundary, not VLM answers."""

    connected_participants = {_PID}

    def __init__(self):
        self.listeners = []
        self._image_capture_listeners = []
        self.frame = None
        self.path = None
        self.sequence = 0
        self.sources = {}

    def on_frame(self, callback):
        self.listeners.append(callback)

    def on_participant(self, _callback):
        pass

    def _on_image_capture(self, callback):
        self._image_capture_listeners.append(callback)

        def unsubscribe():
            if callback in self._image_capture_listeners:
                self._image_capture_listeners.remove(callback)

        return unsubscribe

    async def _request_image_capture(self, request):
        # Playback cannot take a new still when source frames run out or expire.
        # Use the client's rejection protocol rather than refreshing old pixels.
        rejected = SimpleNamespace(
            participant_id=request.participant_id, request_id=request.request_id,
            pts_us=time.time_ns() // 1000,
            mime_type="application/vnd.xr-ai.capture-rejection+json",
            data=b'{"version":1,"status":"rejected"}',
        )
        for callback in tuple(self._image_capture_listeners):
            await callback(rejected)

    async def _cancel_image_capture(self, _request):
        # Rejections are immediate; there is no client-side capture to cancel.
        pass

    async def request_frame(self, _signal):
        return self.frame

    async def publish(self, *, timestamp_us=None):
        with Image.open(self.path) as source:
            image = source.convert("RGB")
            image.thumbnail((1280, 1280))
        self.sequence += 1
        self.sources[self.sequence] = str(self.path)
        pixels = image.tobytes()
        timestamp = timestamp_us if timestamp_us is not None else time.time_ns() // 1000
        self.frame = FrameData(self.sequence, timestamp, image.width, image.height,
                               PixelFormat.RGB24, pixels, _PID, "fixture-camera")
        signal = FrameSignal(0, self.sequence, timestamp, image.width, image.height,
                             PixelFormat.RGB24, len(pixels), _PID, "fixture-camera")
        for callback in self.listeners:
            await callback(signal)


class Speech(Agent):
    def __init__(self, log):
        super().__init__()
        self.outputs = []
        self.log = log

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def receive(self, output: ResponseOutput, _ctx: RuntimeContext):
        # Reply terminators and withdraws carry no speech.
        if not output.text:
            return
        self.outputs.append(output)
        self.log("speech", text=output.text, timestamp_us=output.timestamp_us)


@dataclass
class TurnResult:
    """Delivered answer and model tool requests belonging to one user turn."""

    text: str
    turn_id: int
    calls: tuple[dict, ...]


@dataclass
class EvaluationContext:
    """Expose observed runtime behavior to a packet or scenario evaluation."""

    camera: Camera
    speech: Speech
    events: list[dict]
    visual_results: list[dict]
    log: Callable[..., None]
    wait_until: Callable[..., Awaitable[None]]
    say: Callable[[str], Awaitable[TurnResult]]
    quiet: Callable[[float], Awaitable[bool]]
    guidance: Callable[[], Awaitable[dict]]


async def run_evaluation(
    guide: SelectedGuide, raw: bytes, manifest: dict, output: Path,
    exercise: Callable[[EvaluationContext], Awaitable[bool]],
    feed: Callable[..., Awaitable[None]], *, sources: tuple[Path, ...],
    frame_semantics: str, draft_test_only: bool = False, duration_s: float | None = None,
) -> bool:
    """Execute checks and preserve provenance without altering guide approval."""
    if hashlib.sha256(raw).hexdigest() != guide.sha256:
        raise ValueError("guide changed while loading; retry with a fixed guide")
    (output / "guide.yaml").write_bytes(raw)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()
    events = []

    def log(kind, **data):
        event = {"elapsed_s": round(time.monotonic() - started, 3), "kind": kind, **data}
        events.append(event)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    with tempfile.TemporaryDirectory(prefix="guide-eval-config-") as temporary:
        request = Path(temporary) / "request.json"
        request.write_text(json.dumps({"worker_config": str(_SAMPLE / "yaml/worker.yaml"),
                                       "mode": "guide", "guide_path": str(guide.path.resolve())}))
        settings = load_config(request)
    models = xr_ai_models.load_models_config(settings.models_config)
    llm, vlm = xr_ai_models.make_llm(models, "llm"), xr_ai_models.make_vlm(models, "vlm")
    decisions = xr_ai_models.make_decision(models, "decision")
    decide = decisions.decide
    model_origin: ContextVar[tuple[str, int | None, asyncio.Task | None]] = ContextVar(
        "model_origin", default=("unscoped", None, None),
    )

    def call_origin():
        origin, turn_id, owner = model_origin.get()
        # Child tasks inherit context, but not ownership of a foreground turn
        # or observation. Classify both model services at the execution boundary.
        return (origin, turn_id) if owner is asyncio.current_task() else ("unscoped", None)

    async def logged_decide(state, questions, **kwargs):
        origin, turn_id = call_origin()
        result = await decide(state, questions, **kwargs)
        log("decision", origin=origin, turn_id=turn_id, state=state, answers={
            name: {"choice": answer.choice, "confidence": answer.confidence}
            for name, answer in result.answers.items()
        })
        return result

    decisions.decide = logged_decide
    chat = llm.chat
    async def logged_chat(*args, **kwargs):
        origin, turn_id = call_origin()
        result = await chat(*args, **kwargs)
        log("model", origin=origin, turn_id=turn_id, text=result.content, calls=[
            {"name": call.name, "arguments": call.arguments} for call in result.tool_calls or ()
        ])
        return result

    llm.chat = logged_chat

    class TracedGuideAgent(GuideAgent):
        async def _answer(self, query, participant_id, request_id):
            token = model_origin.set(("foreground", query.timestamp_us, asyncio.current_task()))
            try:
                await super()._answer(query, participant_id, request_id)
            finally:
                model_origin.reset(token)

        async def _tool_loop(self, system, user, tools, *, foreground=False):
            # A monitor can inherit the context of the turn that started it.
            # Its observations must never be attributed to that foreground turn.
            origin = model_origin.get() if foreground else ("observation", None, asyncio.current_task())
            token = model_origin.set(origin)
            try:
                return await super()._tool_loop(system, user, tools, foreground=foreground)
            finally:
                model_origin.reset(token)

    camera, images = Camera(), ImageRegistry()
    image_sources = {}

    class Frames(CurrentFrameTool):
        async def execute(self, request):
            result = await super().execute(request)
            image_sources[result.image.uri] = camera.sources[result.sequence]
            log("selected_frame", image=camera.sources[result.sequence], sequence=result.sequence,
                timestamp_us=result.timestamp_us)
            return result

    frames = Frames(endpoint=camera, images=images, frame_timeout_s=settings.frame_timeout_s,
                    frame_max_age_s=settings.frame_max_age_s)
    visual_results = []

    class Vision(ImageQueryTool):
        async def execute(self, request):
            path = image_sources[request.image.uri]
            before = time.monotonic()
            result = await super().execute(request)
            item = {"image": path, "question": request.query, "answer": result.text,
                    "available": result.available, "latency_s": time.monotonic() - before}
            visual_results.append(item)
            log("vision", **item)
            return result

    vision = Vision(images=images, vlm=vlm, system_prompt=(
        _SAMPLE / "worker/procedure_capture_guide_worker/guide/prompts/guide_vision.txt"
    ).read_text())
    runtime = AgentRuntime()
    agent = runtime.register("guide", TracedGuideAgent(
        guide=guide, llm=llm, decisions=decisions, current_frame=frames, image_query=vision,
        vision_timeout_s=settings.guide.vision_timeout_s,
        decision_timeout_s=settings.guide.decision_timeout_s,
    ))
    tick = agent._backend._tick

    async def traced_tick(session):
        token = model_origin.set(("observation", None, asyncio.current_task()))
        try:
            return await tick(session)
        finally:
            model_origin.reset(token)

    agent._backend._tick = traced_tick
    speech = runtime.register("speech", Speech(log))
    agent.bind_runtime(runtime)
    playback_start = asyncio.Event()
    feeder = asyncio.create_task(feed(camera, playback_start, log))
    deadline = None
    passed = True
    quiet_offset = 0

    async def wait_until(predicate, timeout=90):
        if deadline is not None:
            timeout = min(timeout, max(0, deadline - time.monotonic()))
        async with asyncio.timeout(timeout):
            while not predicate():
                if feeder.done():
                    feeder.result()
                await asyncio.sleep(0.02)

    async def say(text):
        nonlocal quiet_offset
        timestamp = time.time_ns() // 1000
        log("user", text=text, timestamp_us=timestamp, turn_id=timestamp)
        await runtime.publish(USER_QUERY_TOPIC, UserQuery(text=text, timestamp_us=timestamp),
                              participant_id=_PID)
        # Correlate the delivered answer, not an unrelated background notice.
        await wait_until(lambda: any(item.timestamp_us == timestamp for item in speech.outputs))
        answer_index = next(index for index, item in enumerate(speech.outputs) if item.timestamp_us == timestamp)
        # Include notices delivered during the following image switch, even before quiet begins.
        quiet_offset = answer_index + 1
        calls = tuple(call for event in events
                      if event["kind"] == "model" and event["origin"] == "foreground"
                      and event["turn_id"] == timestamp for call in event["calls"])
        return TurnResult(speech.outputs[answer_index].text, timestamp, calls)

    def relay_event(event):
        with (output / "relay.jsonl").open("a") as stream:
            stream.write(event.to_json() + "\n")

    async def quiet(duration_s):
        await asyncio.sleep(duration_s)
        return len(speech.outputs) == quiet_offset

    async def guidance():
        snapshot = await agent._backend.guidance(_PID)
        log("guidance_snapshot", context=snapshot.context)
        return snapshot.context

    context = EvaluationContext(camera, speech, events, visual_results, log, wait_until, say, quiet, guidance)
    subscription = f"guide-eval-{time.time_ns()}"
    nemo_relay.subscribers.register(subscription, relay_event)
    try:
        async with runtime:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id=_PID)
            await wait_until(lambda: bool(speech.outputs))
            if duration_s is not None:
                deadline = time.monotonic() + duration_s
            playback_start.set()
            passed = await exercise(context)
    except Exception as exc:  # noqa: BLE001 — Loguru traceback at the evaluation boundary.
        # isolation boundary: report a failed evaluation without losing its artifacts.
        logger.exception("Guide evaluation failed")
        passed = False
        log("error", error=repr(exc))
    finally:
        await agent.stop()
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister(subscription)
        await llm.close()
        await vlm.close()
        await decisions.close()
        images.clear()
        # Include SDK modules loaded lazily by model adapters and tools during execution.
        (output / "runtime.json").write_text(json.dumps(_runtime_hashes(settings.models_config, sources), indent=2))
        if hashlib.sha256(guide.path.read_bytes()).hexdigest() != guide.sha256:
            passed = False
            log("error", error="guide changed during evaluation; the saved snapshot was evaluated, not the new file")
        (output / "result.json").write_text(json.dumps({
            "passed": passed, "checks_passed": sum(event["kind"] == "check" and event["passed"] for event in events),
            "fail_fast": manifest.get("fail_fast", False),
            "guide_sha256": guide.sha256, "elapsed_s": time.monotonic() - started,
            "checks_planned": len(manifest["actions"]),
            "checks_not_scored": len(manifest["actions"]) - sum(event["kind"] == "check" for event in events),
            "camera": "fixture pixels through CurrentFrameTool; not a live camera",
            "frame_semantics": frame_semantics,
            "draft_test_only": draft_test_only,
            "input": "typed runtime messages; STT and audio playback not tested",
        }, indent=2))
    return passed
