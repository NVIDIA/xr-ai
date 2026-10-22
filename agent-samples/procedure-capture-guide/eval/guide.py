# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a draft guide against recorded frames or controlled image fixtures.

The camera endpoint supplies fixture pixels instead of a physical camera; frame
selection, visual inference, conversation routing, monitoring, and timers use
the sample's production components. Typed utterances enter the same runtime
topic as transcribed speech. This does not test microphones, STT, or audio playback.
Externally managed model servers are never started or stopped by this script.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import nemo_relay
import yaml
from _recorded_camera import capture_frames, feed_recording
from loguru import logger
from PIL import Image, ImageDraw
from procedure_capture_guide_worker.config import load_config
from procedure_capture_guide_worker.guide._workflow_spec import parse_workflow
from procedure_capture_guide_worker.guide.agent import GuideAgent
from procedure_capture_guide_worker.guide.events import PARTICIPANT_JOINED_TOPIC, USER_QUERY_TOPIC
from procedure_capture_guide_worker.guide.guides import SelectedGuide, load_guide
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
from xr_ai_interaction import ParticipantJoined, ResponseOutput, UserQuery
from xr_ai_models import load_models_config, make_llm, make_vlm
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceAggregationAgent

_SAMPLE = Path(__file__).resolve().parents[1]
_PID = "guide-eval"


def _runtime_hashes(models_config: Path) -> dict[str, str]:
    """Identify sample sources and loaded SDK code without scanning environments."""
    paths = {models_config.resolve(), _SAMPLE / "yaml/worker.yaml",
             _SAMPLE / "yaml/guide-agent-config.yaml", Path(__file__).resolve(),
             Path(__file__).with_name("_recorded_camera.py").resolve()}
    package = _SAMPLE / "worker/procedure_capture_guide_worker"
    for pattern in ("*.py", "*.txt"):
        paths.update(package.rglob(pattern))
    sdk_packages = {"xr_ai_runtime", "xr_ai_hub", "xr_ai_models", "xr_ai_tools", "xr_ai_voice"}
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
        self.frame = None
        self.path = None
        self.sequence = 0
        self.sources = {}

    def on_frame(self, callback):
        self.listeners.append(callback)

    def on_participant(self, _callback):
        pass

    def _on_image_capture(self, _callback):
        return lambda: None

    async def request_frame(self, _signal):
        return self.frame

    async def feed(self):
        while True:
            if self.path is not None:
                await self.publish()
            await asyncio.sleep(0.25)

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
        self.outputs.append(output)
        self.log("speech", text=output.text, timestamp_us=output.timestamp_us)


async def evaluate(manifest_path: Path | None, output: Path, *, packet: Path | None = None,
                   guide_path: Path | None = None, tail_s: float = 5) -> bool:
    output.mkdir(parents=True, exist_ok=False)
    rows = capture_frames(packet) if packet else None
    if packet:
        raw = guide_path.read_bytes()
        workflow = parse_workflow(raw)
        # Test eligibility never approves or rewrites the author's draft on disk.
        guide = SelectedGuide(guide_path, hashlib.sha256(raw).hexdigest(), replace(workflow, status="approved"))
        actions = [{"action": "say", "text": "Start guide", "tools": ["workflow__restart"]}]
        step_id = workflow.start_step
        while step_id is not None:
            step = workflow.steps[step_id]
            message = workflow.complete_message if step.finish_on_complete else step.complete_message
            actions.append({"action": "wait_speech", "expect": re.escape(message), "step": step.id,
                            "timeout_s": (rows[-1]["timestamp_us"] - rows[0]["timestamp_us"]) / 1e6 + tail_s})
            if not step.finish_on_complete:
                action = {"action": "say", "text": "Next", "tools": ["workflow__advance"],
                          "arguments": [{"skip": False}]}
                if step.next_step is None:
                    action["expect"] = re.escape(workflow.complete_message)
                actions.append(action)
            step_id = step.next_step
        manifest = {"packet": str(packet.resolve()), "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                    "frames": rows, "source_status": workflow.status, "draft_test_only": True,
                    "tail_s": tail_s, "actions": actions}
    else:
        manifest = yaml.safe_load(manifest_path.read_text())
        guide = load_guide(manifest_path.parent / manifest["guide_path"])
        raw = guide.path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != guide.sha256:
        raise ValueError("guide changed while loading; retry with a fixed guide")
    fixtures = {}
    for name, scene in ({} if packet else manifest.get("scenes", {})).items():
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        if "rectangle" in scene:
            draw.rectangle((120, 100, 520, 380), fill=scene["rectangle"])
        if "text" in scene:
            draw.multiline_text((40, 40), scene["text"], fill="black", font_size=42, spacing=20)
        path = output / f"{name}.png"
        image.save(path)
        fixtures[name] = path.resolve()
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
    models = load_models_config(settings.models_config)
    llm, vlm = make_llm(models, "llm"), make_vlm(models, "vlm")
    chat = llm.chat

    async def logged_chat(*args, **kwargs):
        result = await chat(*args, **kwargs)
        log("model", text=result.content, calls=[
            {"name": call.name, "arguments": call.arguments} for call in result.tool_calls or ()
        ])
        return result

    llm.chat = logged_chat
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
    aggregation = runtime.register("aggregation", VoiceAggregationAgent(
        llm=llm, coalesce_window_s=0.01, minimum_playback_s=0, speech_rate_wpm=60000,
    ))
    agent = runtime.register("guide", GuideAgent(
        guide=guide, llm=llm, current_frame=frames, image_query=vision,
        vision_timeout_s=settings.guide.vision_timeout_s, aggregation=aggregation,
    ))
    speech = runtime.register("speech", Speech(log))
    agent.bind_runtime(runtime)
    playback_start = asyncio.Event()
    feeder = asyncio.create_task(feed_recording(camera, rows, playback_start, log) if packet else camera.feed())
    deadline = None
    passed = True
    speech_offset = 0

    async def wait_until(predicate, timeout=90):
        if deadline is not None:
            timeout = min(timeout, max(0, deadline - time.monotonic()))
        async with asyncio.timeout(timeout):
            while not predicate():
                if feeder.done():
                    feeder.result()
                await asyncio.sleep(0.02)

    async def say(text):
        timestamp = time.time_ns() // 1000
        log("user", text=text, timestamp_us=timestamp)
        await runtime.publish(USER_QUERY_TOPIC, UserQuery(text=text, timestamp_us=timestamp),
                              participant_id=_PID)
        # Correlate the delivered answer, not an unrelated background notice.
        await wait_until(lambda: any(item.timestamp_us == timestamp for item in speech.outputs))
        return next(item.text for item in speech.outputs if item.timestamp_us == timestamp)

    def relay_event(event):
        with (output / "relay.jsonl").open("a") as stream:
            stream.write(event.to_json() + "\n")

    subscription = f"guide-eval-{time.time_ns()}"
    nemo_relay.subscribers.register(subscription, relay_event)
    try:
        async with runtime:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, ParticipantJoined(), participant_id=_PID)
            await wait_until(lambda: bool(speech.outputs))
            if packet:
                deadline = time.monotonic() + (rows[-1]["timestamp_us"] - rows[0]["timestamp_us"]) / 1e6 + tail_s
                playback_start.set()
            for index, action in enumerate(manifest["actions"]):
                kind = action["action"]
                offset = len(events)
                action_passed = True
                if kind == "image":
                    camera.path = fixtures[action["scene"]] if "scene" in action else (
                        manifest_path.parent / action["path"]
                    ).resolve()
                    sequence = camera.sequence
                    await wait_until(lambda: camera.sequence > sequence
                                     and camera.sources[camera.sequence] == str(camera.path))
                    log("image", path=str(camera.path), sha256=hashlib.sha256(camera.path.read_bytes()).hexdigest())
                elif kind == "observe":
                    before = len(visual_results)
                    await wait_until(lambda: len([item for item in visual_results[before:]
                                                 if item["image"] == str(camera.path)]) >= action.get("count", 2))
                    answers = [item["answer"].strip() for item in visual_results[before:]
                               if item["image"] == str(camera.path)]
                    action_passed = all(answer in action["answers"] for answer in answers)
                elif kind == "say":
                    speech_offset = len(speech.outputs)
                    answer = await say(action["text"])
                    if "expect" in action:
                        action_passed &= re.search(action["expect"], answer) is not None
                    if "tools" in action:
                        calls = [call for event in events[offset:] if event["kind"] == "model"
                                 for call in event["calls"]]
                        action_passed &= [call["name"] for call in calls] == action["tools"]
                        if "arguments" in action:
                            action_passed &= [json.loads(call["arguments"]) for call in calls] == action["arguments"]
                elif kind == "wait_speech":
                    await wait_until(lambda: any(re.search(action["expect"], item.text)
                                                 for item in speech.outputs[speech_offset:]),
                                     action.get("timeout_s", 90))
                else:
                    raise ValueError(f"unknown action: {kind}")
                if packet:
                    snapshot = await agent._backend.guidance(_PID)
                    log("guidance_snapshot", context=snapshot.context)
                    if kind == "wait_speech":
                        step = guide.workflow.steps[action["step"]]
                        action_passed &= (not snapshot.context["active"] if step.finish_on_complete else
                                          step.is_complete(snapshot.context["state"]))
                passed &= action_passed
                log("check", index=index, action=kind, passed=action_passed)
                if packet and not action_passed:
                    # Later turns depend on this state; do not score an imagined successful history.
                    break
            if packet:
                snapshot = await agent._backend.guidance(_PID)
                passed &= not snapshot.context["active"]
    except Exception as exc:  # noqa: BLE001 — Loguru traceback at the evaluation boundary.
        # isolation boundary: report a failed evaluation without losing its artifacts.
        logger.exception("Guide evaluation failed")
        passed = False
        log("error", error=repr(exc))
    finally:
        await agent.stop()
        await aggregation.stop()
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister(subscription)
        await llm.close()
        await vlm.close()
        images.clear()
        # Include SDK modules loaded lazily by model adapters and tools during execution.
        (output / "runtime.json").write_text(json.dumps(_runtime_hashes(settings.models_config), indent=2))
        if hashlib.sha256(guide.path.read_bytes()).hexdigest() != guide.sha256:
            passed = False
            log("error", error="guide changed during evaluation; the saved snapshot was evaluated, not the new file")
        (output / "result.json").write_text(json.dumps({
            "passed": passed, "checks_passed": sum(event["kind"] == "check" and event["passed"] for event in events),
            "guide_sha256": guide.sha256, "elapsed_s": time.monotonic() - started,
            "checks_planned": len(manifest["actions"]),
            "checks_not_scored": len(manifest["actions"]) - sum(event["kind"] == "check" for event in events),
            "camera": "fixture pixels through CurrentFrameTool; not a live camera",
            "frame_semantics": (
                "Recorded frames once at original intervals; no refreshing the final frame" if packet
                else "Static synthetic scene republished as live frames; not independent recorded views"
            ),
            "draft_test_only": bool(packet),
            "input": "typed runtime messages; STT and audio playback not tested",
        }, indent=2))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--packet", type=Path)
    parser.add_argument("--guide", type=Path, help="Draft or approved YAML to test with --packet; never modified")
    parser.add_argument("--tail-s", type=float, default=5, help="Packet playback deadline extension in real seconds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.packet) != bool(args.guide):
        parser.error("--packet and --guide must be supplied together")
    if not math.isfinite(args.tail_s) or args.tail_s < 0:
        parser.error("--tail-s must be finite and nonnegative")
    manifest = args.manifest or (None if args.packet else Path(__file__).with_name("guide_cases.yaml"))
    success = asyncio.run(evaluate(manifest.resolve() if manifest else None, args.output,
                                   packet=args.packet, guide_path=args.guide, tail_s=args.tail_s))
    raise SystemExit(0 if success else 1)
