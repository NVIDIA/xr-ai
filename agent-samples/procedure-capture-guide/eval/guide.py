# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise a guide through the guide runtime with local models and image fixtures.

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
import re
import time
from pathlib import Path

import yaml
from loguru import logger
from PIL import Image, ImageDraw
from procedure_capture_guide_worker.guide.agent import GuideAgent
from procedure_capture_guide_worker.guide.events import PARTICIPANT_JOINED_TOPIC, USER_QUERY_TOPIC
from procedure_capture_guide_worker.guide.guides import select_guide
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
from xr_ai_models import load_models_config, make_llm, make_vlm
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC, UserQuery, VoiceAggregationAgent, VoiceOutput, VoiceParticipantJoined

_SAMPLE = Path(__file__).resolve().parents[1]
_PID = "guide-eval"


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
                with Image.open(self.path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((1280, 1280))
                self.sequence += 1
                self.sources[self.sequence] = str(self.path)
                pixels = image.tobytes()
                timestamp = time.time_ns() // 1000
                self.frame = FrameData(self.sequence, timestamp, image.width, image.height,
                                       PixelFormat.RGB24, pixels, _PID, "fixture-camera")
                signal = FrameSignal(0, self.sequence, timestamp, image.width, image.height,
                                     PixelFormat.RGB24, len(pixels), _PID, "fixture-camera")
                for callback in self.listeners:
                    await callback(signal)
            await asyncio.sleep(0.25)


class Speech(Agent):
    def __init__(self, log):
        super().__init__()
        self.outputs = []
        self.log = log

    @subscribe(VOICE_OUTPUT_TOPIC)
    async def receive(self, output: VoiceOutput, _ctx: RuntimeContext):
        self.outputs.append(output)
        self.log("speech", text=output.text, timestamp_us=output.timestamp_us)


async def evaluate(manifest_path: Path, output: Path) -> bool:
    output.mkdir(parents=True, exist_ok=False)
    manifest = yaml.safe_load(manifest_path.read_text())
    guide = select_guide((manifest_path.parent / manifest["guides_dir"]).resolve(), manifest["guide"])
    fixtures = {}
    for name, scene in manifest.get("scenes", {}).items():
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        if "rectangle" in scene:
            draw.rectangle((120, 100, 520, 380), fill=scene["rectangle"])
        if "text" in scene:
            draw.multiline_text((40, 40), scene["text"], fill="black", font_size=42, spacing=20)
        path = output / f"{name}.png"
        image.save(path)
        fixtures[name] = path.resolve()
    (output / "guide.yaml").write_bytes(guide.path.read_bytes())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()
    events = []

    def log(kind, **data):
        event = {"elapsed_s": round(time.monotonic() - started, 3), "kind": kind, **data}
        events.append(event)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    models = load_models_config(_SAMPLE / "yaml/models.json")
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
            return result

    frames = Frames(endpoint=camera, images=images, frame_timeout_s=1)
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
        vision_timeout_s=60, aggregation=aggregation,
    ))
    speech = runtime.register("speech", Speech(log))
    agent.bind_runtime(runtime)
    feeder = asyncio.create_task(camera.feed())
    passed = True
    speech_offset = 0

    async def wait_until(predicate, timeout=90):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.02)

    async def say(text):
        timestamp = time.time_ns() // 1000
        log("user", text=text, timestamp_us=timestamp)
        await runtime.publish(USER_QUERY_TOPIC, UserQuery(text=text, timestamp_us=timestamp),
                              participant_id=_PID)
        # Correlate the delivered answer, not an unrelated background notice.
        await wait_until(lambda: any(item.timestamp_us == timestamp for item in speech.outputs))
        return next(item.text for item in speech.outputs if item.timestamp_us == timestamp)

    try:
        async with runtime:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), participant_id=_PID)
            await wait_until(lambda: bool(speech.outputs))
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
                    assert all(answer in action["answers"] for answer in answers), answers
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
                passed &= action_passed
                log("check", index=index, action=kind, passed=action_passed)
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
        await llm.close()
        await vlm.close()
        images.clear()
        (output / "result.json").write_text(json.dumps({
            "passed": passed, "checks_passed": sum(event["kind"] == "check" and event["passed"] for event in events),
            "guide_sha256": guide.sha256, "elapsed_s": time.monotonic() - started,
            "camera": "fixture pixels through CurrentFrameTool; not a live camera",
            "input": "typed runtime messages; STT and audio playback not tested",
        }, indent=2))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("guide_cases.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    success = asyncio.run(evaluate(args.manifest.resolve(), args.output))
    raise SystemExit(0 if success else 1)
