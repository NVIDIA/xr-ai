# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise a guide through the replay runtime with local models and image fixtures.

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
import time
from pathlib import Path

from PIL import Image
from procedure_capture_replay_worker.replay.agent import ReplayAgent
from procedure_capture_replay_worker.replay.events import PARTICIPANT_JOINED_TOPIC, USER_QUERY_TOPIC
from procedure_capture_replay_worker.replay.guides import select_guide
from xr_ai_hub import FrameData, FrameSignal, PixelFormat
from xr_ai_models import load_models_config, make_llm, make_vlm
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import ImageQueryTool
from xr_ai_voice import VOICE_OUTPUT_TOPIC, UserQuery, VoiceAggregationAgent, VoiceOutput, VoiceParticipantJoined

_SAMPLE = Path(__file__).resolve().parents[1]
_PID = "replay-eval"


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
        self.outputs.append(output.text)
        self.log("speech", text=output.text)


async def evaluate(manifest_path: Path, output: Path, guide_name: str, wait_timer: bool) -> bool:
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    guide = select_guide(_SAMPLE / "guides", guide_name)
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
        _SAMPLE / "worker/procedure_capture_replay_worker/replay/prompts/replay_vision.txt"
    ).read_text())
    runtime = AgentRuntime()
    aggregation = runtime.register("aggregation", VoiceAggregationAgent(
        llm=llm, coalesce_window_s=0.01, minimum_playback_s=0, speech_rate_wpm=60000,
    ))
    agent = runtime.register("replay", ReplayAgent(
        guide=guide, llm=llm, current_frame=frames, image_query=vision,
        vision_timeout_s=60, aggregation=aggregation,
    ))
    speech = runtime.register("speech", Speech(log))
    agent.bind_runtime(runtime)
    feeder = asyncio.create_task(camera.feed())
    passed = True
    timer_completed = False

    async def wait_until(predicate, timeout=90):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.05)

    async def say(text):
        before = len(speech.outputs)
        log("user", text=text)
        await runtime.publish(USER_QUERY_TOPIC, UserQuery(text=text, timestamp_us=time.time_ns() // 1000),
                              participant_id=_PID)
        await wait_until(lambda: len(speech.outputs) > before)
        await wait_until(lambda: _PID not in agent._turns)

    try:
        async with runtime:
            await runtime.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), participant_id=_PID)
            await wait_until(lambda: bool(speech.outputs))
            session = agent._backend._sessions[_PID]
            assert not session.active and not agent._backend._monitors
            # Supply a frame before start so the real observer never needs a missing-camera fallback.
            camera.path = (manifest_path.parent / manifest[0]["cases"][0]["image"]).resolve()
            await wait_until(lambda: camera.frame is not None)
            await say("start replay")
            for stage in manifest:
                assert session.step_id == stage["step"]
                for case in stage["cases"]:
                    camera.path = (manifest_path.parent / case["image"]).resolve()
                    digest = hashlib.sha256(camera.path.read_bytes()).hexdigest()
                    offset = len(visual_results)
                    await wait_until(lambda: len([
                        item for item in visual_results[offset:] if item["image"] == str(camera.path)
                    ]) >= case.get("observations", 2) or agent._backend._is_complete(session))
                    await asyncio.sleep(0.1)  # Allow the completed tool result to reach its state commit.
                    answers = [item["answer"].strip() for item in visual_results[offset:]
                               if item["image"] == str(camera.path)]
                    complete = agent._backend._is_complete(session)
                    ok = bool(answers) and all(a in case["answers"] for a in answers)
                    state_ok = complete == case["complete"]
                    answers_ok = ok
                    ok = ok and state_ok
                    passed &= ok
                    log("case", step=stage["step"], name=case["name"], source=case.get("source"),
                        sha256=digest, answers=answers, complete=complete,
                        expected_complete=case["complete"], state_passed=state_ok,
                        answers_passed=answers_ok, passed=ok)
                    if complete and not case["complete"]:
                        raise AssertionError("negative visual case falsely completed the step")
                if not agent._backend._is_complete(session):
                    raise AssertionError(f"positive case did not complete {session.step_id}")
                await say("next")
            assert session.step.trigger.function == "clock__timer"
            assert session.timer is not None and not session.timer.read().expired
            timer_started = session.timer.started
            timer_duration = session.timer.duration_s
            await say("How much time remains?")
            await say("What happens when the steeping time is up?")
            await say("next")
            assert session.active and session.step.trigger.function == "clock__timer"
            log("timer_wait", duration_s=session.timer.duration_s, real_time=wait_timer)
            if wait_timer:
                await wait_until(lambda: not session.active, timeout=session.timer.duration_s + 30)
                assert time.monotonic() - timer_started >= timer_duration
                await wait_until(lambda: any(guide.workflow.complete_message in text for text in speech.outputs))
                timer_completed = True
                log("automatic_finish", active=session.active, state=session.state,
                    timer_elapsed_s=time.monotonic() - timer_started)
                await say("start replay")
                assert session.active and session.step_id == guide.workflow.start_step
            await say("stop replay")
            assert not session.active and not agent._backend._monitors
    except Exception as exc:
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
            "passed": passed, "timer_completed_in_real_time": timer_completed,
            "guide_sha256": guide.sha256, "elapsed_s": time.monotonic() - started,
            "camera": "fixture pixels through CurrentFrameTool; not a live camera",
            "input": "typed runtime messages; STT and audio playback not tested",
        }, indent=2))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guide", default="Tea Making")
    parser.add_argument("--wait-timer", action="store_true", help="Wait the real configured timer duration.")
    args = parser.parse_args()
    success = asyncio.run(evaluate(args.manifest.resolve(), args.output, args.guide, args.wait_timer))
    raise SystemExit(0 if success else 1)
