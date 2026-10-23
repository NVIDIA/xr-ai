# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the tea-making guide with generated scenes and scripted user turns.

Use the real guide agent and local models to test recognition, conversation,
progress, and elapsed timers. Scene and assertion files live beside this runner;
recorded-packet authoring evaluation belongs to the evaluate-guide skill.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml
from _scene_rendering import load_scenes, render_scene
from _visual_expectations import matches_answer, visual_cases
from procedure_capture_guide_worker.guide.guides import load_guide

_SAMPLE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SAMPLE / "eval"))
import guide_runtime as runtime  # noqa: E402


def validate_actions(manifest: dict, cases: dict) -> None:
    """Reject malformed checks before rendering fixtures or contacting models."""
    actions = manifest.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a nonempty list")
    quiet_allowed = False
    for index, action in enumerate(actions):
        label = f"action {index}"
        if not isinstance(action, dict):
            raise ValueError(f"{label}: expected an action mapping")
        kind = action.get("action")
        if kind not in {"image", "say", "observe", "wait_speech", "quiet"}:
            raise ValueError(f"{label}: unknown action {kind!r}")
        for field in ("expect", "after_speech"):
            if field in action:
                if not isinstance(action[field], str) or not action[field].strip():
                    raise ValueError(f"{label}: {field} must be a nonempty regex string")
                try:
                    re.compile(action[field])
                except re.error as exc:
                    raise ValueError(f"{label}: invalid {field} regex") from exc
        if kind == "wait_speech" and "expect" not in action:
            raise ValueError(f"{label}: wait_speech requires expect")
        if ("after_speech" in action) != ("minimum_elapsed_s" in action):
            raise ValueError(f"{label}: after_speech and minimum_elapsed_s must appear together")
        if "after_speech" in action and kind != "wait_speech":
            raise ValueError(f"{label}: after_speech requires wait_speech")
        if kind == "quiet" and ("duration_s" not in action or not quiet_allowed):
            raise ValueError(f"{label}: quiet requires duration_s and a preceding say with only image actions between")
        for field in ("duration_s", "minimum_elapsed_s", "timeout_s"):
            if field not in action:
                continue
            value = action[field]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (field != "minimum_elapsed_s" and value == 0)):
                bound = "nonnegative" if field == "minimum_elapsed_s" else "positive"
                raise ValueError(f"{label}: invalid {field}; use a finite {bound} number")
        if kind == "observe":
            count = action.get("count", 2)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{label}: observe count must be a positive integer")
            if "case" in action:
                if not isinstance(action["case"], str) or action["case"] not in cases or "answers" in action:
                    raise ValueError(f"{label}: observe requires a defined case or answers, not both")
            elif (not isinstance(action.get("answers"), list) or not action["answers"]
                  or not all(isinstance(item, str) for item in action["answers"])):
                raise ValueError(f"{label}: observe requires answers or a defined case")
        # Quiet measures from the last answer, not from an intervening check.
        if kind != "image":
            quiet_allowed = kind == "say"


def announcement_elapsed(events: list[dict], origin: str, completion: str) -> float | None:
    """Measure first announcements within the latest restart, not later paraphrases."""
    start = 0
    for index, event in enumerate(events):
        if event["kind"] == "model" and any(call["name"] == "workflow__restart" for call in event["calls"]):
            start = index + 1
    speeches = [event for event in events[start:] if event["kind"] == "speech"]
    first = next((index for index, event in enumerate(speeches) if re.search(origin, event["text"])), None)
    if first is None:
        return None
    last = next((event for event in speeches[first + 1:] if re.search(completion, event["text"])), None)
    return None if last is None else last["elapsed_s"] - speeches[first]["elapsed_s"]


async def feed_scenes(camera, start, _log):
    await start.wait()
    while True:
        if camera.path is not None:
            await camera.publish()
        await asyncio.sleep(0.25)


async def evaluate(manifest_path: Path, output: Path) -> bool:
    manifest = yaml.safe_load(manifest_path.read_text())
    fail_fast = manifest.get("fail_fast", False)
    if not isinstance(fail_fast, bool):
        raise ValueError("fail_fast must be a boolean")
    cases = {}
    if manifest.get("visual_cases_file"):
        cases = visual_cases(yaml.safe_load((manifest_path.parent / manifest["visual_cases_file"]).read_text()))
    validate_actions(manifest, cases)
    guide = load_guide(manifest_path.parent / manifest["guide_path"])
    raw = guide.path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != guide.sha256:
        raise ValueError("guide changed while loading; retry with a fixed guide")
    output.mkdir(parents=True, exist_ok=False)
    fixtures = {}
    for name, scene in load_scenes(manifest_path, manifest).items():
        path = output / f"{name}.png"
        render_scene(scene).save(path)
        fixtures[name] = path.resolve()
    if cases:
        (output / "visual_cases.json").write_text(json.dumps(cases, indent=2))

    async def exercise(run: runtime.EvaluationContext) -> bool:
        camera, speech = run.camera, run.speech
        events, visual_results = run.events, run.visual_results
        say, wait_until, log = run.say, run.wait_until, run.log
        speech_offset = 0
        passed = True
        for index, action in enumerate(manifest["actions"]):
            kind = action["action"]
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
                results = [item for item in visual_results[before:] if item["image"] == str(camera.path)]
                if "case" in action:
                    case = cases[action["case"]]
                    question = guide.workflow.steps[case["step"]].trigger.arguments["question"]
                    action_passed = str(camera.path) == str(fixtures[case["scene"]]) and all(
                        item["available"] and item["question"] == question
                        and matches_answer(case["answer"], item["answer"]) for item in results
                    )
                else:
                    action_passed = all(item["available"] and any(
                        matches_answer(re.escape(" ".join(answer.split())), item["answer"])
                        for answer in action["answers"]
                    ) for item in results)
            elif kind == "say":
                speech_offset = len(speech.outputs)
                answer = await say(action["text"])
                if "expect" in action:
                    action_passed &= re.search(action["expect"], answer.text) is not None
                if "tools" in action:
                    calls = answer.calls
                    action_passed &= [call["name"] for call in calls] == action["tools"]
                    if "arguments" in action:
                        action_passed &= [json.loads(call["arguments"]) for call in calls] == action["arguments"]
            elif kind == "wait_speech":
                await wait_until(lambda: any(re.search(action["expect"], item.text)
                                             for item in speech.outputs[speech_offset:]),
                                 action.get("timeout_s", 90))
                if "after_speech" in action:
                    elapsed = announcement_elapsed(events, action["after_speech"], action["expect"])
                    action_passed = elapsed is not None and elapsed >= action["minimum_elapsed_s"]
            elif kind == "quiet":
                action_passed = await run.quiet(action["duration_s"])
            else:
                raise ValueError(f"unknown action: {kind}")
            passed &= action_passed
            log("check", index=index, action=kind, passed=action_passed)
            if fail_fast and not action_passed:
                break
        return passed

    return await runtime.run_evaluation(
        guide, raw, manifest, output, exercise, feed_scenes,
        sources=(Path(__file__), Path(__file__).with_name("_scene_rendering.py"),
                 Path(__file__).with_name("_visual_expectations.py")),
        frame_semantics="Static synthetic scene republished as live frames; not independent recorded views",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.manifest.resolve(), args.output)) else 1)
