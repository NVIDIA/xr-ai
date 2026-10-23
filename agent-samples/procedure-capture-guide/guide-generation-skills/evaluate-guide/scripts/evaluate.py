# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a model-authored guide against its original capture packet.

Replay saved frames through the local XR-AI guide runtime, collect actual
verification results, and leave the draft unchanged for author revision.
This script does not generate YAML, approve guides, or manage model servers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

from _guide_runtime import EvaluationContext, run_evaluation
from _recorded_camera import capture_frames, feed_recording
from procedure_capture_guide_worker.guide._workflow_spec import parse_workflow
from procedure_capture_guide_worker.guide.guides import SelectedGuide


async def evaluate(packet: Path, guide_path: Path, output: Path, *, tail_s: float = 5) -> bool:
    if not math.isfinite(tail_s) or tail_s < 0:
        raise ValueError("tail_s must be finite and nonnegative")
    rows = capture_frames(packet)
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
                "tail_s": tail_s, "actions": actions, "fail_fast": True}
    output.mkdir(parents=True, exist_ok=False)

    async def exercise(run: EvaluationContext) -> bool:
        speech_offset = 0
        for index, action in enumerate(actions):
            if action["action"] == "say":
                speech_offset = len(run.speech.outputs)
                answer = await run.say(action["text"])
                calls = answer.calls
                passed = [call["name"] for call in calls] == action["tools"]
                if "arguments" in action:
                    passed &= [json.loads(call["arguments"]) for call in calls] == action["arguments"]
                if "expect" in action:
                    passed &= re.search(action["expect"], answer.text) is not None
                await run.guidance()
            else:
                await run.wait_until(
                    lambda: any(re.search(action["expect"], item.text)
                                for item in run.speech.outputs[speech_offset:]),
                    action["timeout_s"],
                )
                context = await run.guidance()
                step = workflow.steps[action["step"]]
                passed = not context["active"] if step.finish_on_complete else step.is_complete(context["state"])
            run.log("check", index=index, action=action["action"], passed=passed)
            if not passed:
                return False
        return not (await run.guidance())["active"]

    async def feed(camera, start, log):
        await feed_recording(camera, rows, start, log)

    return await run_evaluation(
        guide, raw, manifest, output, exercise, feed,
        sources=(Path(__file__), Path(__file__).with_name("_recorded_camera.py")),
        frame_semantics="Recorded frames once at original intervals; no refreshing the final frame",
        draft_test_only=True,
        duration_s=(rows[-1]["timestamp_us"] - rows[0]["timestamp_us"]) / 1e6 + tail_s,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True, help="Draft or approved YAML to test; never modified")
    parser.add_argument("--tail-s", type=float, default=5, help="Playback deadline extension in real seconds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.tail_s) or args.tail_s < 0:
        parser.error("--tail-s must be finite and nonnegative")
    success = asyncio.run(evaluate(args.packet, args.guide, args.output, tail_s=args.tail_s))
    raise SystemExit(0 if success else 1)
