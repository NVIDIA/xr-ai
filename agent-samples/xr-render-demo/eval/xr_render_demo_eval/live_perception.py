# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live perception-routing tests: real conversation history, reply-text verdict.

The live eval stack publishes no camera track, so a correctly routed
perception turn must come back as an honest inability report from
vision_agent. That invariant catches the misses the offline tiers cannot
reproduce (their [Recent conversation] is far shorter than a live one):
a perception question answered from the transcript, or a describe-the-view
request answered by reciting SCENE OBJECTS. Same-session turns inside the
supervisor's recall window are out of scope here; diagnose those from the
worker log and the transcript store.

Run subsets: `live_perception.py [case-name ...]`.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from xr_ai_hub import DataMessage
from xr_ai_tools.rpc import RPCClient
from xr_ai_tools.text_memory import AddTranscriptRequest, RecallConversationRequest, TextMemoryTools
from xr_render_scene import AddPrimitiveRequest, EmptyRequest, RemovePrimitiveRequest, SceneClient

from ._live_endpoint import LiveEvalEndpoint, live_participant

CANONICAL = {"position": {"x": 0, "y": 1.6, "z": 0}, "forward": {"x": 0, "y": 0, "z": -1},
             "right": {"x": 1, "y": 0, "z": 0}, "up": {"x": 0, "y": 1, "z": 0},
             "yaw_deg": 0.0, "pitch_deg": 0.0, "ts": 1}

def _text_memory_dir() -> str:
    """The driver reads the same transcript store the worker writes."""
    yaml_path = Path(__file__).resolve().parents[2] / "yaml" / "xr_render_demo_worker.yaml"
    try:
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("text_memory_dir:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "/dev/shm/xr-ai/text-memory"


# Bare "camera"/"vision" are excluded: a confabulated answer may mention the
# camera while still claiming to see.
INABILITY_MARKERS = (
    "can't see", "cannot see", "couldn't see", "can not see",
    "unable", "not able", "no camera", "no view",
    "unavailable", "not available", "can't access", "cannot access",
    "don't have", "do not have", "can't tell", "cannot tell",
    "can't check", "cannot check", "couldn't check",
)

# Words that appear in a reply only when the model recites the XR scene.
SCENE_RECITAL_WORDS = ("cube", "box", "sphere", "ball", "cone", "capsule", "cylinder", "pyramid")

_WORD_TYPES = {"cube": "box", "ball": "sphere", "donut": "torus"}

# Each case: history prompts that must change the scene (real turns that
# fill [Recent conversation]), then one perception prompt judged on the
# reply text and on the scene staying unchanged.
CASES = [
    {
        "name": "holding_mid_conversation",
        "history": ["Make a green cube.", "Move it to the left."],
        "prompt": "What is the thing in my hand?",
    },
    {
        "name": "describe_view_not_scene",
        "history": ["Make a green cube.", "Add a blue ball next to it."],
        "prompt": "Tell me what everything around me looks like.",
    },
    {
        "name": "camera_imperative_pending",
        "history": ["Make a green cube."],
        "pending": "Make a ball the color of the item in my hand.",
        "prompt": "Go ahead and look.",
    },
    {
        # The transcript store persists across sessions, so [Recent
        # conversation] can carry old turns where perception questions were
        # answered without the camera. Those replies are few-shot poison:
        # the supervisor must still delegate vision, never imitate them.
        "name": "holding_after_poisoned_history",
        "fixtures": [("box", 0.0, 1.6, -1.5, 1, 0, 0, 0.1)],
        "poisoned": [
            ("user", "What do you see?"),
            ("agent", "I see a red box located at position x=0.0, y=1.6, z=-1.5."),
            ("user", "Look at the camera and describe what you see."),
            ("agent", "The camera shows a red cube (id: box-0) positioned at "
                      "coordinates (0.0, 1.6, -1.5) in the scene."),
            ("user", "What is in my hand?"),
            ("agent", "You are not holding anything."),
        ],
        "prompt": "What am I holding?",
    },
]


async def _inject_poisoned(memory, participant, turns):
    # Two hours old: unambiguously cross-session, the condition this eval
    # reproduces.
    base_us = time.time_ns() // 1_000 - 7_200_000_000
    for offset, (role, text) in enumerate(turns):
        await memory.add_transcript.execute(AddTranscriptRequest(
            source_id=f"{participant}:{role}",
            timestamp_us=base_us + offset * 5_000_000,
            text=text,
        ))


async def clear_scene(scene):
    state = await scene.get_scene_state(EmptyRequest())
    for item in state.objects:
        await scene.remove_primitive(RemovePrimitiveRequest(obj_id=item.id))


async def _agent_replies_since(memory, participant, after_us):
    recalled = await memory.recall_conversation.execute(
        RecallConversationRequest(participant_id=participant))
    return [
        entry.text
        for entry in recalled.entries
        if entry.role == "agent" and entry.timestamp_us >= after_us
    ]


async def _send_and_wait_reply(endpoint, memory, participant, prompt, timeout_s=75):
    # Correlating by timestamp keeps a straggler reply to the previous
    # prompt from being judged as this one's.
    sent_us = time.time_ns() // 1_000
    await endpoint.inject_data(DataMessage(
        participant_id=participant, topic="", pts_us=sent_us, data=prompt.encode()))
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(2)
        replies = await _agent_replies_since(memory, participant, sent_us)
        if replies:
            return replies[0]
    return ""


def _judge(reply, scene_unchanged, asked_words=()):
    lowered = reply.lower().replace("’", "'")
    if not reply:
        return False, "no reply within timeout"
    if not scene_unchanged:
        return False, "scene changed on a perception turn"
    # A shape word the case's own utterances used is the request echoed,
    # not the scene recited.
    recited = [w for w in SCENE_RECITAL_WORDS if w in lowered and w not in asked_words]
    if recited:
        return False, f"reply recites scene objects {recited}: {reply!r}"
    if not any(marker in lowered for marker in INABILITY_MARKERS):
        return False, f"reply claims an answer with no camera: {reply!r}"
    return True, reply


async def main() -> None:
    scene = SceneClient("tcp://127.0.0.1:8320")
    memory = TextMemoryTools(_text_memory_dir())
    tracking = RPCClient("tcp://127.0.0.1:8330", timeout_s=10.0)
    endpoint = LiveEvalEndpoint()
    await asyncio.sleep(0.5)
    try:
        await tracking.call("set_sim_pose", CANONICAL)
    except Exception as error:
        print(f"openxr service refused set_sim_pose ({error}); set allow_sim_pose: true in "
              "../yaml/openxr_service.yaml and restart the stack")
        await scene.close()
        await tracking.close()
        await endpoint.close()
        raise SystemExit(2) from None
    try:
        wanted = set(sys.argv[1:])
        passed = failed = 0
        for index, case in enumerate(CASES):
            if wanted and case["name"] not in wanted:
                continue
            participant = f"live-perception-{os.getpid()}-{int(time.time())}-{index}"
            async with live_participant(endpoint, participant):
                await clear_scene(scene)
                ok, detail = True, ""
                for prim_type, x, y, z, r, g, b, size in case.get("fixtures", ()):
                    await scene.add_primitive(AddPrimitiveRequest(
                        prim_type=prim_type, x=x, y=y, z=z, r=r, g=g, b=b, size=size))
                if "poisoned" in case:
                    await _inject_poisoned(memory, participant, case["poisoned"])
                for turn in case.get("history", ()):
                    before = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
                    await _send_and_wait_reply(endpoint, memory, participant, turn)
                    after = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
                    if before == after:
                        ok, detail = False, f"history turn made no scene change: {turn!r}"
                        break
                if ok and "pending" in case:
                    await _send_and_wait_reply(endpoint, memory, participant, case["pending"])
                if ok:
                    before = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
                    reply = await _send_and_wait_reply(endpoint, memory, participant, case["prompt"])
                    after = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
                    # Only the judged turn's own words are exempt, and only
                    # while no scene object of that type exists; history
                    # words must still count as recital.
                    asked = f"{case['prompt']} {case.get('pending', '')}".lower()
                    scene_types = {item.type for item in before.values()}
                    asked_words = tuple(
                        w for w in SCENE_RECITAL_WORDS
                        if w in asked and _WORD_TYPES.get(w, w) not in scene_types
                    )
                    ok, detail = _judge(reply, before == after, asked_words)
            print(f"{'PASS' if ok else 'FAIL'} {case['name']:28s} {detail[:220]}", flush=True)
            passed += ok
            failed += not ok
        print(f"\nlive perception: {passed} passed, {failed} failed", flush=True)
    finally:
        try:
            await tracking.call("clear_sim_pose", {})
        except Exception:
            # Best-effort cleanup; the stack may already be gone.
            pass
        await scene.close()
        await tracking.close()
        await endpoint.close()
    sys.exit(1 if failed else 0)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
