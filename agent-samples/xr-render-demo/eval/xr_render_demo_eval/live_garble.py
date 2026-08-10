# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live STT-noise tests: garbled, corrected, and fragmentary prompts.

Each case sends one or more turns from a single participant (corrections need
the same conversation), scoring after every turn. A turn either expects a
scene change satisfying its check, or expects restraint: no new objects, or
no change at all. Wrong mutations on noisy input are the failure mode under
test; a clarifying reply with an untouched scene always passes a restraint
turn.
"""

import asyncio
import sys
import time

from xr_ai_hub import DataMessage, ParticipantEvent, ProcessorEndpoint
from xr_ai_nat.functions._service.rpc import RPCClient
from xr_render_scene import AddPrimitiveRequest, EmptyRequest, SceneClient

CANONICAL = {"position": {"x": 0, "y": 1.6, "z": 0}, "forward": {"x": 0, "y": 0, "z": -1},
             "right": {"x": 1, "y": 0, "z": 0}, "up": {"x": 0, "y": 1, "z": 0},
             "yaw_deg": 0.0, "pitch_deg": 0.0, "ts": 1}

GREEN_SPHERE = ("sphere", -1.0, 1.6, -1.5, 0, 0.8, 0, 0.1)
BLUE_SPHERE = ("sphere", 1.0, 1.6, -1.5, 0, 0, 1, 0.1)


def _new(before, after):
    return {k: v for k, v in after.items() if k not in before}


# Turn kinds: "change" polls until the scene differs and applies check(ids,
# before, after); "no_new_object" and "no_change" wait out the window and
# fail on any created object / any difference respectively.
CASES = [
    {
        "name": "homophone_anchor_spear",
        "fixtures": [GREEN_SPHERE, BLUE_SPHERE],
        "turns": [
            {"prompt": "Add a red box above the green spear.", "kind": "change",
             "check": lambda ids, before, after: any(
                 item.type == "box" and abs(item.position.x + 1.0) < 0.25 and item.position.y > 1.65
                 for item in _new(before, after).values())},
        ],
    },
    {
        "name": "homophone_color_blew",
        "fixtures": [],
        "turns": [
            {"prompt": "Make a blew sphere.", "kind": "change",
             "check": lambda ids, before, after: any(
                 item.type == "sphere" and item.color.b > 0.5
                 for item in _new(before, after).values())},
        ],
    },
    {
        "name": "homophone_shape_spear",
        "fixtures": [BLUE_SPHERE],
        "turns": [
            {"prompt": "Make a blue spear.", "kind": "change",
             "check": lambda ids, before, after: (
                 len(_new(before, after)) == 1
                 and all(item.type == "sphere" and item.color.b > 0.5 and item.position.z < -0.5
                         for item in _new(before, after).values()))},
        ],
    },
    {
        "name": "bare_create_after_work_stays_bare",
        "fixtures": [GREEN_SPHERE],
        "turns": [
            {"prompt": "Add a teal cube.", "kind": "change",
             "check": lambda ids, before, after: any(
                 item.type == "box" for item in _new(before, after).values())},
            {"prompt": "Make a red cube.", "kind": "change",
             "check": lambda ids, before, after: (
                 len(_new(before, after)) == 1
                 and all(item.type == "box" and item.color.r > 0.5
                         and abs(item.position.x) < 0.3 and item.position.y > 1.3
                         for item in _new(before, after).values()))},
        ],
    },
    {
        "name": "correction_never_creates",
        "fixtures": [GREEN_SPHERE, BLUE_SPHERE],
        "turns": [
            {"prompt": "Add a red box above the blue sphere.", "kind": "change",
             "check": lambda ids, before, after: any(
                 item.type == "box" for item in _new(before, after).values())},
            {"prompt": "That's the wrong sphere.", "kind": "no_new_object"},
        ],
    },
    {
        "name": "filler_fragment",
        "fixtures": [BLUE_SPHERE],
        "turns": [
            {"prompt": "Um, hang on a second.", "kind": "no_change"},
        ],
    },
    {
        "name": "truncated_command",
        "fixtures": [GREEN_SPHERE, BLUE_SPHERE],
        "turns": [
            {"prompt": "Put the sphere on the", "kind": "no_change"},
        ],
    },
    {
        "name": "truncated_then_completed",
        "fixtures": [("box", 0.5, 1.3, -1.2, 0, 0.8, 0.8, 0.15), BLUE_SPHERE],
        "turns": [
            {"prompt": "Put the sphere on the", "kind": "no_change"},
            {"prompt": "On the box.", "kind": "change",
             "check": lambda ids, before, after: (
                 not _new(before, after)
                 and abs(after[ids[1]].position.x - 0.5) < 0.3
                 and after[ids[1]].position.y > 1.3)},
        ],
    },
    {
        "name": "self_correction_single_create",
        "fixtures": [],
        "turns": [
            {"prompt": "Make a red, no, a green cube.", "kind": "change",
             "check": lambda ids, before, after: (
                 len(_new(before, after)) == 1
                 and all(item.type == "box" and item.color.g > 0.5 and item.color.r < 0.3
                         for item in _new(before, after).values()))},
        ],
    },
    {
        "name": "stutter_single_create",
        "fixtures": [],
        "turns": [
            {"prompt": "Add a a small cube cube.", "kind": "change",
             "check": lambda ids, before, after: (
                 len(_new(before, after)) == 1
                 and all(item.type == "box" for item in _new(before, after).values()))},
        ],
    },
]


async def clear_scene(scene):
    from xr_render_scene import RemovePrimitiveRequest
    state = await scene.get_scene_state(EmptyRequest())
    for item in state.objects:
        await scene.remove_primitive(RemovePrimitiveRequest(obj_id=item.id))


async def snapshot(scene):
    return {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}


async def run_turn(scene, endpoint, participant, turn, ids):
    before = await snapshot(scene)
    await endpoint.inject_data(DataMessage(
        participant_id=participant, topic="live.smoke.text",
        pts_us=time.time_ns() // 1_000, data=turn["prompt"].encode()))
    if turn["kind"] == "change":
        deadline = asyncio.get_running_loop().time() + 75
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(2)
            after = await snapshot(scene)
            if after != before:
                await asyncio.sleep(4)
                after = await snapshot(scene)
                ok = turn["check"](ids, before, after)
                return ok, _describe(before, after)
        return False, "no change within 75s"
    # Restraint turns: wait out the window, then judge what happened.
    await asyncio.sleep(30)
    after = await snapshot(scene)
    created = _new(before, after)
    if turn["kind"] == "no_new_object":
        gone = [k for k in before if k not in after]
        ok = not created and not gone
        return ok, _describe(before, after) if not ok else "no new or removed objects"
    return after == before, _describe(before, after) if after != before else "no change"


def _describe(before, after):
    parts = [f"NEW {k}:{v.model_dump()}" for k, v in _new(before, after).items()]
    parts += [f"GONE {k}" for k in before if k not in after]
    parts += [f"MOVED {k}" for k in before
              if k in after and after[k].model_dump() != before[k].model_dump()]
    return "; ".join(parts)[:400] or "unchanged"


async def main() -> None:
    tracking = RPCClient("tcp://127.0.0.1:8330", timeout_s=10.0)
    scene = SceneClient("tcp://127.0.0.1:8320")
    await clear_scene(scene)
    endpoint = ProcessorEndpoint(sub_addr="ipc:///tmp/xr_hub_pub", push_addr="ipc:///tmp/xr_hub_in")
    run_task = asyncio.create_task(endpoint.run())
    await asyncio.sleep(0.5)
    try:
        await tracking.call("set_sim_pose", CANONICAL)
    except Exception as error:
        print(f"openxr service refused set_sim_pose ({error}); set allow_sim_pose: true in "
              "agent-samples/xr-render-demo/yaml/openxr_service.yaml and restart the stack")
        await scene.close()
        await tracking.close()
        run_task.cancel()
        raise SystemExit(2) from None

    try:
        wanted = set(sys.argv[1:])
        passed = failed = 0
        for index, case in enumerate(CASES):
            if wanted and case["name"] not in wanted:
                continue
            participant = f"live-garble-{int(time.time())}-{index}"
            await endpoint.inject_participant_event(ParticipantEvent(
                participant_id=participant, joined=True, pts_us=time.time_ns() // 1_000))
            await asyncio.sleep(1.0)
            await clear_scene(scene)
            ids = []
            for prim_type, x, y, z, r, g, b, size in case["fixtures"]:
                result = await scene.add_primitive(AddPrimitiveRequest(
                    prim_type=prim_type, x=x, y=y, z=z, r=r, g=g, b=b, size=size))
                ids.append(result.id)
            ok, detail = True, ""
            for number, turn in enumerate(case["turns"], start=1):
                turn_ok, turn_detail = await run_turn(scene, endpoint, participant, turn, ids)
                detail = f"turn {number}: {turn_detail}"
                if not turn_ok:
                    ok = False
                    break
            verdict = "PASS" if ok else "FAIL"
            print(f"{verdict} {case['name']:28s} {detail}", flush=True)
            passed += ok
            failed += not ok
        print(f"\nlive garble: {passed} passed, {failed} failed", flush=True)
    finally:
        try:
            await tracking.call("clear_sim_pose", {})
        except Exception:
            pass
        await scene.close()
        await tracking.close()
        run_task.cancel()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
