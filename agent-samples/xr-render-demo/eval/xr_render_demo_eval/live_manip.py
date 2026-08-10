# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live manipulation tests: fixture objects via scene RPC, prompt, scene-diff verdict."""

import asyncio
import math
import sys
import time

from xr_ai_hub import DataMessage, ParticipantEvent, ProcessorEndpoint
from xr_ai_nat.functions._service.rpc import RPCClient
from xr_render_scene import AddPrimitiveRequest, EmptyRequest, SceneClient

CANONICAL = {"position": {"x": 0, "y": 1.6, "z": 0}, "forward": {"x": 0, "y": 0, "z": -1},
             "right": {"x": 1, "y": 0, "z": 0}, "up": {"x": 0, "y": 1, "z": 0},
             "yaw_deg": 0.0, "pitch_deg": 0.0, "ts": 1}

# Each case: fixtures (type, x, y, z, r, g, b, size), prompt, then checks on
# the scene: expressions over {id: object} keyed by fixture creation order.
CASES = [
    {
        "name": "move_left_one_meter",
        "fixtures": [("box", 0.0, 1.7, -1.5, 0, 0.4, 1, 0.1)],
        "prompt": "Move the cube left one meter.",
        "check": lambda ids, o: abs(o[ids[0]].position.x + 1.0) < 0.15 and abs(o[ids[0]].position.z + 1.5) < 0.15,
    },
    {
        "name": "recolor",
        "fixtures": [("sphere", 0.0, 1.6, -1.5, 1, 0, 0, 0.1)],
        "prompt": "Make the sphere green.",
        "check": lambda ids, o: o[ids[0]].color.g > 0.5 and o[ids[0]].color.r < 0.3,
    },
    {
        "name": "containment",
        "fixtures": [("sphere", 1.0, 1.6, -1.5, 1, 0, 0, 0.1), ("box", -0.5, 1.3, -1.5, 0, 0.4, 1, 0.25)],
        "prompt": "Put the sphere in the cube.",
        "check": lambda ids, o: math.dist(
            (o[ids[0]].position.x, o[ids[0]].position.y, o[ids[0]].position.z),
            (o[ids[1]].position.x, o[ids[1]].position.y, o[ids[1]].position.z)) < 0.2,
    },
    {
        "name": "bring_closer",
        "fixtures": [("sphere", 0.0, 1.6, -3.0, 1, 1, 0, 0.1)],
        "prompt": "Bring the sphere closer to me.",
        "check": lambda ids, o: o[ids[0]].position.z > -2.9,
    },
    {
        "name": "remove_left_of_two",
        "fixtures": [("box", -1.0, 1.5, -2.0, 0.4, 0.4, 0.4, 0.2), ("box", 1.0, 1.5, -2.0, 0.4, 0.4, 0.4, 0.2)],
        "prompt": "Remove the cube on the left.",
        "check": lambda ids, o: ids[0] not in o and ids[1] in o,
    },
    {
        "name": "double_size",
        "fixtures": [("sphere", 0.3, 1.5, -1.2, 0, 0, 1, 0.1)],
        "prompt": "Double its size.",
        "check": lambda ids, o: abs(o[ids[0]].size - 0.2) < 0.02,
    },
    {
        "name": "anchored_add_right_anchor",
        "fixtures": [("sphere", -1.0, 1.6, -1.5, 0, 0, 1, 0.1), ("sphere", 1.0, 1.6, -1.5, 0, 0.8, 0, 0.1)],
        "prompt": "Add a red box above the blue sphere.",
        "check": lambda ids, o: any(
            item.type == "box" and abs(item.position.x + 1.0) < 0.2 and item.position.y > 1.65
            for key, item in o.items() if key not in ids
        ),
    },
    {
        "name": "anchored_add_garbled",
        "fixtures": [("sphere", -1.0, 1.6, -1.5, 0, 0, 1, 0.1), ("sphere", 1.0, 1.6, -1.5, 0, 0.8, 0, 0.1)],
        "prompt": "It had a red box above the blue sphere.",
        "no_change_ok": True,
        "check": lambda ids, o: (
            not any(item.type == "box" and abs(item.position.x - 1.0) < 0.3
                    for key, item in o.items() if key not in ids)
        ),
    },
    {
        "name": "swap",
        "fixtures": [("sphere", 1.0, 1.6, -1.5, 1, 0, 0, 0.1), ("box", -1.0, 1.6, -1.5, 0, 0.4, 1, 0.1)],
        "prompt": "Swap the sphere and the cube.",
        "check": lambda ids, o: abs(o[ids[0]].position.x + 1.0) < 0.15 and abs(o[ids[1]].position.x - 1.0) < 0.15,
    },
]



async def clear_scene(scene):
    from xr_render_scene import RemovePrimitiveRequest
    state = await scene.get_scene_state(EmptyRequest())
    for item in state.objects:
        await scene.remove_primitive(RemovePrimitiveRequest(obj_id=item.id))

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
            participant = f"live-manip-{int(time.time())}-{index}"
            await endpoint.inject_participant_event(ParticipantEvent(
                participant_id=participant, joined=True, pts_us=time.time_ns() // 1_000))
            await asyncio.sleep(1.0)
            await clear_scene(scene)
            ids = []
            for prim_type, x, y, z, r, g, b, size in case["fixtures"]:
                result = await scene.add_primitive(AddPrimitiveRequest(
                    prim_type=prim_type, x=x, y=y, z=z, r=r, g=g, b=b, size=size))
                ids.append(result.id)
            snapshot = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
            await endpoint.inject_data(DataMessage(
                participant_id=participant, topic="live.smoke.text",
                pts_us=time.time_ns() // 1_000, data=case["prompt"].encode()))
            verdict = "PASS" if case.get("no_change_ok") else "FAIL"
            detail = "no change within 75s"
            deadline = asyncio.get_running_loop().time() + 75
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(2)
                objects = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
                if objects != snapshot:
                    await asyncio.sleep(4)
                    objects = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
                    try:
                        ok = case["check"](ids, objects)
                    except Exception as error:
                        ok, detail = False, f"check error: {error}"
                    else:
                        new_ids = [k for k in objects if k not in snapshot]
                        detail = "; ".join(
                            [f"{i}:{'GONE' if i not in objects else objects[i].model_dump()}" for i in ids]
                            + [f"NEW {k}:{objects[k].model_dump()}" for k in new_ids]
                        )[:400]
                    verdict = "PASS" if ok else "FAIL"
                    break
            print(f"{verdict} {case['name']:22s} {detail}", flush=True)
            passed += verdict == "PASS"
            failed += verdict == "FAIL"
        print(f"\nlive manipulation: {passed} passed, {failed} failed", flush=True)
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
