# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose-matrix live test: sim pose -> typed prompt -> scene-state verdict."""

import asyncio
import math
import sys
import time

from xr_ai_hub import DataMessage
from xr_ai_tools.rpc import RPCClient
from xr_render_scene import EmptyRequest, SceneClient

from ._live_endpoint import LiveEvalEndpoint, live_participant

CANONICAL = {"position": {"x": 0, "y": 1.6, "z": 0}, "forward": {"x": 0, "y": 0, "z": -1},
             "right": {"x": 1, "y": 0, "z": 0}, "up": {"x": 0, "y": 1, "z": 0},
             "yaw_deg": 0.0, "pitch_deg": 0.0, "ts": 1}


def pose(x, y, z, yaw_deg):
    yaw = math.radians(yaw_deg)
    forward = {"x": -math.sin(yaw), "y": 0.0, "z": -math.cos(yaw)}
    right = {"x": math.cos(yaw), "y": 0.0, "z": -math.sin(yaw)}
    return {"position": {"x": x, "y": y, "z": z}, "forward": forward, "right": right,
            "up": {"x": 0, "y": 1, "z": 0}, "yaw_deg": yaw_deg, "pitch_deg": 0.0, "ts": 1}


POSES = {
    "canonical": CANONICAL,
    "user_live": pose(0.5, 1.33, 0.7, 0),
    "walked_off": pose(2.0, 1.6, 1.5, 0),
    "turned_180": pose(0.0, 1.6, 0.0, 180),
    "turned_left_90": pose(0.0, 1.45, -0.5, 90),
}

PROMPT_SETS = {
    "canonical": [("Make a red sphere.", 1.5), ("Add a green cube.", 1.5),
                  ("Create a yellow sphere two meters ahead of me.", 2.0)],
    "user_live": [("Make an orange sphere.", 1.5), ("Add a purple cube.", 1.5),
                  ("Create a cyan sphere two meters ahead of me.", 2.0)],
    "walked_off": [("Make a white sphere.", 1.5), ("Add a black cube.", 1.5),
                   ("Create a magenta sphere two meters ahead of me.", 2.0)],
    "turned_180": [("Make a blue sphere.", 1.5), ("Add a yellow cube.", 1.5),
                   ("Create a red sphere two meters ahead of me.", 2.0)],
    "turned_left_90": [("Make a green sphere.", 1.5), ("Add a cyan cube.", 1.5),
                       ("Create a white sphere two meters ahead of me.", 2.0)],
}


def expected_spot(p, distance):
    fx, fz = p["forward"]["x"], p["forward"]["z"]
    magnitude = math.sqrt(fx * fx + fz * fz)
    fx, fz = fx / magnitude, fz / magnitude
    return (p["position"]["x"] + fx * distance, p["position"]["y"], p["position"]["z"] + fz * distance)



async def clear_scene(scene):
    from xr_render_scene import RemovePrimitiveRequest
    state = await scene.get_scene_state(EmptyRequest())
    for item in state.objects:
        await scene.remove_primitive(RemovePrimitiveRequest(obj_id=item.id))

async def main() -> None:
    tracking = RPCClient("tcp://127.0.0.1:8330", timeout_s=10.0)
    scene = SceneClient("tcp://127.0.0.1:8320")
    await clear_scene(scene)
    endpoint = LiveEvalEndpoint()
    await asyncio.sleep(0.5)

    failed = 0
    try:
        passed = failed = 0
        case_index = 0
        for pose_name, p in POSES.items():
            try:
                await tracking.call("set_sim_pose", p)
            except Exception as error:
                print(f"openxr service refused set_sim_pose ({error}); set allow_sim_pose: true in "
                      "yaml/openxr_service.yaml and restart the stack")
                raise SystemExit(2) from None
            for prompt, distance in PROMPT_SETS[pose_name]:
                participant = f"live-pose-{int(time.time())}-{case_index}"
                async with live_participant(endpoint, participant):
                    await clear_scene(scene)
                    before = {i.id for i in (await scene.get_scene_state(EmptyRequest())).objects}
                    await endpoint.inject_data(DataMessage(
                        participant_id=participant, topic="",
                        pts_us=time.time_ns() // 1_000, data=prompt.encode()))
                    new = None
                    deadline = asyncio.get_running_loop().time() + 75
                    while asyncio.get_running_loop().time() < deadline:
                        await asyncio.sleep(2)
                        objects = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
                        fresh = [objects[k] for k in objects.keys() - before]
                        if fresh:
                            await asyncio.sleep(3)
                            objects = {i.id: i for i in (await scene.get_scene_state(EmptyRequest())).objects}
                            fresh = [objects[k] for k in objects.keys() - before]
                            new = fresh
                            break
                    if not new:
                        print(f"FAIL {pose_name:15s} {prompt!r}: nothing created")
                        failed += 1
                        continue
                    if len(new) > 1:
                        print(f"FAIL {pose_name:15s} {prompt!r}: {len(new)} objects created")
                        failed += 1
                        continue
                    item = new[0]
                    ex, ey, ez = expected_spot(p, distance)
                    dx, dy, dz = item.position.x - ex, item.position.y - ey, item.position.z - ez
                    miss = math.sqrt(dx * dx + dy * dy + dz * dz)
                    verdict = "PASS" if miss <= 0.25 else "FAIL"
                    print(f"{verdict} {pose_name:15s} {prompt!r}: {item.type} at "
                          f"({item.position.x:.2f},{item.position.y:.2f},{item.position.z:.2f}) "
                          f"expected ({ex:.2f},{ey:.2f},{ez:.2f}) miss={miss:.2f}")
                    passed += verdict == "PASS"
                    failed += verdict == "FAIL"
                case_index += 1
        print(f"\npose matrix: {passed} passed, {failed} failed", flush=True)
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
