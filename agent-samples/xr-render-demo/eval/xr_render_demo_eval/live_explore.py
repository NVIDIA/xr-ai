# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exploratory live driver: novel user-style utterances scored by invariants.

Unlike the case tiers, nothing here asserts an exact outcome. Each utterance
carries an intent class, and the scene diff after the turn is checked against
that class's invariants:

  create   at least one new object; nothing pre-existing moved or removed
  mutate   at least one pre-existing object changed; nothing added or removed
  remove   at least one pre-existing object gone; nothing added
  none     no scene change at all (questions, comments, courtesies)

Utterances are deliberately phrased unlike both the case corpus and the
prompt examples; when a turn violates its invariant, promote the utterance
into the appropriate tier as a permanent case, then fix.

Run subsets: `live_explore.py [utterance-index ...]`.
"""

import asyncio
import sys
import time

from xr_ai_hub import DataMessage, ParticipantEvent, ProcessorEndpoint
from xr_ai_tools.rpc import RPCClient
from xr_render_scene import AddPrimitiveRequest, EmptyRequest, SceneClient

CANONICAL = {"position": {"x": 0, "y": 1.6, "z": 0}, "forward": {"x": 0, "y": 0, "z": -1},
             "right": {"x": 1, "y": 0, "z": 0}, "up": {"x": 0, "y": 1, "z": 0},
             "yaw_deg": 0.0, "pitch_deg": 0.0, "ts": 1}

# One shared starting scene: a couple of referents so pronouns and vague
# references have something to bite on.
FIXTURES = [
    ("box", 0.4, 1.3, -1.2, 1, 0.5, 0, 0.12),
    ("sphere", -0.6, 1.5, -1.4, 0.6, 0, 1, 0.1),
]

# (utterance, intent): phrased like speech, not like the corpus.
PROBES = [
    ("Can you stick a ball over there by the box?", "create"),
    ("Throw another cube in somewhere.", "create"),
    ("I want something floating right here in front of my face.", "create"),
    ("Give me one more of those spheres.", "create"),
    ("Pop a little one up near the ceiling.", "create"),
    ("Actually make it way bigger.", "mutate"),
    ("Scoot the box over a bit.", "mutate"),
    ("Can the sphere come down to the floor?", "mutate"),
    ("Flip their spots.", "mutate"),
    ("Push everything back a little, it feels crowded.", "mutate"),
    ("Get rid of that sphere.", "remove"),
    ("Clear out the box, I don't need it.", "remove"),
    ("What have we got in here so far?", "none"),
    ("This looks pretty good actually.", "none"),
    ("Hang on, my coffee is ready.", "none"),
    ("Never mind, forget that.", "none"),
    ("Wait no.", "none"),
    ("How big is the box?", "none"),
    ("Could you, um, you know the thing next to the other one, yeah that.", "none"),
    ("Do the same thing again but on the other side.", "any"),
    ("Put a yellow cube to the left of the blue cube.", "create"),
]


async def clear_scene(scene):
    from xr_render_scene import RemovePrimitiveRequest
    state = await scene.get_scene_state(EmptyRequest())
    for item in state.objects:
        await scene.remove_primitive(RemovePrimitiveRequest(obj_id=item.id))


async def snapshot(scene):
    return {i.id: i.model_dump_json() for i in (await scene.get_scene_state(EmptyRequest())).objects}


def judge(intent, before, after):
    added = [k for k in after if k not in before]
    gone = [k for k in before if k not in after]
    changed = [k for k in before if k in after and after[k] != before[k]]
    facts = f"added={added} gone={gone} changed={changed}"
    if intent == "create":
        return (len(added) >= 1 and not gone and not changed), facts
    if intent == "mutate":
        return (len(changed) >= 1 and not added and not gone), facts
    if intent == "remove":
        return (len(gone) >= 1 and not added), facts
    if intent == "none":
        return (not added and not gone and not changed), facts
    return True, facts


async def main() -> None:
    tracking = RPCClient("tcp://127.0.0.1:8330", timeout_s=10.0)
    scene = SceneClient("tcp://127.0.0.1:8320")
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
        wanted = {int(argument) for argument in sys.argv[1:]}
        passed = failed = 0
        for index, (utterance, intent) in enumerate(PROBES):
            if wanted and index not in wanted:
                continue
            participant = f"live-explore-{int(time.time())}-{index}"
            await endpoint.inject_participant_event(ParticipantEvent(
                participant_id=participant, joined=True, pts_us=time.time_ns() // 1_000))
            await asyncio.sleep(1.0)
            await clear_scene(scene)
            for prim_type, x, y, z, r, g, b, size in FIXTURES:
                await scene.add_primitive(AddPrimitiveRequest(
                    prim_type=prim_type, x=x, y=y, z=z, r=r, g=g, b=b, size=size))
            before = await snapshot(scene)
            await endpoint.inject_data(DataMessage(
                participant_id=participant, topic="",
                pts_us=time.time_ns() // 1_000, data=utterance.encode()))
            # Expected-change intents may finish early; restraint intents must
            # wait out the window.
            deadline = asyncio.get_running_loop().time() + (30 if intent == "none" else 75)
            after = before
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(2)
                after = await snapshot(scene)
                if intent != "none" and after != before:
                    await asyncio.sleep(5)
                    after = await snapshot(scene)
                    break
            ok, facts = judge(intent, before, after)
            verdict = "PASS" if ok else "FAIL"
            print(f"{verdict} [{index:02d}] ({intent:6s}) {utterance!r} {facts}", flush=True)
            passed += ok
            failed += not ok
        print(f"\nlive explore: {passed} passed, {failed} failed", flush=True)
    finally:
        try:
            await tracking.call("clear_sim_pose", {})
        except Exception:
            pass
        await scene.close()
        await tracking.close()
        run_task.cancel()
    sys.exit(1 if failed else 0)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
