# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the live render stack with typed text and verify against scene state."""

import asyncio
import sys
import time

from xr_ai_hub import DataMessage, ParticipantEvent, ProcessorEndpoint
from xr_render_scene import EmptyRequest, SceneClient


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Make a red sphere."
    participant = f"live-smoke-{int(time.time())}"
    endpoint = ProcessorEndpoint(sub_addr="ipc:///tmp/xr_hub_pub", push_addr="ipc:///tmp/xr_hub_in")
    scene = SceneClient("tcp://127.0.0.1:8320")
    before = {item.id: item for item in (await scene.get_scene_state(EmptyRequest())).objects}

    run_task = asyncio.create_task(endpoint.run())
    await asyncio.sleep(0.5)
    await endpoint.inject_participant_event(ParticipantEvent(
        participant_id=participant, joined=True, pts_us=time.time_ns() // 1_000))
    await asyncio.sleep(1.0)
    await endpoint.inject_data(DataMessage(
        participant_id=participant,
        topic="live.smoke.text",
        pts_us=time.time_ns() // 1_000,
        data=prompt.encode(),
    ))
    print(f"sent: {prompt!r}")

    try:
        deadline = asyncio.get_running_loop().time() + 60
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(2)
            after = {item.id: item for item in (await scene.get_scene_state(EmptyRequest())).objects}
            if after != before:
                for object_id in sorted(set(after) - set(before)):
                    print("added:", after[object_id].model_dump())
                for object_id in sorted(set(before) - set(after)):
                    print("removed:", object_id)
                for object_id in sorted(set(before) & set(after)):
                    if before[object_id] != after[object_id]:
                        print("changed:", after[object_id].model_dump())
                break
        else:
            print("no scene change within 60s")
            sys.exit(1)
    finally:
        await scene.close()
        run_task.cancel()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
