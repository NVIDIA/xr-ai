# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the live render stack with typed text and verify against scene state."""

import asyncio
import sys
import time

from xr_ai_hub import DataMessage
from xr_render_scene import EmptyRequest, SceneClient

from ._live_endpoint import LiveEvalEndpoint, live_participant


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Make a red sphere."
    participant = f"live-smoke-{int(time.time())}"
    endpoint = LiveEvalEndpoint()
    scene = SceneClient("tcp://127.0.0.1:8320")
    before = {item.id: item for item in (await scene.get_scene_state(EmptyRequest())).objects}

    await asyncio.sleep(0.5)
    changed = False
    try:
        async with live_participant(endpoint, participant):
            await endpoint.inject_data(DataMessage(
                participant_id=participant,
                topic="",
                pts_us=time.time_ns() // 1_000,
                data=prompt.encode(),
            ))
            print(f"sent: {prompt!r}")

            deadline = asyncio.get_running_loop().time() + 60
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(2)
                after = {item.id: item for item in (await scene.get_scene_state(EmptyRequest())).objects}
                if after != before:
                    changed = True
                    for object_id in sorted(set(after) - set(before)):
                        print("added:", after[object_id].model_dump())
                    for object_id in sorted(set(before) - set(after)):
                        print("removed:", object_id)
                    for object_id in sorted(set(before) & set(after)):
                        if before[object_id] != after[object_id]:
                            print("changed:", after[object_id].model_dump())
                    break
    finally:
        await scene.close()
        await endpoint.close()
    if not changed:
        print("no scene change within 60s")
        sys.exit(1)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
