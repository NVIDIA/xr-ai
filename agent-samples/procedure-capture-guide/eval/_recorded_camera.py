# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supply captured camera evidence to the shared guide evaluation harness.

Validate and identify the packet's saved frames, then publish each once at its
original relative time. Playback never refreshes a stale last frame or waits
for verification to succeed, so timing gaps remain visible in the results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path


def capture_frames(packet: Path) -> list[dict]:
    directory = packet.resolve().parent
    metadata = json.loads(packet.read_text())
    if metadata.get("status") != "complete":
        raise ValueError("evaluation requires a complete capture packet")

    def local_file(relative):
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"invalid packet file: {relative}")
        return path

    index = local_file(metadata.get("files", {}).get("frame_index", "frames/index.jsonl"))
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("capture needs a nonempty frame index")
    identities = set()
    previous = -1
    for row in rows:
        timestamp = row["timestamp_us"]
        if type(timestamp) is not int or timestamp < previous:
            raise ValueError("capture needs chronological integer timestamps")
        if row["frame_id"] in identities:
            raise ValueError("duplicate source frame ID")
        identities.add(row["frame_id"])
        previous = timestamp
        path = local_file(row["path"])
        row["source_path"] = str(path)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


async def feed_recording(camera, rows, start, log):
    await start.wait()
    origin = time.monotonic()
    wall_origin = time.time_ns() // 1000
    for row in rows:
        offset = (row["timestamp_us"] - rows[0]["timestamp_us"]) / 1e6
        await asyncio.sleep(max(0, origin + offset - time.monotonic()))
        camera.path = Path(row["source_path"])
        await camera.publish(timestamp_us=wall_origin + row["timestamp_us"] - rows[0]["timestamp_us"])
        log("source_frame", sequence=camera.sequence, source=row)
    log("camera_finished")
