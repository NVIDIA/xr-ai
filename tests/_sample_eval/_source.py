# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own test hub and real-time input in a process isolated from native workers."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np

from ._recording import _Recording


async def _receive(connection):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def readable():
        if future.done():
            return
        try:
            future.set_result(connection.recv())
        except (EOFError, OSError) as exc:
            future.set_exception(RuntimeError(f"evaluation control peer disconnected: {exc}"))

    loop.add_reader(connection.fileno(), readable)
    try:
        return await future
    finally:
        loop.remove_reader(connection.fileno())


async def _guarded(awaitable, owner):
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait([task, owner], return_when=asyncio.FIRST_COMPLETED)
        if owner in done:
            await owner
            raise RuntimeError("evaluation owner stopped the source")
        return await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _serve(scenario, audio, images, scenes, output, push, pub, name, connection, drain):
    from device_io_hub.ipc import ConnectorEndpoint, HubEndpoint

    from ._replay import _replay

    hub = HubEndpoint(pull_addr=push, pub_addr=pub)
    connector, owner, tasks = None, None, []
    with _Recording(output, filename="source-events.jsonl", hooks=False) as recording:
        try:
            owner = asyncio.create_task(_receive(connection))
            tasks.append(asyncio.create_task(hub.run()))
            connector = ConnectorEndpoint(
                push_addr=push,
                sub_addr=pub,
                connector_id=name,
                shm_name=name.replace("-", "_"),
                num_slots=4,
                max_frame_bytes=6_000_000,
            )

            async def audio_out(chunk):
                samples = np.frombuffer(chunk.data, dtype="<f4")
                rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0
                recording.emit("audio_out", rms=rms, samples=chunk.samples, sample_rate=chunk.sample_rate)

            connector.on_return_audio(audio_out)
            await _guarded(connector.register(), owner)
            tasks.append(asyncio.create_task(connector.run()))
            connection.send("ready")
            command = await owner
            if command != {"command": "join"}:
                raise ValueError("source expected join")
            owner = asyncio.create_task(_receive(connection))
            await _guarded(connector.notify_participant_joined("fixture-speaker", pts_us=time.time_ns() // 1000), owner)
            connection.send("joined")
            command = await owner
            if command.get("command") != "start":
                raise ValueError("source expected start")
            owner = asyncio.create_task(_receive(connection))
            connection.send("started")

            def render(scene):
                return scenes[json.dumps(scene, sort_keys=True)]

            await _guarded(
                _replay(
                    scenario,
                    audio,
                    images,
                    render,
                    connector,
                    owner,
                    recording,
                    started=command["monotonic"],
                    audio_epoch_us=command["epoch_us"],
                ),
                owner,
            )
            await _guarded(asyncio.sleep(drain), owner)
            connection.send("done")
            command = await owner
            if command != {"command": "stop"}:
                raise ValueError("source expected stop")
        finally:
            if owner is not None:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if connector is not None:
                connector.stop()
                connector.close()
            hub.stop()
            hub.close()


def _main(*args):
    connection = args[-2]
    try:
        asyncio.run(_serve(*args))
    finally:
        connection.close()
