# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own participant polling and SDK resources for the gesture-capture sample."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time
from contextlib import AsyncExitStack
from pathlib import Path

from xr_ai_hub import DataMessage, FrameUnavailable, ParticipantEvent, ProcessorEndpoint
from xr_ai_logging import setup_logging
from xr_ai_models import load_models_config, make_vlm
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.hand_tracking import HandTrackingTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.image_crop import ImageCropTool
from xr_ai_tools.still_capture import StillCaptureTool
from xr_ai_tools.vision import ImageQueryTool

from .agent import GestureCaptureAgent
from .config import Settings, load_settings
from .diagnostics import GestureDiagnostics
from .feedback import GestureFeedback
from .gesture import RegionSelector

_LOG = logging.getLogger(__name__)
_HUB_PUB = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"
_REGION_SYSTEM_PROMPT = (Path(__file__).parent / "prompts/system.txt").read_text()


async def _run(settings: Settings, ready_file: Path | None) -> None:
    async with AsyncExitStack() as resources:
        endpoint = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        resources.callback(endpoint.close)
        resources.callback(endpoint.stop)
        images = ImageRegistry()
        resources.callback(images.clear)
        frames = CurrentFrameTool(endpoint=endpoint, images=images)
        capture = StillCaptureTool(endpoint=endpoint, images=images)
        resources.callback(capture.close)
        models = load_models_config(settings.models)
        vlm = make_vlm(models, "vlm")
        resources.push_async_callback(vlm.close)
        crop = ImageCropTool(images=images)
        query = ImageQueryTool(images=images, vlm=vlm, system_prompt=_REGION_SYSTEM_PROMPT)
        tasks: dict[str, asyncio.Task] = {}
        owned: set[asyncio.Task] = set()
        stopped = asyncio.Event()

        async def poll(participant_id: str) -> None:
            async with AsyncExitStack() as participant_resources:
                constructing = asyncio.create_task(asyncio.to_thread(
                    HandTrackingTool, images=images, model_path=settings.model_path, mode="video"))
                try:
                    hands = await asyncio.shield(constructing)
                except asyncio.CancelledError:
                    result = (await asyncio.gather(constructing, return_exceptions=True))[0]
                    if not isinstance(result, BaseException):
                        await asyncio.to_thread(result.close)
                    raise
                participant_resources.push_async_callback(asyncio.to_thread, hands.close)
                diagnostics = GestureDiagnostics(settings.gesture_diagnostics)
                feedback = GestureFeedback(endpoint, participant_id, enabled=settings.gesture_audio,
                                           max_gap_s=settings.max_gap_s)
                agent = GestureCaptureAgent(
                    frames=frames, hands=hands, capture=capture, crop=crop, query=query,
                    selector=RegionSelector(settings, diagnostics=diagnostics),
                    feedback=feedback, diagnostics=diagnostics,
                    settings=settings,
                )
                try:
                    while True:
                        try:
                            text = await agent.step(participant_id)
                            if text is not None:
                                await endpoint.send_return_data(DataMessage(
                                    participant_id=participant_id, topic="gesture.response",
                                    pts_us=time.time_ns() // 1_000, data=text.encode(),
                                ))
                        except FrameUnavailable:
                            agent.unavailable()
                            await feedback.stop()
                        except Exception:  # noqa: BLE001 — per-participant application boundary
                            # A failed observation/action must not kill other participants
                            # or rearm a held gesture. The user releases before retrying.
                            agent.unavailable()
                            await feedback.stop()
                            _LOG.exception("Gesture capture failed for %s", participant_id)
                        finally:
                            await agent.finish_turn()
                            images.release_owner(participant_id)
                        await asyncio.sleep(settings.poll_interval_s)
                finally:
                    await feedback.close()
                    frames.release(participant_id)

        async def participant(event: ParticipantEvent) -> None:
            if event.joined:
                if event.participant_id in tasks:
                    return
                task = asyncio.create_task(poll(event.participant_id))
                tasks[event.participant_id] = task
                owned.add(task)
                task.add_done_callback(owned.discard)
            else:
                previous = tasks.pop(event.participant_id, None)
                if previous:
                    previous.cancel()
                    await asyncio.gather(previous, return_exceptions=True)

        endpoint.on_participant(participant)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)
        endpoint_task = asyncio.create_task(endpoint.run())
        stop_task = asyncio.create_task(stopped.wait())
        try:
            await endpoint.wait_until_running()
            if ready_file:
                ready_file.touch()
            done, _ = await asyncio.wait((endpoint_task, stop_task), return_when=asyncio.FIRST_COMPLETED)
            if endpoint_task in done:
                await endpoint_task
        finally:
            # Stop admission before draining participants; native detection may outlive
            # a cancelled coroutine; each participant drains its detector on exit.
            endpoint_task.cancel()
            stop_task.cancel()
            for task in tuple(owned):
                task.cancel()
            await asyncio.gather(endpoint_task, stop_task, *owned, return_exceptions=True)
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)


def run() -> None:
    parser = argparse.ArgumentParser(description="Box or pinch-circle gesture → selected image → region VLM query.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    setup_logging("worker", namespace="gesture-capture")
    asyncio.run(_run(load_settings(args.config), args.ready_file))


if __name__ == "__main__":
    run()
