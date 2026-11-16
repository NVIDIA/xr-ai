# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check network, received frame rate and resolution without recording or models."""

import asyncio
import time

from xr_ai_hub import FrameUnavailable

from .quality import _MESSAGES, QualityConfig, QualityState


async def check_setup(acquire, quality=None, *, capture_fps=2.0, duration_s=None, config=None):
    config = quality.config if quality is not None else config or QualityConfig()
    duration_s = config.setup_duration_s if duration_s is None else duration_s
    state = QualityState(config, capture_fps)
    if quality is not None:
        quality.state = state
        if quality.latest is not None and time.monotonic() - quality.latest_at <= config.telemetry_ttl_s:
            state.telemetry_received(quality.latest)
    started = time.monotonic()
    seen = set()
    samples = 0
    try:
        async with asyncio.timeout(duration_s):
            while True:
                tick = time.monotonic()
                try:
                    frame = await acquire()
                except (FrameUnavailable, TimeoutError):
                    state.sample(None)
                else:
                    identity = (frame.track_id, frame.timestamp_us, frame.sequence)
                    if identity not in seen:
                        seen.add(identity)
                        samples += 1
                        state.sample(frame)
                state.evaluate()
                await asyncio.sleep(max(0, 1 / capture_fps - (time.monotonic() - tick)))
    except TimeoutError:
        pass
    finally:
        if quality is not None and quality.state is state:
            quality.state = None
    conditions = sorted(state.evaluate())
    network = (
        state.telemetry.quality
        if state.telemetry is not None and state.clock() - state.telemetry_at <= config.telemetry_ttl_s
        else "unknown"
    )
    received = state.metrics["received_fps"]
    fps = received if received is not None else samples / max(time.monotonic() - started, 0.001)
    messages = _MESSAGES["setup"]
    parts = [messages["network"].format(network=network)]
    parts.append(messages["video_fps" if received is not None else "sampled_fps"].format(fps=fps))
    if state.resolution is not None:
        width, height = state.resolution
        parts.append(messages["resolution"].format(width=width, height=height))
    else:
        parts.append(messages["unavailable"])
    parts.extend(_MESSAGES["conditions"][name] for name in conditions)
    return {
        "network_quality": network,
        "fps": fps,
        "fps_source": "received_video" if received is not None else "sampled_images",
        "resolution": state.resolution,
        "conditions": conditions,
    }, " ".join(parts)
