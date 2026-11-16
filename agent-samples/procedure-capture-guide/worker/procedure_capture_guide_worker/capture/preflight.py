# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report current media telemetry without delaying for a sampling window."""

from .quality import _MESSAGES, QualityConfig


async def check_setup(quality=None, *, config=None):
    config = quality.config if quality is not None else config or QualityConfig()
    data = None
    frame = None
    if quality is not None:
        now = quality.clock()
        if now - quality.latest_at <= config.telemetry_ttl_s:
            data = quality.latest
        if now - quality.latest_frame_at <= config.telemetry_ttl_s:
            frame = quality.latest_frame
    network = data.quality if data is not None else "unknown"
    live = data is not None and data.mode == "live" and data.streaming
    fps = data.sent_fps if live else None
    resolution = None
    if live:
        if frame is not None and frame.track_id == data.track_id:
            resolution = (frame.width, frame.height)
        elif data.sent_width is not None and data.sent_height is not None:
            resolution = (data.sent_width, data.sent_height)
    conditions = []
    if network in {"poor", "lost"}:
        conditions.append("poor_network")
    if resolution is not None and min(resolution) < config.min_short_edge:
        conditions.append("low_resolution")
    messages = _MESSAGES["setup"]
    parts = [messages["network"].format(network=network)]
    parts.append(messages["video_fps"].format(fps=fps) if fps is not None else messages["fps_unavailable"])
    parts.append(
        messages["resolution"].format(width=resolution[0], height=resolution[1])
        if resolution is not None
        else messages["unavailable"]
    )
    parts.extend(_MESSAGES["conditions"][name] for name in conditions)
    return {
        "network_quality": network,
        "fps": fps,
        "fps_source": "client_video" if fps is not None else "unavailable",
        "resolution": resolution,
        "conditions": conditions,
    }, " ".join(parts)
