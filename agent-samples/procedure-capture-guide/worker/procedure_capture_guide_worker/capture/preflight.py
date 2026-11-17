# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the latest media observations without waiting for new samples."""

from .quality import _MESSAGES, QualityFeedback


def check_setup(quality: QualityFeedback) -> str:
    now = quality.clock()
    data = quality.preview.fresh_telemetry(now)
    network = data.quality if data is not None else "unknown"
    fps, resolution = quality.preview.setup_video(now)
    messages = _MESSAGES["setup"]
    parts = [messages["network"].format(network=network)]
    parts.append(messages["video_fps"].format(fps=fps) if fps is not None else messages["fps_unavailable"])
    if resolution is None:
        parts.append(messages["unavailable"])
    else:
        width, height = resolution
        parts.append(messages["resolution"].format(width=width, height=height))
        if min(width, height) < quality.config.min_short_edge:
            parts.append(_MESSAGES["conditions"]["low_resolution"])
    if network in {"poor", "lost"}:
        parts.append(_MESSAGES["conditions"]["poor_network"])
    return " ".join(parts)
