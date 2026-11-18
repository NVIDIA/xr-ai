# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the latest media observations without waiting for new samples."""

from .quality import _MESSAGES, QualityFeedback


def check_setup(quality: QualityFeedback) -> str:
    now = quality.clock()
    fps, resolution = quality.state.setup_video(now)
    messages = _MESSAGES["setup"]
    parts = [messages["video_fps"].format(fps=fps) if fps is not None else messages["fps_unavailable"]]
    if resolution is None:
        parts.append(messages["unavailable"])
    else:
        width, height = resolution
        parts.append(messages["resolution"].format(width=width, height=height))
        if min(width, height) < quality.config.min_short_edge:
            parts.append(_MESSAGES["conditions"]["low_resolution"])
    return " ".join(parts)
