# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
xr_media_hub.video — NVENC video recording for the XR-Media-Hub.

Records incoming video frames to Annex B H.264 chunk files using NVENC
for consistent-quality hardware encoding.  Chunks are concatenable with
`cat`, and each starts with an IDR frame.
"""
from ._recorder import VideoRecorder, VideoRecorderConfig

__all__ = ["VideoRecorder", "VideoRecorderConfig"]
