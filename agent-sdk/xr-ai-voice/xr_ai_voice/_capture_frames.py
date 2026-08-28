# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private frames that align capture metadata with the media queue."""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.frames.frames import Frame


@dataclass
class _CaptureTtsCaptionFrame(Frame):
    text: str
