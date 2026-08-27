# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free NV12 caption composition for recorded video."""
from __future__ import annotations

import textwrap

import numpy as np
from xr_ai_hub import FrameData

from device_io_hub.video._recorder import _to_nv12

# Five-bit rows for a compact 5x7 recording font. Input is normalized to upper
# case so the table stays small and deterministic across server installations.
_FONT: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0, 0, 0), "?": (14, 17, 1, 2, 4, 0, 4),
    "A": (14, 17, 17, 31, 17, 17, 17), "B": (30, 17, 17, 30, 17, 17, 30),
    "C": (14, 17, 16, 16, 16, 17, 14), "D": (30, 17, 17, 17, 17, 17, 30),
    "E": (31, 16, 16, 30, 16, 16, 31), "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 15), "H": (17, 17, 17, 31, 17, 17, 17),
    "I": (14, 4, 4, 4, 4, 4, 14), "J": (7, 2, 2, 2, 18, 18, 12),
    "K": (17, 18, 20, 24, 20, 18, 17), "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17), "N": (17, 25, 21, 19, 17, 17, 17),
    "O": (14, 17, 17, 17, 17, 17, 14), "P": (30, 17, 17, 30, 16, 16, 16),
    "Q": (14, 17, 17, 17, 21, 18, 13), "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30), "T": (31, 4, 4, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 17, 17, 14), "V": (17, 17, 17, 17, 17, 10, 4),
    "W": (17, 17, 17, 21, 21, 21, 10), "X": (17, 17, 10, 4, 10, 17, 17),
    "Y": (17, 17, 10, 4, 4, 4, 4), "Z": (31, 1, 2, 4, 8, 16, 31),
    "0": (14, 17, 19, 21, 25, 17, 14), "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 2, 4, 8, 31), "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2), "5": (31, 16, 16, 30, 1, 1, 30),
    "6": (14, 16, 16, 30, 17, 17, 14), "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14), "9": (14, 17, 17, 15, 1, 1, 14),
    ".": (0, 0, 0, 0, 0, 12, 12), ",": (0, 0, 0, 0, 12, 12, 8),
    ":": (0, 12, 12, 0, 12, 12, 0), ";": (0, 12, 12, 0, 12, 12, 8),
    "!": (4, 4, 4, 4, 4, 0, 4), "-": (0, 0, 0, 31, 0, 0, 0),
    "_": (0, 0, 0, 0, 0, 0, 31), "/": (1, 2, 2, 4, 8, 8, 16),
    "'": (4, 4, 2, 0, 0, 0, 0), '"': (10, 10, 5, 0, 0, 0, 0),
    "(": (2, 4, 8, 8, 8, 4, 2), ")": (8, 4, 2, 2, 2, 4, 8),
    "+": (0, 4, 4, 31, 4, 4, 0), "=": (0, 0, 31, 0, 31, 0, 0),
}


def _draw_line(luma: np.ndarray, text: str, x: int, y: int, scale: int) -> None:
    cursor = x
    for char in text.upper():
        rows = _FONT.get(char, _FONT["?"])
        for row_index, bits in enumerate(rows):
            for column in range(5):
                if bits & (1 << (4 - column)):
                    y0 = y + row_index * scale
                    x0 = cursor + column * scale
                    luma[y0:y0 + scale, x0:x0 + scale] = 235
        cursor += 6 * scale


def compose_caption(
    frame: FrameData,
    caption: str,
    *,
    max_lines: int,
) -> tuple[np.ndarray, int, int]:
    """Append a stable caption panel while preserving every sensor pixel."""
    if frame.width % 2 or frame.height % 2:
        raise ValueError("NV12 capture requires even video dimensions")
    source = _to_nv12(frame.data, frame.width, frame.height, frame.fmt)
    if source is None:
        raise ValueError(f"unsupported capture pixel format: {frame.fmt!r}")

    scale = max(1, min(4, frame.width // 480 + 1))
    padding = 4 * scale
    line_height = 8 * scale
    panel_height = padding * 2 + max_lines * line_height
    panel_height += panel_height % 2
    output_height = frame.height + panel_height

    output = np.empty((output_height * 3 // 2, frame.width), dtype=np.uint8)
    output[:output_height] = 16
    output[output_height:] = 128
    output[:frame.height] = source[:frame.height]
    output[output_height:output_height + frame.height // 2] = source[frame.height:]
    output[frame.height:frame.height + 2] = 96

    max_chars = max(1, (frame.width - 2 * padding) // (6 * scale))
    normalized = " ".join(caption.split())
    lines = textwrap.wrap(
        normalized,
        width=max_chars,
        replace_whitespace=True,
        drop_whitespace=True,
    )[:max_lines]
    for index, line in enumerate(lines):
        _draw_line(
            output[:output_height],
            line,
            padding,
            frame.height + padding + index * line_height,
            scale,
        )
    return np.ascontiguousarray(output), frame.width, output_height
