# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from livekit import rtc


@dataclass(frozen=True)
class ByteStreamReadLimits:
    max_bytes: int
    idle_timeout_s: float
    total_timeout_s: float
    require_declared_size: bool = False

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.idle_timeout_s <= 0:
            raise ValueError("idle_timeout_s must be positive")
        if self.total_timeout_s <= 0:
            raise ValueError("total_timeout_s must be positive")


async def read_byte_stream(
    reader: rtc.ByteStreamReader,
    limits: ByteStreamReadLimits,
) -> bytes:
    try:
        declared_size = reader.info.size
        if declared_size is None:
            if limits.require_declared_size:
                raise ValueError("a declared size is required")
        elif declared_size < 0:
            raise ValueError("declared size must be nonnegative")
        elif declared_size > limits.max_bytes:
            raise ValueError(f"declared size exceeds {limits.max_bytes} bytes")

        data = bytearray()
        async with asyncio.timeout(limits.total_timeout_s):
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        anext(reader),
                        timeout=limits.idle_timeout_s,
                    )
                except StopAsyncIteration:
                    break
                data.extend(chunk)
                if len(data) > limits.max_bytes:
                    raise ValueError(f"observed size exceeds {limits.max_bytes} bytes")

        if declared_size is not None and len(data) != declared_size:
            raise ValueError(
                f"observed size {len(data)} does not match declared size {declared_size}"
            )
        return bytes(data)
    finally:
        reader.close()
