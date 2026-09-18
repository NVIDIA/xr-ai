# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from device_io_hub.transport.livekit._byte_stream import (
    ByteStreamReadLimits,
    read_byte_stream,
)


class _Reader:
    def __init__(self, chunks: tuple[bytes, ...], declared_size: int | None) -> None:
        self.info = SimpleNamespace(size=declared_size)
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    def close(self) -> None:
        self.closed = True


def _limits(**overrides) -> ByteStreamReadLimits:
    values = {
        "max_bytes": 8,
        "idle_timeout_s": 0.1,
        "total_timeout_s": 0.2,
    }
    values.update(overrides)
    return ByteStreamReadLimits(**values)


@pytest.mark.asyncio
async def test_byte_stream_reader_returns_complete_payload_and_closes() -> None:
    reader = _Reader((b"abc", b"def"), declared_size=6)

    assert await read_byte_stream(reader, _limits()) == b"abcdef"
    assert reader.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_size", [None, -1, 9])
async def test_byte_stream_reader_rejects_invalid_declared_size(
    declared_size: int | None,
) -> None:
    reader = _Reader((), declared_size=declared_size)
    limits = _limits(require_declared_size=declared_size is None)

    with pytest.raises(ValueError):
        await read_byte_stream(reader, limits)

    assert reader.closed


@pytest.mark.asyncio
async def test_byte_stream_reader_rejects_observed_size_over_limit() -> None:
    reader = _Reader((b"12345", b"6789"), declared_size=None)

    with pytest.raises(ValueError, match="observed size exceeds"):
        await read_byte_stream(reader, _limits())

    assert reader.closed


@pytest.mark.asyncio
async def test_byte_stream_reader_rejects_declared_size_mismatch() -> None:
    reader = _Reader((b"123",), declared_size=4)

    with pytest.raises(ValueError, match="does not match"):
        await read_byte_stream(reader, _limits())

    assert reader.closed


@pytest.mark.asyncio
async def test_byte_stream_reader_enforces_idle_timeout_and_closes() -> None:
    class StalledReader(_Reader):
        async def __anext__(self) -> bytes:
            await asyncio.Event().wait()
            raise StopAsyncIteration

    reader = StalledReader((), declared_size=None)

    with pytest.raises(TimeoutError):
        await read_byte_stream(reader, _limits(idle_timeout_s=0.01))

    assert reader.closed


@pytest.mark.asyncio
async def test_byte_stream_reader_enforces_total_timeout_and_closes() -> None:
    class SlowReader(_Reader):
        async def __anext__(self) -> bytes:
            await asyncio.sleep(0.01)
            return b"x"

    reader = SlowReader((), declared_size=None)

    with pytest.raises(TimeoutError):
        await read_byte_stream(
            reader,
            _limits(max_bytes=1_000, idle_timeout_s=0.1, total_timeout_s=0.02),
        )

    assert reader.closed
