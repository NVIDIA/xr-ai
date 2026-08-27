# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read already-routed return traffic from the media-hub publish socket."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

import zmq
import zmq.asyncio
from xr_ai_hub import AudioChunk, DataMessage, MsgType, ReturnAudioFlush, decode
from xr_ai_hub._capture import CAPTURE_PUBLISH_PREFIX

AudioCallback = Callable[[AudioChunk], Awaitable[None]]
DataCallback = Callable[[DataMessage], Awaitable[None]]
FlushCallback = Callable[[ReturnAudioFlush], Awaitable[None]]


class ReturnTrafficSubscriber:
    """Capture outbound hub traffic without changing the agent SDK surface."""

    def __init__(self, sub_addr: str) -> None:
        self._socket: zmq.asyncio.Socket = zmq.asyncio.Context.instance().socket(zmq.SUB)
        self._socket.connect(sub_addr)
        for prefix in (
            b"return_audio.",
            b"return_data.",
            b"return_audio_flush.",
            CAPTURE_PUBLISH_PREFIX,
        ):
            self._socket.setsockopt(zmq.SUBSCRIBE, prefix)
        self._audio_callbacks: list[AudioCallback] = []
        self._data_callbacks: list[DataCallback] = []
        self._flush_callbacks: list[FlushCallback] = []

    def on_audio(self, callback: AudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_data(self, callback: DataCallback) -> None:
        self._data_callbacks.append(callback)

    def on_flush(self, callback: FlushCallback) -> None:
        self._flush_callbacks.append(callback)

    async def run(self) -> None:
        while True:
            _topic, payload = await self._socket.recv_multipart()
            type_id, message = decode(payload)
            if type_id == MsgType.RETURN_AUDIO:
                for callback in self._audio_callbacks:
                    await callback(message)
            elif type_id == MsgType.RETURN_DATA:
                for callback in self._data_callbacks:
                    await callback(message)
            elif type_id == MsgType.RETURN_AUDIO_FLUSH:
                for callback in self._flush_callbacks:
                    await callback(message)

    def close(self) -> None:
        self._socket.close(linger=0)
