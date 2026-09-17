# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read already-routed return traffic from the media-hub publish socket."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import zmq
import zmq.asyncio
from xr_ai_hub import (
    AudioChunk,
    DataMessage,
    MsgType,
    ParticipantEvent,
    ReturnAudioFlush,
    decode,
)
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
            b"participant",
        ):
            self._socket.setsockopt(zmq.SUBSCRIBE, prefix)
        self._audio_callbacks: list[AudioCallback] = []
        self._data_callbacks: list[DataCallback] = []
        self._flush_callbacks: list[FlushCallback] = []
        self._departures: dict[tuple[str, int, str], asyncio.Event] = {}
        self._running = True

    def on_audio(self, callback: AudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_data(self, callback: DataCallback) -> None:
        self._data_callbacks.append(callback)

    def on_flush(self, callback: FlushCallback) -> None:
        self._flush_callbacks.append(callback)

    async def run(self) -> None:
        while self._running:
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
            elif type_id == MsgType.PARTICIPANT_EVENT and not message.joined:
                self._departure_event(message).set()

    def stop(self) -> None:
        """Stop after the active callback and release departure waiters."""
        self._running = False
        for departure in self._departures.values():
            departure.set()

    async def wait_for_departure(self, event: ParticipantEvent) -> None:
        """Wait until return traffic published before *event* has been handled."""
        key = (event.participant_id, event.pts_us, event.connector_id)
        departure = self._departure_event(event)
        try:
            await departure.wait()
        finally:
            if self._departures.get(key) is departure:
                self._departures.pop(key, None)

    def _departure_event(self, event: ParticipantEvent) -> asyncio.Event:
        key = (event.participant_id, event.pts_us, event.connector_id)
        return self._departures.setdefault(key, asyncio.Event())

    def close(self) -> None:
        self._socket.close(linger=0)
