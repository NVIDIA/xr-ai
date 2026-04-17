"""
Consumer-side IPC endpoint (subscriber).

Connects to the hub's PUB socket and receives real-time audio and control
messages. Video chunk queries (MP4, frame sets) are left to the application
layer — this module is transport-only.

Multiple consumers can connect to the same hub simultaneously.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from ._codec import decode
from ._types import AudioChunk, ControlMessage, MsgType

log = logging.getLogger(__name__)

AudioCallback   = Callable[[AudioChunk],     Awaitable[None]]
ControlCallback = Callable[[ControlMessage], Awaitable[None]]


class ConsumerEndpoint:
    """
    Consumer-side IPC endpoint.

    Parameters
    ----------
    sub_addr : ZMQ address of the hub's PUB socket.
    topics   : Topics to subscribe to. Pass None to subscribe to everything.
               Use the TOPIC_AUDIO / TOPIC_CONTROL constants from _hub for
               well-known topics.

    Usage
    -----
    ep = ConsumerEndpoint(sub_addr="ipc:///tmp/xr_hub_pub")
    ep.on_audio(my_audio_handler)
    ep.on_control(my_control_handler)
    await ep.run()
    """

    def __init__(self, sub_addr: str, topics: list[str | bytes] | None = None) -> None:
        ctx       = zmq.asyncio.Context.instance()
        self._sub: zmq.asyncio.Socket = ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)

        if topics is None:
            # Subscribe to everything.
            self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            for t in topics:
                self.subscribe_topic(t)

        self._audio_cbs:   list[AudioCallback]   = []
        self._control_cbs: list[ControlCallback] = []
        self._running = False

    def subscribe_topic(self, topic: str | bytes) -> None:
        t = topic.encode() if isinstance(topic, str) else topic
        self._sub.setsockopt(zmq.SUBSCRIBE, t)

    def on_audio(self,   cb: AudioCallback)   -> None: self._audio_cbs.append(cb)
    def on_control(self, cb: ControlCallback) -> None: self._control_cbs.append(cb)

    async def run(self) -> None:
        """Receive and dispatch messages until stop() is called."""
        self._running = True
        while self._running:
            try:
                _topic, raw = await self._sub.recv_multipart()
            except zmq.ZMQError as exc:
                if not self._running:
                    break
                log.error("ZMQ recv error: %s", exc)
                continue
            try:
                type_id, msg = decode(raw)
                await self._dispatch(type_id, msg)
            except Exception:
                log.exception("Error dispatching message")

    async def _dispatch(self, type_id: int, msg) -> None:
        if type_id == MsgType.AUDIO_CHUNK:
            for cb in self._audio_cbs:
                await cb(msg)
        elif type_id == MsgType.CONTROL:
            for cb in self._control_cbs:
                await cb(msg)
        else:
            log.debug("Unhandled message type %d on consumer", type_id)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._sub.close(linger=0)
