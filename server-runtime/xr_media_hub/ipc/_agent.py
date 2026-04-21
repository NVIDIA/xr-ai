"""
Agent-side IPC endpoint (subscriber + publisher).

Connects to the hub's PUB socket to receive real-time audio, data, participant,
and control messages. Also connects a PUSH socket so the agent can send
RETURN_DATA and RETURN_AUDIO messages back through the hub to the originating
client.

A single AgentEndpoint instance can serve all participants simultaneously —
participant routing is handled by the hub once the agent pushes a return message
with the correct participant_id.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from ._codec import decode, encode
from ._types import AudioChunk, DataMessage, MsgType, ParticipantEvent

log = logging.getLogger(__name__)

AudioCallback       = Callable[[AudioChunk],       Awaitable[None]]
DataCallback        = Callable[[DataMessage],      Awaitable[None]]
ParticipantCallback = Callable[[ParticipantEvent], Awaitable[None]]

_DEFAULT_TOPICS: tuple[bytes, ...] = (b"audio", b"data", b"participant", b"control")


class AgentEndpoint:
    """
    Agent-side IPC endpoint.

    Subscribes to the hub's PUB socket and pushes return traffic back to the
    hub's PULL socket. Pattern mirrors ConsumerEndpoint but adds a PUSH socket
    for the return path.

        ep = AgentEndpoint(
            sub_addr="ipc:///tmp/xr_hub_pub",
            push_addr="ipc:///tmp/xr_hub_in",
        )
        ep.on_audio(my_audio_handler)
        ep.on_data(my_data_handler)
        ep.on_participant(my_participant_handler)
        await ep.run()
    """

    def __init__(self, sub_addr: str, push_addr: str) -> None:
        ctx = zmq.asyncio.Context.instance()

        self._sub: zmq.asyncio.Socket = ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)

        for t in _DEFAULT_TOPICS:
            self._sub.setsockopt(zmq.SUBSCRIBE, t)

        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)

        self._audio_cbs:       list[AudioCallback]       = []
        self._data_cbs:        list[DataCallback]        = []
        self._participant_cbs: list[ParticipantCallback] = []
        self._running = False

    def on_audio(self,       cb: AudioCallback)       -> None: self._audio_cbs.append(cb)
    def on_data(self,        cb: DataCallback)        -> None: self._data_cbs.append(cb)
    def on_participant(self, cb: ParticipantCallback) -> None: self._participant_cbs.append(cb)

    async def send_return_data(self, msg: DataMessage) -> None:
        await self._push.send(encode(MsgType.RETURN_DATA, msg))

    async def send_return_audio(self, chunk: AudioChunk) -> None:
        await self._push.send(encode(MsgType.RETURN_AUDIO, chunk))

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
        elif type_id == MsgType.DATA_MESSAGE:
            for cb in self._data_cbs:
                await cb(msg)
        elif type_id == MsgType.PARTICIPANT_EVENT:
            for cb in self._participant_cbs:
                await cb(msg)
        else:
            log.debug("Unhandled message type %d on agent endpoint", type_id)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._sub.close(linger=0)
        self._push.close(linger=0)
