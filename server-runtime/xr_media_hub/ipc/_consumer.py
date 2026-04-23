"""
Consumer-side IPC endpoint (subscriber).

Connects to the hub's PUB socket and receives real-time audio, data, and
control messages. Video chunk queries (MP4, frame sets) are left to the
application layer — this module is transport-only.

Multiple consumers can connect to the same hub simultaneously.

Isolation: return_audio.* and return_data.* topics are connector-only and
are excluded from the default subscription. Subscribing to those topics from
a consumer is unsupported and would break the participant isolation contract.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from ._codec import decode
from ._types import AudioChunk, ControlMessage, DataMessage, MsgType, ParticipantEvent

log = logging.getLogger(__name__)

AudioCallback       = Callable[[AudioChunk],       Awaitable[None]]
DataCallback        = Callable[[DataMessage],      Awaitable[None]]
ParticipantCallback = Callable[[ParticipantEvent], Awaitable[None]]
ControlCallback     = Callable[[ControlMessage],   Awaitable[None]]


class ConsumerEndpoint:
    """
    Consumer-side IPC endpoint.

    Supports subscribing at any granularity using ZMQ prefix matching.
    Topics follow the format "<type>.<participant_id>.<track_or_topic>":

        ep = ConsumerEndpoint(sub_addr="ipc:///tmp/xr_hub_pub")
        # all audio from all participants:
        ep.subscribe_topic("audio")
        # all audio from one participant:
        ep.subscribe_topic("audio.alice")
        # one specific track from one participant:
        ep.subscribe_topic("audio.alice.TR_mic_001")
        # all data channels from all participants:
        ep.subscribe_topic("data")
        # one participant's data channel by topic:
        ep.subscribe_topic("data.alice.chat")

    Pass topics=None (default) to subscribe to everything.
    """

    # Topics a consumer subscribes to by default. Deliberately excludes
    # return_audio.* and return_data.* which are connector-only channels.
    _DEFAULT_TOPICS: tuple[bytes, ...] = (b"audio", b"data", b"participant", b"control")

    def __init__(self, sub_addr: str, topics: list[str | bytes] | None = None) -> None:
        ctx       = zmq.asyncio.Context.instance()
        self._sub: zmq.asyncio.Socket = ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)

        for t in (topics if topics is not None else self._DEFAULT_TOPICS):
            self.subscribe_topic(t)

        self._audio_cbs:       list[AudioCallback]       = []
        self._data_cbs:        list[DataCallback]        = []
        self._participant_cbs: list[ParticipantCallback] = []
        self._control_cbs:     list[ControlCallback]     = []
        self._running = False

    def subscribe_topic(self, topic: str | bytes) -> None:
        t = topic.encode() if isinstance(topic, str) else topic
        if t.startswith((b"return_audio", b"return_data")):
            log.warning(
                "ConsumerEndpoint.subscribe_topic(%r): return_* topics are "
                "connector-only — subscribing here breaks participant isolation. "
                "Use HubEndpoint.send_return_audio/send_return_data instead.",
                topic,
            )
        self._sub.setsockopt(zmq.SUBSCRIBE, t)

    def on_audio(self,       cb: AudioCallback)       -> None: self._audio_cbs.append(cb)
    def on_data(self,        cb: DataCallback)        -> None: self._data_cbs.append(cb)
    def on_participant(self, cb: ParticipantCallback) -> None: self._participant_cbs.append(cb)
    def on_control(self,     cb: ControlCallback)     -> None: self._control_cbs.append(cb)

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
        elif type_id == MsgType.CONTROL:
            for cb in self._control_cbs:
                await cb(msg)
        else:
            log.debug("Unhandled message type %d on consumer", type_id)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._sub.close(linger=0)
