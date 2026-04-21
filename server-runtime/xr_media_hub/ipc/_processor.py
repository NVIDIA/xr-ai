"""
Processor-side IPC endpoint (subscriber + publisher).

Connects to the hub's PUB socket to receive real-time audio, data, and
participant events. Also connects a PUSH socket to send RETURN_DATA and
RETURN_AUDIO back through the hub to connected clients.

Works for any downstream processing workload — analytics, ML inference,
transcription, echo, recording — not just agentic pipelines.

A single ProcessorEndpoint instance serves all participants simultaneously.
It maintains its own connected-participant set so processors always know
who is present without having to track participant events manually.

    ep = ProcessorEndpoint(
        sub_addr="ipc:///tmp/xr_hub_pub",
        push_addr="ipc:///tmp/xr_hub_in",
    )
    ep.on_audio(my_audio_handler)
    ep.on_data(my_data_handler)
    await ep.run()
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


class ProcessorEndpoint:
    """
    Downstream IPC endpoint for data processors.

    Receives audio, data, and participant events from the hub. Maintains a
    live set of connected participants updated automatically as join/leave
    events arrive. Can send return data and audio back to clients via the hub.

        ep = ProcessorEndpoint(
            sub_addr="ipc:///tmp/xr_hub_pub",
            push_addr="ipc:///tmp/xr_hub_in",
        )
        ep.on_audio(handle_audio)
        ep.on_data(handle_data)
        ep.on_participant(handle_participant)  # optional — set is auto-maintained
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

        self._participants: set[str] = set()

        self._audio_cbs:       list[AudioCallback]       = []
        self._data_cbs:        list[DataCallback]        = []
        self._participant_cbs: list[ParticipantCallback] = []
        self._running = False

    # ── participant roster ────────────────────────────────────────────────────

    @property
    def connected_participants(self) -> frozenset[str]:
        """Participant IDs currently connected to the hub, auto-updated."""
        return frozenset(self._participants)

    # ── callback registration ─────────────────────────────────────────────────

    def on_audio(self,       cb: AudioCallback)       -> None: self._audio_cbs.append(cb)
    def on_data(self,        cb: DataCallback)        -> None: self._data_cbs.append(cb)
    def on_participant(self, cb: ParticipantCallback) -> None: self._participant_cbs.append(cb)

    # ── return path ───────────────────────────────────────────────────────────

    async def send_return_data(self, msg: DataMessage) -> None:
        await self._push.send(encode(MsgType.RETURN_DATA, msg))

    async def send_return_audio(self, chunk: AudioChunk) -> None:
        await self._push.send(encode(MsgType.RETURN_AUDIO, chunk))

    # ── receive loop ──────────────────────────────────────────────────────────

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
            if msg.joined:
                self._participants.add(msg.participant_id)
            else:
                self._participants.discard(msg.participant_id)
            for cb in self._participant_cbs:
                await cb(msg)
        else:
            log.debug("Unhandled message type %d on processor endpoint", type_id)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._sub.close(linger=0)
        self._push.close(linger=0)
