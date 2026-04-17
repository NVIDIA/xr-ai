"""
Hub-side IPC endpoint (server).

Creates and owns the shared-memory ring buffer. Receives frame signals and
audio from the connector via ZMQ PULL, dispatches to registered async
callbacks, then broadcasts audio/control to downstream consumers via ZMQ PUB.

                  ┌──────────────────────────────────────┐
  connector ──PUSH──► PULL   HubEndpoint   PUB ──SUB──► consumers
                  │    ↓ dispatch                        │
                  │  on_frame / on_audio / on_control    │
                  └──────────────────────────────────────┘

Frame callbacks receive a SlotView (zero-copy memoryview). The slot is released
automatically after ALL frame callbacks return — do not hold the view beyond
the callback boundary.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from ._codec import decode, encode
from ._shm import ShmRingBuffer, SlotView
from ._types import AudioChunk, ControlMessage, DataMessage, MsgType

log = logging.getLogger(__name__)

FrameCallback   = Callable[[SlotView],       Awaitable[None]]
AudioCallback   = Callable[[AudioChunk],     Awaitable[None]]
DataCallback    = Callable[[DataMessage],    Awaitable[None]]
ControlCallback = Callable[[ControlMessage], Awaitable[None]]

# Topic prefixes for ZMQ PUB/SUB.
# The hub publishes to "<type>.<participant_id>.<track_or_topic>" so consumers
# can subscribe at any granularity using ZMQ prefix matching:
#
#   b"audio"                    — all audio, all participants
#   b"audio.alice"              — all of alice's audio tracks
#   b"audio.alice.TR_mic_001"   — alice's specific mic track
#   b"data"                     — all data channels, all participants
#   b"data.alice"               — all of alice's data channels
#   b"data.alice.chat"          — alice's "chat" topic only
#   b"control"                  — hub control messages (no participant/track)
TOPIC_AUDIO   = b"audio"
TOPIC_DATA    = b"data"
TOPIC_CONTROL = b"control"


class HubEndpoint:
    """
    Hub-side IPC endpoint.

    Parameters
    ----------
    shm_name        : Shared-memory segment name (e.g. "xr_hub_frames").
    num_slots       : Ring buffer slot count. 10 slots @ 1080p NV12 ≈ 30 MB.
    max_frame_bytes : Maximum bytes per slot. 4K NV12 = 12_441_600.
    pull_addr       : ZMQ address the hub binds for connector PUSH traffic.
    pub_addr        : ZMQ address the hub binds for consumer SUB traffic.
    """

    def __init__(
        self,
        shm_name:        str,
        num_slots:       int,
        max_frame_bytes: int,
        pull_addr:       str,
        pub_addr:        str,
    ) -> None:
        self._ring = ShmRingBuffer(
            name=shm_name,
            num_slots=num_slots,
            max_frame_bytes=max_frame_bytes,
            create=True,
        )
        ctx = zmq.asyncio.Context.instance()

        self._pull: zmq.asyncio.Socket = ctx.socket(zmq.PULL)
        self._pull.bind(pull_addr)

        self._pub: zmq.asyncio.Socket = ctx.socket(zmq.PUB)
        self._pub.bind(pub_addr)

        self._frame_cbs:   list[FrameCallback]   = []
        self._audio_cbs:   list[AudioCallback]   = []
        self._data_cbs:    list[DataCallback]    = []
        self._control_cbs: list[ControlCallback] = []
        self._running = False

    # ── callback registration ─────────────────────────────────────────────────

    def on_frame(self,   cb: FrameCallback)   -> None: self._frame_cbs.append(cb)
    def on_audio(self,   cb: AudioCallback)   -> None: self._audio_cbs.append(cb)
    def on_data(self,    cb: DataCallback)    -> None: self._data_cbs.append(cb)
    def on_control(self, cb: ControlCallback) -> None: self._control_cbs.append(cb)

    # ── outbound broadcast (hub → consumers) ─────────────────────────────────

    async def broadcast(self, topic: bytes | str, type_id: int, msg) -> None:
        """Send an arbitrary message to all consumers subscribed to topic."""
        t = topic.encode() if isinstance(topic, str) else topic
        await self._pub.send_multipart([t, encode(type_id, msg)])

    # ── receive loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Receive and dispatch messages until stop() is called."""
        self._running = True
        while self._running:
            try:
                raw = await self._pull.recv()
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
        if type_id == MsgType.FRAME_SIGNAL:
            view = self._ring.read_slot(msg)
            try:
                for cb in self._frame_cbs:
                    await cb(view)
            finally:
                self._ring.release_slot(msg.slot)

        elif type_id == MsgType.AUDIO_CHUNK:
            for cb in self._audio_cbs:
                await cb(msg)
            topic = f"audio.{msg.participant_id}.{msg.track_id}".encode()
            await self._pub.send_multipart([topic, encode(MsgType.AUDIO_CHUNK, msg)])

        elif type_id == MsgType.DATA_MESSAGE:
            for cb in self._data_cbs:
                await cb(msg)
            topic = f"data.{msg.participant_id}.{msg.topic}".encode()
            await self._pub.send_multipart([topic, encode(MsgType.DATA_MESSAGE, msg)])

        elif type_id == MsgType.CONTROL:
            for cb in self._control_cbs:
                await cb(msg)
            await self._pub.send_multipart([TOPIC_CONTROL, encode(MsgType.CONTROL, msg)])

        else:
            log.warning("Unknown message type %d — ignored", type_id)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._pull.close(linger=0)
        self._pub.close(linger=0)
        self._ring.close()

    def unlink(self) -> None:
        """Remove the shared-memory segment. Call once on clean shutdown."""
        self._ring.unlink()
