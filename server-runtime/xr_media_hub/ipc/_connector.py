"""
Connector-side IPC endpoint (producer + receiver).

Opens the shared-memory ring buffer created by HubEndpoint, pushes inbound
media to the hub via ZMQ PUSH, and receives outbound media (return audio,
return data) from the hub via ZMQ SUB.

                        ┌─────────────────┐
  LiveKit inbound  ──►  │   Connector     │ ──PUSH──► Hub
  LiveKit outbound ◄──  │   Endpoint      │ ◄──SUB──  Hub
                        └─────────────────┘

The connector process only needs: pyzmq, msgpack (no CUDA, no GPU deps).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from ._codec import decode, encode
from ._shm import ShmRingBuffer
from ._types import AudioChunk, ControlMessage, DataMessage, FrameSignal, MsgType, ParticipantEvent, PixelFormat

log = logging.getLogger(__name__)

ReturnAudioCallback = Callable[[AudioChunk],  Awaitable[None]]
ReturnDataCallback  = Callable[[DataMessage], Awaitable[None]]


class ConnectorEndpoint:
    """
    Producer + receiver endpoint for the LiveKit connector process.

    Participants are dynamic: call notify_participant_joined() / left() as
    LiveKit room events arrive. Each call atomically updates the return-traffic
    SUB subscriptions and notifies the hub so agents can react.

    Usage
    -----
    ep = ConnectorEndpoint(shm_name="xr_hub_frames",
                           push_addr="ipc:///tmp/xr_hub_in",
                           sub_addr="ipc:///tmp/xr_hub_pub")
    ep.on_return_audio(send_to_livekit)

    # LiveKit participant-joined event fires:
    await ep.notify_participant_joined("alice", pts_us=t)

    await ep.push_frame(data, 1920, 1080, PixelFormat.NV12, t, "alice", "TR_cam_001")
    await ep.push_audio(AudioChunk(..., participant_id="alice", track_id="TR_mic_001"))
    await ep.push_data(DataMessage(participant_id="alice", topic="chat", pts_us=t, data=b"hi"))

    # LiveKit participant-left event fires:
    await ep.notify_participant_left("alice", pts_us=t)
    ep.close()
    """

    def __init__(self, shm_name: str, push_addr: str, sub_addr: str) -> None:
        self._ring = ShmRingBuffer(name=shm_name, create=False)
        ctx        = zmq.asyncio.Context.instance()

        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)

        self._sub: zmq.asyncio.Socket = ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)
        # No subscriptions yet — added dynamically as participants join.

        self._seq: dict[tuple[str, str], int] = defaultdict(int)

        self._return_audio_cbs: list[ReturnAudioCallback] = []
        self._return_data_cbs:  list[ReturnDataCallback]  = []
        self._running = False

    async def push_frame(
        self,
        data:           bytes | memoryview,
        width:          int,
        height:         int,
        fmt:            PixelFormat,
        pts_us:         int,
        participant_id: str = "default",
        track_id:       str = "default",
    ) -> None:
        """
        Write a decoded CPU frame into the ring buffer and signal the hub.

        Raises RuntimeError (propagated from ShmRingBuffer) if all slots are
        occupied — caller should drop the frame and log a warning.
        """
        key = (participant_id, track_id)
        self._seq[key] += 1
        seq  = self._seq[key]
        slot = self._ring.write_frame(data, width, height, fmt, pts_us, seq)
        sig  = FrameSignal(
            slot=slot, seq=seq, pts_us=pts_us,
            width=width, height=height, fmt=fmt, data_sz=len(data),
            participant_id=participant_id, track_id=track_id,
        )
        await self._push.send(encode(MsgType.FRAME_SIGNAL, sig))

    async def push_audio(self, chunk: AudioChunk) -> None:
        await self._push.send(encode(MsgType.AUDIO_CHUNK, chunk))

    async def push_data(self, msg: DataMessage) -> None:
        await self._push.send(encode(MsgType.DATA_MESSAGE, msg))

    async def send_control(self, msg: ControlMessage) -> None:
        await self._push.send(encode(MsgType.CONTROL, msg))

    # ── participant lifecycle ─────────────────────────────────────────────────

    async def notify_participant_joined(self, participant_id: str, pts_us: int = 0) -> None:
        """
        Call when a LiveKit participant connects to the room.

        Subscribes the connector to return traffic for this participant and
        notifies the hub so agents/consumers can react.
        """
        self._sub.setsockopt(zmq.SUBSCRIBE, f"return_audio.{participant_id}".encode())
        self._sub.setsockopt(zmq.SUBSCRIBE, f"return_data.{participant_id}".encode())
        event = ParticipantEvent(participant_id=participant_id, joined=True, pts_us=pts_us)
        await self._push.send(encode(MsgType.PARTICIPANT_EVENT, event))

    async def notify_participant_left(self, participant_id: str, pts_us: int = 0) -> None:
        """
        Call when a LiveKit participant disconnects from the room.

        Unsubscribes the connector from return traffic for this participant,
        cleans up per-track sequence counters, and notifies the hub.
        """
        self._sub.setsockopt(zmq.UNSUBSCRIBE, f"return_audio.{participant_id}".encode())
        self._sub.setsockopt(zmq.UNSUBSCRIBE, f"return_data.{participant_id}".encode())
        # Drop stale sequence counters for this participant.
        stale = [k for k in self._seq if k[0] == participant_id]
        for k in stale:
            del self._seq[k]
        event = ParticipantEvent(participant_id=participant_id, joined=False, pts_us=pts_us)
        await self._push.send(encode(MsgType.PARTICIPANT_EVENT, event))

    # ── return-path callbacks ─────────────────────────────────────────────────

    def on_return_audio(self, cb: ReturnAudioCallback) -> None:
        """Register a callback for agent/TTS audio to be sent back to the client."""
        self._return_audio_cbs.append(cb)

    def on_return_data(self, cb: ReturnDataCallback) -> None:
        """Register a callback for agent text/binary to be sent back to the client."""
        self._return_data_cbs.append(cb)

    # ── receive loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Receive return audio and data from the hub until stop() is called."""
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
                if type_id == MsgType.RETURN_AUDIO:
                    for cb in self._return_audio_cbs:
                        await cb(msg)
                elif type_id == MsgType.RETURN_DATA:
                    for cb in self._return_data_cbs:
                        await cb(msg)
                else:
                    log.debug("Connector: unhandled return type %d", type_id)
            except Exception:
                log.exception("Error dispatching return message")

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._push.close(linger=0)
        self._sub.close(linger=0)
        self._ring.close()
