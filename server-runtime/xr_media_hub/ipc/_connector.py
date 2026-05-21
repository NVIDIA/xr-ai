# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Connector-side IPC endpoint (producer + receiver).

Each connector creates and owns its own shared-memory ring buffer, then
registers with the hub by sending a ConnectorRegistration message. From that
point the hub can read frames from this connector's buffer regardless of how
many other connectors are connected.

Participants are dynamic: call notify_participant_joined() / left() as
LiveKit room events arrive.

                        ┌─────────────────┐
  LiveKit inbound  ──►  │   Connector     │ ──PUSH──► Hub
  LiveKit outbound ◄──  │   Endpoint      │ ◄──SUB──  Hub
                        └─────────────────┘

The connector process only needs: pyzmq, msgpack (no CUDA, no GPU deps).
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Awaitable, Callable

import zmq
import zmq.asyncio
from loguru import logger

from xr_ai_agent import (AudioChunk, ConnectorRegistration, ControlMessage,
                         DataMessage, FrameSignal, MsgType, ParticipantEvent, PixelFormat,
                         ReturnAudioFlush, ShmRingBuffer, decode, encode)

ReturnAudioCallback      = Callable[[AudioChunk],        Awaitable[None]]
ReturnDataCallback       = Callable[[DataMessage],       Awaitable[None]]
ReturnAudioFlushCallback = Callable[[ReturnAudioFlush],  Awaitable[None]]

_DEFAULT_NUM_SLOTS       = 16
_DEFAULT_MAX_FRAME_BYTES = 12_441_600  # 4K NV12


class ConnectorEndpoint:
    """
    Producer + receiver endpoint for the LiveKit connector process.

    Each instance owns a dedicated ring buffer so multiple connectors can
    write frames concurrently without any locking. The hub is agnostic to
    how many connectors exist or how many participants each carries.

    Usage
    -----
    ep = ConnectorEndpoint(push_addr="ipc:///tmp/xr_hub_in",
                           sub_addr="ipc:///tmp/xr_hub_pub")
    ep.on_return_audio(send_to_livekit)
    await ep.register()                          # announce to hub

    await ep.notify_participant_joined("alice", pts_us=t)
    await ep.push_frame(data, 1920, 1080, PixelFormat.NV12, t, "alice", "TR_cam_001")
    await ep.push_audio(AudioChunk(..., participant_id="alice", track_id="TR_mic_001"))
    await ep.push_data(DataMessage(participant_id="alice", topic="chat", pts_us=t, data=b"hi"))
    await ep.notify_participant_left("alice", pts_us=t)

    ep.stop(); ep.close()
    """

    def __init__(
        self,
        push_addr:       str,
        sub_addr:        str,
        connector_id:    str = "",
        shm_name:        str = "",
        num_slots:       int = _DEFAULT_NUM_SLOTS,
        max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        """
        Parameters
        ----------
        push_addr       : Hub's PULL address — connector connects and PUSHes here.
        sub_addr        : Hub's PUB address  — connector subscribes for return traffic.
        connector_id    : Unique ID for this connector. Defaults to a UUID.
        shm_name        : Shared-memory segment name. Defaults to xr_conn_<connector_id>.
        num_slots       : Ring buffer slot count (default 16).
        max_frame_bytes : Max bytes per slot (default 4K NV12 = 12 441 600).
        """
        self._connector_id = connector_id or uuid.uuid4().hex
        self._shm_name     = shm_name or f"xr_conn_{self._connector_id[:8]}"

        # Each connector owns and creates its own ring buffer.
        self._ring = ShmRingBuffer(
            name=self._shm_name,
            num_slots=num_slots,
            max_frame_bytes=max_frame_bytes,
            create=True,
        )

        ctx = zmq.asyncio.Context.instance()

        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)

        self._sub: zmq.asyncio.Socket = ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)
        # No subscriptions yet — added dynamically as participants join.

        self._seq: dict[tuple[str, str], int] = defaultdict(int)

        self._return_audio_cbs:       list[ReturnAudioCallback]      = []
        self._return_data_cbs:        list[ReturnDataCallback]       = []
        self._return_audio_flush_cbs: list[ReturnAudioFlushCallback] = []
        self._running = False

    # ── registration ─────────────────────────────────────────────────────────

    async def register(self) -> None:
        """
        Announce this connector to the hub.

        Must be called once before pushing any media. The hub opens the
        ring buffer upon receiving the registration message.
        """
        reg = ConnectorRegistration(connector_id=self._connector_id, shm_name=self._shm_name)
        await self._push.send(encode(MsgType.CONNECTOR_REGISTER, reg))

    # ── inbound media ─────────────────────────────────────────────────────────

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
        Write a decoded CPU frame into this connector's ring buffer and signal
        the hub. Raises RuntimeError if all slots are occupied — caller should
        drop the frame and log a warning.
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

        Subscribes to return traffic for this participant and notifies the hub.
        The hub uses the embedded connector_id to maintain its participant →
        connector mapping.
        """
        self._sub.setsockopt(zmq.SUBSCRIBE, f"return_audio.{participant_id}".encode())
        self._sub.setsockopt(zmq.SUBSCRIBE, f"return_audio_flush.{participant_id}".encode())
        self._sub.setsockopt(zmq.SUBSCRIBE, f"return_data.{participant_id}".encode())
        event = ParticipantEvent(
            participant_id=participant_id, joined=True,
            pts_us=pts_us, connector_id=self._connector_id,
        )
        await self._push.send(encode(MsgType.PARTICIPANT_EVENT, event))

    async def notify_participant_left(self, participant_id: str, pts_us: int = 0) -> None:
        """
        Call when a LiveKit participant disconnects from the room.

        Unsubscribes from return traffic, cleans up sequence counters, and
        notifies the hub.
        """
        self._sub.setsockopt(zmq.UNSUBSCRIBE, f"return_audio.{participant_id}".encode())
        self._sub.setsockopt(zmq.UNSUBSCRIBE, f"return_audio_flush.{participant_id}".encode())
        self._sub.setsockopt(zmq.UNSUBSCRIBE, f"return_data.{participant_id}".encode())
        stale = [k for k in self._seq if k[0] == participant_id]
        for k in stale:
            del self._seq[k]
        event = ParticipantEvent(
            participant_id=participant_id, joined=False,
            pts_us=pts_us, connector_id=self._connector_id,
        )
        await self._push.send(encode(MsgType.PARTICIPANT_EVENT, event))

    # ── return-path callbacks ─────────────────────────────────────────────────

    def on_return_audio(self, cb: ReturnAudioCallback) -> None:
        self._return_audio_cbs.append(cb)

    def on_return_data(self, cb: ReturnDataCallback) -> None:
        self._return_data_cbs.append(cb)

    def on_return_audio_flush(self, cb: ReturnAudioFlushCallback) -> None:
        self._return_audio_flush_cbs.append(cb)

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
                logger.error("ZMQ recv error: {}", exc)
                continue
            try:
                type_id, msg = decode(raw)
                if type_id == MsgType.RETURN_AUDIO:
                    for cb in self._return_audio_cbs:
                        await cb(msg)
                elif type_id == MsgType.RETURN_DATA:
                    for cb in self._return_data_cbs:
                        await cb(msg)
                elif type_id == MsgType.RETURN_AUDIO_FLUSH:
                    for cb in self._return_audio_flush_cbs:
                        await cb(msg)
                else:
                    logger.debug("Connector: unhandled return type {}", type_id)
            except Exception:
                logger.exception("Error dispatching return message")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        """Close sockets and release the ring buffer. Unlinks the shm segment."""
        self._push.close(linger=0)
        self._sub.close(linger=0)
        self._ring.close()
        self._ring.unlink()
