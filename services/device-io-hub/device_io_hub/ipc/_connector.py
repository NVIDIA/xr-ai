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

import asyncio
import uuid
from collections import defaultdict
from typing import Any, Awaitable, Callable

import zmq
import zmq.asyncio
from loguru import logger

from xr_ai_hub import (AudioChunk, ConnectorRegistration, ControlMessage, DataMessage,
                       FrameSignal, MsgType, ParticipantEvent, PixelFormat,
                       ReturnAudioFlush, ShmRingBuffer, decode, encode)

ReturnAudioCallback      = Callable[[AudioChunk],        Awaitable[None]]
ReturnDataCallback       = Callable[[DataMessage],       Awaitable[None]]
ReturnAudioFlushCallback = Callable[[ReturnAudioFlush],  Awaitable[None]]

_DEFAULT_NUM_SLOTS       = 16
_DEFAULT_MAX_FRAME_BYTES = 12_441_600  # 4K NV12
_DEFAULT_REGISTRATION_TIMEOUT_S = 3.0
_DEFAULT_REGISTRATION_ATTEMPTS  = 3
_REGISTRATION_RESEND_INTERVAL_S = 0.25
_CONNECTOR_REGISTER_ACK_TOPIC = "_connector.registration_ack"


class _ConnectorRegistrationError(RuntimeError):
    """The hub did not acknowledge a usable shared-memory registration."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


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
        self._connector_id  = connector_id or uuid.uuid4().hex
        self._shm_base_name = shm_name or f"xr_conn_{self._connector_id[:8]}"
        self._shm_name      = ""
        self._num_slots = num_slots
        self._max_frame_bytes = max_frame_bytes
        self._ring: ShmRingBuffer | None = None
        self._ring_generation = 0
        self._registered = False

        ctx = zmq.asyncio.Context.instance()

        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)

        self._sub: zmq.asyncio.Socket = ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)
        # Subscribe during construction, before expensive transport startup, so
        # the PUB/SUB route is established by the time registration needs an ACK.
        self._sub.setsockopt(
            zmq.SUBSCRIBE,
            f"connector.{self._connector_id}.".encode(),
        )
        # Participant return subscriptions are added dynamically on join.

        self._seq: dict[tuple[str, str], int] = defaultdict(int)

        self._return_audio_cbs:       list[ReturnAudioCallback]      = []
        self._return_data_cbs:        list[ReturnDataCallback]       = []
        self._return_audio_flush_cbs: list[ReturnAudioFlushCallback] = []
        self._running = False

    # ── registration ─────────────────────────────────────────────────────────

    async def register(self) -> None:
        """
        Create this connector's ring and register it with the hub.

        The method returns only after the hub has attached and validated the
        ring. Missing shared memory causes bounded recreation with a fresh
        name; incompatible rings and acknowledgement timeouts fail startup.
        """
        timeout = _DEFAULT_REGISTRATION_TIMEOUT_S
        max_attempts = _DEFAULT_REGISTRATION_ATTEMPTS

        for attempt in range(1, max_attempts + 1):
            if self._ring is None:
                self._create_ring()
            reg = ConnectorRegistration(
                connector_id=self._connector_id,
                shm_name=self._shm_name,
            )
            ack = await self._request_registration(reg, timeout)
            if ack["success"]:
                self._registered = True
                logger.info(
                    "Connector {} registration acknowledged (shm={})",
                    self._connector_id,
                    self._shm_name,
                )
                return
            if ack["error_code"] == "shm_not_found" and attempt < max_attempts:
                logger.warning(
                    "Hub could not attach connector {} shared memory {}; "
                    "recreating ring ({}/{})",
                    self._connector_id,
                    self._shm_name,
                    attempt,
                    max_attempts,
                )
                self._destroy_ring()
                continue
            self._registered = False
            raise _ConnectorRegistrationError(
                str(ack["error_code"] or "registration_failed"),
                str(ack["error_message"] or "hub rejected shared-memory registration"),
            )

    def _create_ring(self) -> None:
        """Create a connector-owned ring immediately before registration."""
        self._ring_generation += 1
        self._shm_name = (
            self._shm_base_name
            if self._ring_generation == 1
            else f"{self._shm_base_name}_{uuid.uuid4().hex[:8]}"
        )
        try:
            self._ring = ShmRingBuffer(
                name=self._shm_name,
                num_slots=self._num_slots,
                max_frame_bytes=self._max_frame_bytes,
                create=True,
            )
        except Exception as exc:
            raise _ConnectorRegistrationError(
                "shm_create_failed",
                f"could not create shared memory {self._shm_name!r}: {exc}",
            ) from exc

    async def _request_registration(
        self,
        reg: ConnectorRegistration,
        timeout: float,
    ) -> dict[str, Any]:
        """Send until a correlated ACK arrives or the bounded timeout expires."""
        async def exchange() -> dict[str, Any]:
            while True:
                await self._push.send(encode(MsgType.CONNECTOR_REGISTER, reg))
                try:
                    _topic, raw = await asyncio.wait_for(
                        self._sub.recv_multipart(),
                        timeout=_REGISTRATION_RESEND_INTERVAL_S,
                    )
                except TimeoutError:
                    continue
                type_id, ack = decode(raw)
                if type_id == MsgType.CONTROL and ack.topic == _CONNECTOR_REGISTER_ACK_TOPIC:
                    payload = ack.payload
                    if (payload.get("connector_id"), payload.get("shm_name")) == (
                        self._connector_id,
                        reg.shm_name,
                    ):
                        return payload

        try:
            return await asyncio.wait_for(exchange(), timeout=timeout)
        except TimeoutError:
            self._registered = False
            raise _ConnectorRegistrationError(
                "registration_timeout",
                f"hub did not acknowledge shared memory {reg.shm_name!r} within {timeout:g}s",
            ) from None

    def _require_registered_ring(self) -> ShmRingBuffer:
        if not self._registered or self._ring is None:
            raise _ConnectorRegistrationError(
                "connector_not_registered",
                "media cannot be accepted before shared-memory registration succeeds",
            )
        return self._ring

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
        ring = self._require_registered_ring()
        key = (participant_id, track_id)
        self._seq[key] += 1
        seq  = self._seq[key]
        data_size = data.nbytes if isinstance(data, memoryview) else len(data)
        slot = ring.write_frame(data, width, height, fmt, pts_us, seq)
        sig  = FrameSignal(
            slot=slot, seq=seq, pts_us=pts_us,
            width=width, height=height, fmt=fmt, data_sz=data_size,
            participant_id=participant_id, track_id=track_id,
        )
        await self._push.send(encode(MsgType.FRAME_SIGNAL, sig))

    async def push_audio(self, chunk: AudioChunk) -> None:
        self._require_registered_ring()
        await self._push.send(encode(MsgType.AUDIO_CHUNK, chunk))

    async def push_data(self, msg: DataMessage) -> None:
        self._require_registered_ring()
        await self._push.send(encode(MsgType.DATA_MESSAGE, msg))

    async def send_control(self, msg: ControlMessage) -> None:
        self._require_registered_ring()
        await self._push.send(encode(MsgType.CONTROL, msg))

    # ── participant lifecycle ─────────────────────────────────────────────────

    async def notify_participant_joined(self, participant_id: str, pts_us: int = 0) -> None:
        """
        Call when a LiveKit participant connects to the room.

        Subscribes to return traffic for this participant and notifies the hub.
        The hub uses the embedded connector_id to maintain its participant →
        connector mapping.
        """
        self._require_registered_ring()
        # Trailing "." terminates the pid segment so a subscription for `alice`
        # does not byte-prefix-match a topic addressed to `alice2`. The hub
        # publishes return topics with the same trailing delimiter (see
        # `_hub.py`); the processor subscription path guards the inbound topics
        # identically (`_prefixes` in `xr_ai_hub._processor`).
        self._sub.setsockopt(zmq.SUBSCRIBE, f"return_audio.{participant_id}.".encode())
        self._sub.setsockopt(zmq.SUBSCRIBE, f"return_audio_flush.{participant_id}.".encode())
        self._sub.setsockopt(zmq.SUBSCRIBE, f"return_data.{participant_id}.".encode())
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
        self._require_registered_ring()
        self._sub.setsockopt(zmq.UNSUBSCRIBE, f"return_audio.{participant_id}.".encode())
        self._sub.setsockopt(zmq.UNSUBSCRIBE, f"return_audio_flush.{participant_id}.".encode())
        self._sub.setsockopt(zmq.UNSUBSCRIBE, f"return_data.{participant_id}.".encode())
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
                elif type_id == MsgType.CONTROL and msg.topic == _CONNECTOR_REGISTER_ACK_TOPIC:
                    logger.debug("Ignoring stale connector registration ACK")
                else:
                    logger.debug("Connector: unhandled return type {}", type_id)
            except Exception:
                logger.exception("Error dispatching return message")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def _destroy_ring(self) -> None:
        ring = self._ring
        self._ring = None
        self._registered = False
        if ring is None:
            return
        ring.close()
        ring.unlink()

    def close(self) -> None:
        """Close sockets and release the ring buffer. Unlinks the shm segment."""
        self._push.close(linger=0)
        self._sub.close(linger=0)
        self._destroy_ring()
