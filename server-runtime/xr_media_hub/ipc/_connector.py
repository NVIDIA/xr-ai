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
from ._types import AudioChunk, ControlMessage, DataMessage, FrameSignal, MsgType, PixelFormat

log = logging.getLogger(__name__)

ReturnAudioCallback = Callable[[AudioChunk],  Awaitable[None]]
ReturnDataCallback  = Callable[[DataMessage], Awaitable[None]]


class ConnectorEndpoint:
    """
    Producer endpoint for the LiveKit connector process.

    Supports multiple LiveKit participants, each with multiple video, audio,
    and data tracks. Tracks are addressed by (participant_id, track_id) —
    matching LiveKit's participant identity and track SID.

    Usage
    -----
    ep = ConnectorEndpoint(shm_name="xr_hub_frames", push_addr="ipc:///tmp/xr_hub_in")

    # Two participants each with their own camera and mic:
    await ep.push_frame(data, width=1920, height=1080, fmt=PixelFormat.NV12,
                        pts_us=t, participant_id="alice", track_id="TR_cam_001")
    await ep.push_audio(AudioChunk(..., participant_id="alice", track_id="TR_mic_001"))
    await ep.push_audio(AudioChunk(..., participant_id="bob",   track_id="TR_mic_002"))

    # Data channel message from a participant:
    await ep.push_data(DataMessage(participant_id="alice", topic="chat", pts_us=t, data=b"hi"))
    ep.close()
    """

    def __init__(
        self,
        shm_name:  str,
        push_addr: str,
        sub_addr:  str,
        participant_ids: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        shm_name        : Shared-memory segment name (hub creates it first).
        push_addr       : Hub's PULL address — connector connects and PUSHes here.
        sub_addr        : Hub's PUB address  — connector subscribes for return traffic.
        participant_ids : If given, subscribe only to return traffic for these
                          participant IDs. Pass None to receive return traffic for
                          all participants (useful for a single-client connector).
        """
        self._ring = ShmRingBuffer(name=shm_name, create=False)
        ctx        = zmq.asyncio.Context.instance()

        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)

        self._sub: zmq.asyncio.Socket = ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)
        if participant_ids is None:
            # Receive all return traffic.
            self._sub.setsockopt(zmq.SUBSCRIBE, b"return_audio")
            self._sub.setsockopt(zmq.SUBSCRIBE, b"return_data")
        else:
            for pid in participant_ids:
                self._sub.setsockopt(zmq.SUBSCRIBE, f"return_audio.{pid}".encode())
                self._sub.setsockopt(zmq.SUBSCRIBE, f"return_data.{pid}".encode())

        # Sequence counters keyed by (participant_id, track_id).
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
