"""
Connector-side IPC endpoint (producer).

Opens the shared-memory ring buffer created by HubEndpoint, then pushes
frame signals, audio chunks, and control messages to the hub via ZMQ PUSH.

The connector process only needs: pyzmq, msgpack (no CUDA, no GPU deps).
"""
from __future__ import annotations

import zmq
import zmq.asyncio

from ._codec import encode
from ._shm import ShmRingBuffer
from collections import defaultdict

from ._types import AudioChunk, ControlMessage, DataMessage, FrameSignal, MsgType, PixelFormat


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

    def __init__(self, shm_name: str, push_addr: str) -> None:
        # Hub must have created the shm before connector opens it.
        self._ring = ShmRingBuffer(name=shm_name, create=False)
        ctx        = zmq.asyncio.Context.instance()
        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)
        # Sequence counters keyed by (participant_id, track_id).
        self._seq: dict[tuple[str, str], int] = defaultdict(int)

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

    def close(self) -> None:
        self._push.close(linger=0)
        self._ring.close()
