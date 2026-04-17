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

    Supports multiple video, audio, and data tracks simultaneously.
    Each track is identified by a track_id string (e.g. "camera_main",
    "mic_0", "data_chat"). Sequence numbers are tracked per video track.

    Usage
    -----
    ep = ConnectorEndpoint(shm_name="xr_hub_frames", push_addr="ipc:///tmp/xr_hub_in")
    await ep.push_frame(data, width=1920, height=1080, fmt=PixelFormat.NV12,
                        pts_us=t, track_id="camera_main")
    await ep.push_audio(AudioChunk(..., track_id="mic_0"))
    await ep.push_data(DataMessage(track_id="data_chat", pts_us=t, data=b"hello"))
    ep.close()
    """

    def __init__(self, shm_name: str, push_addr: str) -> None:
        # Hub must have created the shm before connector opens it.
        self._ring = ShmRingBuffer(name=shm_name, create=False)
        ctx        = zmq.asyncio.Context.instance()
        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)
        self._seq: dict[str, int] = defaultdict(int)  # per-track sequence counters

    async def push_frame(
        self,
        data:     bytes | memoryview,
        width:    int,
        height:   int,
        fmt:      PixelFormat,
        pts_us:   int,
        track_id: str = "default",
    ) -> None:
        """
        Write a decoded CPU frame into the ring buffer and signal the hub.

        Raises RuntimeError (propagated from ShmRingBuffer) if all slots are
        occupied — caller should drop the frame and log a warning.
        """
        self._seq[track_id] += 1
        seq  = self._seq[track_id]
        slot = self._ring.write_frame(data, width, height, fmt, pts_us, seq)
        sig  = FrameSignal(
            slot=slot, seq=seq, pts_us=pts_us,
            width=width, height=height, fmt=fmt,
            data_sz=len(data), track_id=track_id,
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
