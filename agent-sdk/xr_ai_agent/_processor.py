"""
Processor-side IPC endpoint (subscriber + publisher).

Connects to the hub's PUB socket to receive real-time video signals, audio,
data, and participant events. Also connects a PUSH socket to send RETURN_DATA,
RETURN_AUDIO, and FRAME_REQUEST back to the hub.

Works for any downstream processing workload — analytics, ML inference,
transcription, echo, recording — not just agentic pipelines.

A single ProcessorEndpoint instance serves all participants simultaneously.
It maintains its own connected-participant set so processors always know
who is present without having to track participant events manually.

Video frame access is two-step:
  1. on_frame callback receives FrameSignal metadata (always, at full rate).
  2. Call await ep.request_frame(signal) to pull pixel data on demand.
     The hub serves from a small cache; returns None if the frame has expired.

    ep = ProcessorEndpoint(
        sub_addr="ipc:///tmp/xr_hub_pub",
        push_addr="ipc:///tmp/xr_hub_in",
    )
    ep.on_frame(handle_frame_signal)   # metadata — fires at full frame rate
    ep.on_audio(my_audio_handler)
    ep.on_data(my_data_handler)
    await ep.run()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from ._codec import decode, encode
from ._types import (AudioChunk, DataMessage, FrameData, FrameRequest,
                     FrameSignal, MsgType, ParticipantEvent)

log = logging.getLogger(__name__)

FrameSignalCallback = Callable[[FrameSignal], Awaitable[None]]
FrameDataCallback   = Callable[[FrameData],   Awaitable[None]]
AudioCallback       = Callable[[AudioChunk],       Awaitable[None]]
DataCallback        = Callable[[DataMessage],      Awaitable[None]]
ParticipantCallback = Callable[[ParticipantEvent], Awaitable[None]]

_DEFAULT_TOPICS: tuple[bytes, ...] = (
    b"video", b"video_data", b"audio", b"data", b"participant", b"control",
)

# Reserved topic for internal SDK status messages — not forwarded to app callbacks.
AGENT_STATUS_TOPIC = "_agent.status"

_FRAME_REQUEST_TIMEOUT = 1.0  # seconds before request_frame() gives up


class ProcessorEndpoint:
    """
    Downstream IPC endpoint for data processors.

    Receives video signals, audio, data, and participant events from the hub.
    Maintains a live set of connected participants updated automatically as
    join/leave events arrive. Can send return data, audio, and frame requests
    back to the hub via the PUSH socket.

        ep = ProcessorEndpoint(
            sub_addr="ipc:///tmp/xr_hub_pub",
            push_addr="ipc:///tmp/xr_hub_in",
        )
        ep.on_frame(handle_frame_signal)
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

        self._frame_cbs:       list[FrameSignalCallback] = []
        self._frame_data_cbs:  list[FrameDataCallback]   = []
        self._audio_cbs:       list[AudioCallback]       = []
        self._data_cbs:        list[DataCallback]        = []
        self._participant_cbs: list[ParticipantCallback] = []

        # Pending request_frame() calls keyed by (participant_id, track_id).
        # Each entry is a list of futures — all resolved when FRAME_DATA arrives.
        # Multiple concurrent requests for the same track share one FRAME_REQUEST.
        self._pending: dict[tuple[str, str], list[asyncio.Future[FrameData]]] = {}

        self._running = False

    # ── participant roster ────────────────────────────────────────────────────

    @property
    def connected_participants(self) -> frozenset[str]:
        """Participant IDs currently connected to the hub, auto-updated."""
        return frozenset(self._participants)

    # ── callback registration ─────────────────────────────────────────────────

    def on_frame(self,       cb: FrameSignalCallback) -> None: self._frame_cbs.append(cb)
    def on_frame_data(self,  cb: FrameDataCallback)   -> None: self._frame_data_cbs.append(cb)
    def on_audio(self,       cb: AudioCallback)       -> None: self._audio_cbs.append(cb)
    def on_data(self,        cb: DataCallback)        -> None: self._data_cbs.append(cb)
    def on_participant(self, cb: ParticipantCallback) -> None: self._participant_cbs.append(cb)

    # ── return path ───────────────────────────────────────────────────────────

    async def send_return_data(self, msg: DataMessage) -> None:
        await self._push.send(encode(MsgType.RETURN_DATA, msg))

    async def send_return_audio(self, chunk: AudioChunk) -> None:
        await self._push.send(encode(MsgType.RETURN_AUDIO, chunk))

    async def set_status(self, status: str,
                         participant_id: str | None = None) -> None:
        """
        Publish agent status to connected clients via the internal SDK channel.

        The status is delivered on the reserved LiveKit topic ``_agent.status``
        and is intercepted client-side by the StreamKit SDK — it never surfaces
        as a raw ``onDataReceived`` message.

        Parameters
        ----------
        status :
            Arbitrary status string.  Conventional values: ``"idle"``,
            ``"processing"``.
        participant_id :
            Target participant.  If *None*, broadcasts to every participant
            currently in ``connected_participants``.
        """
        payload = json.dumps({"status": status}).encode()
        pts_us  = int(time.time() * 1_000_000)
        targets = (
            [participant_id]
            if participant_id is not None
            else list(self._participants)
        )
        for pid in targets:
            await self.send_return_data(DataMessage(
                participant_id=pid,
                topic=AGENT_STATUS_TOPIC,
                pts_us=pts_us,
                data=payload,
            ))

    async def request_frame(self, signal: FrameSignal,
                            timeout: float = _FRAME_REQUEST_TIMEOUT) -> FrameData | None:
        """
        Request a pixel-data snapshot of the latest frame for this participant/track.

        The hub holds the most recent SHM slot and copies pixels only when a
        request arrives — no frame data is sent unless explicitly requested.

        Multiple concurrent calls for the same (participant, track) are coalesced:
        only one FRAME_REQUEST is sent and all callers receive the same response.

        Returns None if the hub has no frame for this track yet, or on timeout.
        """
        key = (signal.participant_id, signal.track_id)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[FrameData] = loop.create_future()

        if key in self._pending:
            # A request is already in-flight — piggyback on it.
            self._pending[key].append(fut)
        else:
            self._pending[key] = [fut]
            await self._push.send(encode(MsgType.FRAME_REQUEST, FrameRequest(
                participant_id=signal.participant_id,
                track_id=signal.track_id,
            )))

        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            waiters = self._pending.get(key, [])
            if fut in waiters:
                waiters.remove(fut)
            if not waiters:
                self._pending.pop(key, None)
            log.debug("request_frame timed out: participant=%s track=%s",
                      signal.participant_id, signal.track_id)
            return None

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
        if type_id == MsgType.FRAME_SIGNAL:
            for cb in self._frame_cbs:
                self._spawn(cb(msg))
        elif type_id == MsgType.FRAME_DATA:
            # Resolve pending request_frame() futures synchronously so they can
            # proceed as soon as the event loop next runs their awaiting coroutine.
            key = (msg.participant_id, msg.track_id)
            waiters = self._pending.pop(key, [])
            for fut in waiters:
                if not fut.done():
                    fut.set_result(msg)
            for cb in self._frame_data_cbs:
                self._spawn(cb(msg))
        elif type_id == MsgType.AUDIO_CHUNK:
            for cb in self._audio_cbs:
                self._spawn(cb(msg))
        elif type_id == MsgType.DATA_MESSAGE:
            for cb in self._data_cbs:
                self._spawn(cb(msg))
        elif type_id == MsgType.PARTICIPANT_EVENT:
            # Update participant set before spawning callbacks.
            if msg.joined:
                self._participants.add(msg.participant_id)
            else:
                self._participants.discard(msg.participant_id)
            for cb in self._participant_cbs:
                self._spawn(cb(msg))
        else:
            log.debug("Unhandled message type %d on processor endpoint", type_id)

    @staticmethod
    def _spawn(coro) -> None:
        t = asyncio.create_task(coro)
        def _on_done(t: asyncio.Task) -> None:
            if not t.cancelled() and (exc := t.exception()):
                log.critical("Unhandled error in processor callback — crashing",
                             exc_info=exc)
                os._exit(1)
        t.add_done_callback(_on_done)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._sub.close(linger=0)
        self._push.close(linger=0)
