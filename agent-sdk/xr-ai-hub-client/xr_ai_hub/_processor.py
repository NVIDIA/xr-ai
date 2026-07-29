# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Processor-side IPC endpoint (subscriber + publisher).

Connects to the hub's PUB socket to receive real-time video signals, audio,
data, and participant events. Also connects a PUSH socket to send RETURN_DATA,
RETURN_AUDIO, and FRAME_REQUEST back to the hub.

Works for any downstream processing workload — analytics, ML inference,
transcription, echo, recording — not just agentic pipelines.

Subscription model
------------------
Participants are the unit of subscription. By default the endpoint
subscribes to every participant who joins (and unsubscribes on leave),
giving each agent the full inbound stream — data, audio, and video — for
every client. Two knobs control this:

* ``filter`` — a :class:`Subscribe` flag that drops whole categories
  (``DATA`` / ``AUDIO`` / ``VIDEO``) at the ZMQ kernel level for
  efficiency. Default is ``Subscribe.ALL``. Set to e.g.
  ``Subscribe.DATA | Subscribe.AUDIO`` to skip video frames.
* ``auto_subscribe`` — when ``True`` (default), the endpoint installs an
  internal participant handler that calls ``subscribe(pid)`` on join and
  ``unsubscribe(pid)`` on leave. Set to ``False`` for agents that only
  service a fixed set of participants — call ``subscribe(pid)`` yourself.

Endpoints created mid-session use a roster request to learn about
participants who joined before they did. The hub re-publishes
``PARTICIPANT_EVENT(joined=True)`` for every current pid, so already-
connected pids are auto-subscribed retroactively. The replays go on the
regular ``participant`` topic; keep your ``on_participant`` callbacks
idempotent.

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
from enum import Flag, auto
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from ._codec import decode, encode
from ._types import (AudioChunk, DataMessage, FrameData, FrameRequest,
                     FrameSignal, MsgType, ParticipantEvent, ReturnAudioFlush,
                     RosterRequest)

log = logging.getLogger(__name__)

FrameSignalCallback = Callable[[FrameSignal], Awaitable[None]]
FrameDataCallback   = Callable[[FrameData],   Awaitable[None]]
AudioCallback       = Callable[[AudioChunk],       Awaitable[None]]
DataCallback        = Callable[[DataMessage],      Awaitable[None]]
ParticipantCallback = Callable[[ParticipantEvent], Awaitable[None]]

# Reserved topic for internal SDK status messages — not forwarded to app callbacks.
AGENT_STATUS_TOPIC = "_agent.status"

_FRAME_REQUEST_TIMEOUT = 1.0  # seconds before request_frame() gives up

# Always-on global topics. ``participant`` is required for auto-subscribe to
# work at all; ``control`` carries hub control messages and is cheap.
_GLOBAL_TOPICS: tuple[bytes, ...] = (b"participant", b"control")


class Subscribe(Flag):
    """Per-participant message-category filter.

    Each flag corresponds to a class of pid-scoped ZMQ topics on the hub
    PUB socket. ``Subscribe.ALL`` (the default) gets every category for
    each subscribed participant; combine flags with ``|`` to scope down.

    Example
    -------
    ::

        # Audio-only processor; ignores data + video on every pid.
        ep = ProcessorEndpoint(..., filter=Subscribe.AUDIO)

        # Per-pid override at subscribe time:
        ep.subscribe("alice", filter=Subscribe.DATA)
    """
    DATA  = auto()  # `data.{pid}.*`
    AUDIO = auto()  # `audio.{pid}.*`
    VIDEO = auto()  # `video.{pid}.*` AND `video_data.{pid}.*` (signal + pixels)
    ALL   = DATA | AUDIO | VIDEO


# Topic-prefix categories used by the subscription machinery. Each Subscribe
# flag maps to one or more pid-scoped ZMQ topic prefixes; the trailing
# ``.{pid}.`` is appended at subscribe time so e.g. ``data.alice`` does not
# accidentally match ``data.alice2.chat``.
_PREFIXES_BY_FLAG: dict["Subscribe", tuple[bytes, ...]] = {
    Subscribe.DATA:  (b"data",),
    Subscribe.AUDIO: (b"audio",),
    Subscribe.VIDEO: (b"video", b"video_data"),
}


def _prefixes(filter_: Subscribe, pid: str) -> list[bytes]:
    """Return the pid-scoped ZMQ topic prefixes implied by ``filter_``."""
    prefixes: list[bytes] = []
    pid_bytes = pid.encode()
    for flag, categories in _PREFIXES_BY_FLAG.items():
        if filter_ & flag:
            for cat in categories:
                prefixes.append(cat + b"." + pid_bytes + b".")
    return prefixes


class ProcessorEndpoint:
    """
    Downstream IPC endpoint for data processors.

    See the module docstring for the subscription model.

    ::

        ep = ProcessorEndpoint(
            sub_addr="ipc:///tmp/xr_hub_pub",
            push_addr="ipc:///tmp/xr_hub_in",
        )
        ep.on_audio(handle_audio)
        ep.on_data(handle_data)
        ep.on_participant(handle_participant)  # optional — set is auto-maintained
        await ep.run()

    Audio-only processor that ignores video frames at the kernel level::

        ep = ProcessorEndpoint(..., filter=Subscribe.AUDIO | Subscribe.DATA)

    Single-client agent — opt out of auto-subscribe and pin one pid::

        ep = ProcessorEndpoint(..., auto_subscribe=False)
        ep.subscribe("alice")  # may be called before alice has joined
    """

    def __init__(
        self,
        sub_addr:        str,
        push_addr:       str,
        *,
        auto_subscribe:  bool = True,
        filter:          Subscribe = Subscribe.ALL,
    ) -> None:
        ctx = zmq.asyncio.Context.instance()

        self._sub: zmq.asyncio.Socket = ctx.socket(zmq.SUB)
        self._sub.connect(sub_addr)      # ZMQ retries until the hub binds — startup order is irrelevant
        for t in _GLOBAL_TOPICS:
            self._sub.setsockopt(zmq.SUBSCRIBE, t)

        self._push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._push.connect(push_addr)    # same — outbound messages queue until hub is ready

        self._auto_subscribe = auto_subscribe
        self._default_filter = filter

        # pid → currently-applied filter. Tracks which subscriptions are
        # live on the SUB socket so subscribe()/unsubscribe() are idempotent.
        self._subscribed: dict[str, Subscribe] = {}

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

    @property
    def subscribed_participants(self) -> frozenset[str]:
        """Participant IDs this endpoint currently has live SUBSCRIBEs for.

        With ``auto_subscribe=True`` this tracks ``connected_participants``.
        With ``auto_subscribe=False`` it reflects whatever the caller has
        explicitly subscribed to via :meth:`subscribe`.
        """
        return frozenset(self._subscribed)

    # ── subscription primitives ──────────────────────────────────────────────

    def subscribe(self, participant_id: str,
                  *, filter: Subscribe | None = None) -> None:
        """Subscribe to (a subset of) traffic for *participant_id*.

        Idempotent. Calling with a different ``filter`` than a previous
        call updates the live subscriptions — the diff is unsubscribed
        and the new categories are subscribed. Subscribing to a pid who
        is not yet connected is fine; ZMQ holds the SUBSCRIBE until
        matching traffic arrives.

        Parameters
        ----------
        participant_id :
            Target participant.
        filter :
            Categories to receive. Defaults to the constructor ``filter``.
        """
        new_filter = filter if filter is not None else self._default_filter
        old_filter = self._subscribed.get(participant_id, Subscribe(0))

        added   = new_filter & ~old_filter
        removed = old_filter & ~new_filter

        for pre in _prefixes(removed, participant_id):
            self._sub.setsockopt(zmq.UNSUBSCRIBE, pre)
        for pre in _prefixes(added, participant_id):
            self._sub.setsockopt(zmq.SUBSCRIBE, pre)

        if new_filter:
            self._subscribed[participant_id] = new_filter
        else:
            self._subscribed.pop(participant_id, None)

    def unsubscribe(self, participant_id: str) -> None:
        """Drop every subscription for *participant_id*. Idempotent."""
        old = self._subscribed.pop(participant_id, Subscribe(0))
        for pre in _prefixes(old, participant_id):
            self._sub.setsockopt(zmq.UNSUBSCRIBE, pre)

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

    async def flush_return_audio(self, participant_id: str) -> None:
        """
        Drop any return audio currently queued at the hub for *participant_id*.

        Use to cleanly interrupt the agent's own audio playback (e.g. when
        cancelling an in-flight TTS response on a new user query). Audio that
        has already left the hub for the client may still play out for the
        duration of the client's jitter buffer (~100 ms).
        """
        await self._push.send(encode(
            MsgType.RETURN_AUDIO_FLUSH,
            ReturnAudioFlush(participant_id=participant_id),
        ))

    async def request_roster(self) -> None:
        """
        Ask the hub to re-publish ``PARTICIPANT_EVENT(joined=True)`` for
        every currently-connected participant.

        Useful when starting up mid-session so the auto-subscribe handler
        can pick up clients who joined before this endpoint connected.
        Called automatically once at the start of :meth:`run` when
        ``auto_subscribe=True``.
        """
        await self._push.send(encode(MsgType.ROSTER_REQUEST, RosterRequest()))

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

    # Time to wait for the SUB↔PUB handshake to complete before sending
    # the first roster request. ZMQ drops PUB messages whose topic has no
    # registered SUBSCRIBE yet (the slow-joiner problem); without this
    # window, an endpoint started mid-session can miss the roster replay
    # for ``participant`` events.
    _ROSTER_HANDSHAKE_WAIT = 0.1

    async def run(self) -> None:
        """Receive and dispatch messages until stop() is called."""
        self._running = True

        # Ask the hub to replay PARTICIPANT_EVENTs for already-connected
        # pids so the auto-subscribe handler can scoop them up. Safe even
        # when there are none: the hub responds with zero events.
        if self._auto_subscribe:
            asyncio.create_task(self._catch_up_roster())

        while self._running:
            try:
                _topic, raw = await self._sub.recv_multipart()
            except asyncio.CancelledError:
                break
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

    async def _catch_up_roster(self) -> None:
        """Send a roster request after the SUB↔PUB handshake settles."""
        try:
            await asyncio.sleep(self._ROSTER_HANDSHAKE_WAIT)
            if self._running:
                await self.request_roster()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("roster catch-up failed")

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
            # Update participant set + auto-subscribe state synchronously
            # before spawning user callbacks so callbacks observe a
            # consistent roster / subscription view.
            if msg.joined:
                self._participants.add(msg.participant_id)
                if self._auto_subscribe:
                    self.subscribe(msg.participant_id)
            else:
                self._participants.discard(msg.participant_id)
                if self._auto_subscribe:
                    self.unsubscribe(msg.participant_id)
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
