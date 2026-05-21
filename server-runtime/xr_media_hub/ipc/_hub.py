# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hub-side IPC endpoint (server).

Connectors register themselves on startup; the hub opens their ring buffers
on demand. From the application's perspective (on_frame, on_audio, etc.) the
connector topology is invisible — callbacks receive participant_id / track_id
regardless of how many connectors exist or how many participants each carries.

  connector_A ──PUSH──┐
  connector_B ──PUSH──┤─► PULL   HubEndpoint   PUB ──SUB──► consumers
  connector_N ──PUSH──┘    ↓ dispatch
                        on_frame / on_audio / on_data / on_participant

Isolation contract
──────────────────
The hub is NOT a routing switch between participants. There is no supported
path for participant A's data to reach participant B. The only supported flow
is: participant → hub → consumer (agent) → hub → same participant.

Enforcement:
  • send_return_audio / send_return_data / send_return_audio_flush validate
    that the target participant is currently connected; unknown targets are
    dropped with a warning.
  • Return-traffic topics (return_audio.*, return_audio_flush.*, return_data.*)
    are connector-only; ProcessorEndpoint's default subscription excludes them.
  • The LiveKit transport publishes one return-audio track per participant
    (xr-hub-return-{pid}) with subscribe permissions restricted so each pid
    can only receive their own track. Return data uses destination_identities
    so it is not broadcast to other participants.

Frame callbacks receive a SlotView (zero-copy memoryview into the originating
connector's ring buffer). The slot is released after ALL frame callbacks
return — do not hold the view beyond the callback boundary.
"""
from __future__ import annotations

from loguru import logger
import time
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from xr_ai_agent import (AudioChunk, ConnectorRegistration, ControlMessage,
                         DataMessage, FrameData, FrameRequest, MsgType, ParticipantEvent,
                         ReturnAudioFlush, ShmRingBuffer, SlotView, decode, encode)


def _now_us() -> int:
    return time.time_ns() // 1_000

FrameCallback       = Callable[[SlotView],          Awaitable[None]]
AudioCallback       = Callable[[AudioChunk],        Awaitable[None]]
DataCallback        = Callable[[DataMessage],       Awaitable[None]]
ParticipantCallback = Callable[[ParticipantEvent],  Awaitable[None]]
ControlCallback     = Callable[[ControlMessage],    Awaitable[None]]

# Topic prefixes for ZMQ PUB/SUB.
# Format: "<type>.<participant_id>.<track_or_topic>"
# ZMQ prefix matching lets consumers subscribe at any granularity:
#   b"audio"                    — all audio, all participants
#   b"audio.alice"              — all of alice's audio tracks
#   b"audio.alice.TR_mic_001"   — alice's specific mic track
#   b"data.alice.chat"          — alice's "chat" data channel only
#   b"participant"              — join/leave events
#   b"control"                  — hub control messages
TOPIC_VIDEO              = b"video"       # FRAME_SIGNAL metadata (fires at full frame rate)
TOPIC_VIDEO_DATA         = b"video_data"  # FRAME_DATA pixel response (on-demand only)
TOPIC_AUDIO              = b"audio"
TOPIC_DATA               = b"data"
TOPIC_CONTROL            = b"control"
TOPIC_RETURN_AUDIO       = b"return_audio"
TOPIC_RETURN_AUDIO_FLUSH = b"return_audio_flush"
TOPIC_RETURN_DATA        = b"return_data"


class HubEndpoint:
    """
    Hub-side IPC endpoint.

    Parameters
    ----------
    pull_addr : ZMQ address the hub binds for connector PUSH traffic.
    pub_addr  : ZMQ address the hub binds for consumer SUB traffic.
    """

    def __init__(self, pull_addr: str, pub_addr: str) -> None:
        ctx = zmq.asyncio.Context.instance()

        self._pull: zmq.asyncio.Socket = ctx.socket(zmq.PULL)
        self._pull.bind(pull_addr)

        self._pub: zmq.asyncio.Socket = ctx.socket(zmq.PUB)
        self._pub.bind(pub_addr)

        # connector_id → ShmRingBuffer (opened on CONNECTOR_REGISTER)
        self._ring_registry: dict[str, ShmRingBuffer] = {}
        # participant_id → connector_id (updated on PARTICIPANT_EVENT)
        self._participant_connector: dict[str, str] = {}
        # (participant_id, track_id) → (ring, SlotView) of the latest frame.
        # The slot is held open (not released) until the next frame for the same
        # track arrives, the participant disconnects, or the hub shuts down — so
        # pixels can be copied on demand without eager allocation while still
        # bounding ring occupancy across participant churn.
        self._latest_slots: dict[tuple[str, str], tuple[ShmRingBuffer, SlotView]] = {}

        self._frame_cbs:       list[FrameCallback]       = []
        self._audio_cbs:       list[AudioCallback]       = []
        self._data_cbs:        list[DataCallback]        = []
        self._participant_cbs: list[ParticipantCallback] = []
        self._control_cbs:     list[ControlCallback]     = []
        self._running = False

    # ── callback registration ─────────────────────────────────────────────────

    def on_frame(self,       cb: FrameCallback)       -> None: self._frame_cbs.append(cb)
    def on_audio(self,       cb: AudioCallback)       -> None: self._audio_cbs.append(cb)
    def on_data(self,        cb: DataCallback)        -> None: self._data_cbs.append(cb)
    def on_participant(self, cb: ParticipantCallback) -> None: self._participant_cbs.append(cb)
    def on_control(self,     cb: ControlCallback)     -> None: self._control_cbs.append(cb)

    # ── outbound (hub → connectors / consumers) ───────────────────────────────

    async def broadcast(self, topic: bytes | str, type_id: int, msg) -> None:
        """Send an arbitrary message to all subscribers of topic."""
        t = topic.encode() if isinstance(topic, str) else topic
        await self._pub.send_multipart([t, encode(type_id, msg)])

    async def send_return_audio(self, chunk: AudioChunk) -> None:
        """
        Send TTS/agent audio back to a specific connected participant.

        Drops the message with a warning if the participant is not currently
        connected — the hub does not support cross-participant routing.
        """
        if not self._is_connected(chunk.participant_id):
            logger.warning(
                "send_return_audio: participant {!r} not connected — dropped",
                chunk.participant_id,
            )
            return
        topic = f"return_audio.{chunk.participant_id}".encode()
        await self._pub.send_multipart([topic, encode(MsgType.RETURN_AUDIO, chunk)])

    async def send_return_data(self, msg: DataMessage) -> None:
        """
        Send agent text/binary back to a specific connected participant.

        Drops the message with a warning if the participant is not currently
        connected — the hub does not support cross-participant routing.
        """
        if not self._is_connected(msg.participant_id):
            logger.warning(
                "send_return_data: participant {!r} not connected — dropped",
                msg.participant_id,
            )
            return
        topic = f"return_data.{msg.participant_id}.{msg.topic}".encode()
        await self._pub.send_multipart([topic, encode(MsgType.RETURN_DATA, msg)])

    async def send_return_audio_flush(self, flush: ReturnAudioFlush) -> None:
        """
        Tell the connector to drop any audio queued for *flush.participant_id*'s
        return track. Used by processors to cleanly interrupt the agent's own
        audio playback. No-op for unknown participants.
        """
        if not self._is_connected(flush.participant_id):
            logger.warning(
                "send_return_audio_flush: participant {!r} not connected — dropped",
                flush.participant_id,
            )
            return
        topic = f"return_audio_flush.{flush.participant_id}".encode()
        await self._pub.send_multipart([topic, encode(MsgType.RETURN_AUDIO_FLUSH, flush)])

    def _is_connected(self, participant_id: str) -> bool:
        return participant_id in self._participant_connector

    # ── receive loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Receive and dispatch messages from all connectors until stop()."""
        self._running = True
        while self._running:
            try:
                raw = await self._pull.recv()
            except zmq.ZMQError as exc:
                if not self._running:
                    break
                logger.error("ZMQ recv error: {}", exc)
                continue
            try:
                type_id, msg = decode(raw)
                await self._dispatch(type_id, msg)
            except Exception:
                logger.exception("Error dispatching message")

    async def _dispatch(self, type_id: int, msg) -> None:
        if type_id == MsgType.CONNECTOR_REGISTER:
            self._handle_registration(msg)

        elif type_id == MsgType.FRAME_SIGNAL:
            connector_id = self._participant_connector.get(msg.participant_id)
            if connector_id is None:
                logger.warning("Frame for unknown participant {} — dropped", msg.participant_id)
                return
            ring = self._ring_registry.get(connector_id)
            if ring is None:
                logger.warning("Ring buffer for connector {} not found — dropped", connector_id)
                return

            key = (msg.participant_id, msg.track_id)

            # Release the previously held slot for this track before taking the new one.
            prev = self._latest_slots.pop(key, None)
            if prev:
                prev[0].release_slot(prev[1].signal.slot)

            # Read and hold the new slot — NOT released until the next frame
            # arrives or the hub shuts down, so pixels remain readable on demand.
            view = ring.read_slot(msg)
            self._latest_slots[key] = (ring, view)

            # Publish metadata so processors know a frame arrived.
            topic = f"video.{msg.participant_id}.{msg.track_id}".encode()
            await self._pub.send_multipart([topic, encode(MsgType.FRAME_SIGNAL, msg)])

            # Call hub-local frame callbacks while the slot is still held.
            # These must not prevent the ZMQ publish above from completing.
            for cb in self._frame_cbs:
                try:
                    await cb(view)
                except Exception:
                    logger.exception("frame callback error")

        elif type_id == MsgType.FRAME_REQUEST:
            key = (msg.participant_id, msg.track_id)
            held = self._latest_slots.get(key)
            if held is None:
                logger.debug(
                    "FRAME_REQUEST for {}/{} — no frame held",
                    msg.participant_id, msg.track_id,
                )
                return
            _, view = held
            sig = view.signal
            frame_data = FrameData(
                seq=sig.seq, pts_us=sig.pts_us,
                width=sig.width, height=sig.height, fmt=sig.fmt,
                data=bytes(view.data[:sig.data_sz]),
                participant_id=sig.participant_id, track_id=sig.track_id,
            )
            topic = f"video_data.{msg.participant_id}.{msg.track_id}".encode()
            await self._pub.send_multipart([topic, encode(MsgType.FRAME_DATA, frame_data)])

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

        elif type_id == MsgType.PARTICIPANT_EVENT:
            if msg.joined:
                self._participant_connector[msg.participant_id] = msg.connector_id
            else:
                self._participant_connector.pop(msg.participant_id, None)
                # Release any slots held for this participant's tracks. Without
                # this the ring fills up after enough connect/publish/disconnect
                # cycles and every subsequent frame is dropped (issue #143).
                stale = [k for k in self._latest_slots if k[0] == msg.participant_id]
                for k in stale:
                    ring, view = self._latest_slots.pop(k)
                    ring.release_slot(view.signal.slot)
            for cb in self._participant_cbs:
                await cb(msg)
            await self._pub.send_multipart([b"participant", encode(MsgType.PARTICIPANT_EVENT, msg)])

        elif type_id == MsgType.CONTROL:
            for cb in self._control_cbs:
                await cb(msg)
            await self._pub.send_multipart([TOPIC_CONTROL, encode(MsgType.CONTROL, msg)])

        elif type_id == MsgType.RETURN_AUDIO:
            await self.send_return_audio(msg)

        elif type_id == MsgType.RETURN_DATA:
            await self.send_return_data(msg)

        elif type_id == MsgType.RETURN_AUDIO_FLUSH:
            await self.send_return_audio_flush(msg)

        elif type_id == MsgType.ROSTER_REQUEST:
            await self._replay_roster()

        else:
            logger.warning("Unknown message type {} — ignored", type_id)

    async def _replay_roster(self) -> None:
        """Re-publish PARTICIPANT_EVENT(joined=True) for every connected pid.

        Used by ProcessorEndpoints starting up mid-session so they can
        subscribe to clients who joined before they connected. The events
        go on the regular ``participant`` topic, so all current
        subscribers see them — keep on_participant callbacks idempotent.
        """
        pts_us = _now_us()
        for pid, connector_id in self._participant_connector.items():
            event = ParticipantEvent(
                participant_id=pid, joined=True,
                pts_us=pts_us, connector_id=connector_id,
            )
            await self._pub.send_multipart([
                b"participant", encode(MsgType.PARTICIPANT_EVENT, event),
            ])

    def _handle_registration(self, reg: ConnectorRegistration) -> None:
        if reg.connector_id in self._ring_registry:
            logger.warning("Connector {} re-registered — replacing ring buffer", reg.connector_id)
            self._ring_registry[reg.connector_id].close()
        try:
            self._ring_registry[reg.connector_id] = ShmRingBuffer(
                name=reg.shm_name, create=False,
            )
            logger.info("Connector {} registered (shm={})", reg.connector_id, reg.shm_name)
        except Exception:
            logger.exception(
                "Failed to open shm {} for connector {}",
                reg.shm_name, reg.connector_id,
            )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._pull.close(linger=0)
        self._pub.close(linger=0)
        for ring, view in self._latest_slots.values():
            view.data.release()  # must release before ring.close() so mmap has no exported pointers
            ring.release_slot(view.signal.slot)
        self._latest_slots.clear()
        for ring in self._ring_registry.values():
            ring.close()
        self._ring_registry.clear()
