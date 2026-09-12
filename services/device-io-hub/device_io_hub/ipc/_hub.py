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

import asyncio
import base64
from loguru import logger
import json
import time
from typing import Awaitable, Callable

import zmq
import zmq.asyncio

from xr_ai_hub import (AGENT_STATUS_TOPIC, AudioChunk, ConnectorRegistration,
                       ControlMessage, DataMessage, FileMessage, FrameData, MsgType,
                       ParticipantEvent, ReturnAudioFlush, ShmRingBuffer, SlotView,
                       decode, encode)

from ._registration import _CONNECTOR_REGISTER_ACK_TOPIC


def _now_us() -> int:
    return time.time_ns() // 1_000


# Room availability is the availability of its least-available agent: an agent
# that has not reported yet, or reports something unrecognised, counts as
# "loading" and "processing" respectively.
_STATUS_PRECEDENCE = ("loading", "processing", "idle", "ready")

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
TOPIC_FILE               = b"file"
TOPIC_CONTROL            = b"control"
TOPIC_RETURN_AUDIO       = b"return_audio"
TOPIC_RETURN_AUDIO_FLUSH = b"return_audio_flush"
TOPIC_RETURN_DATA        = b"return_data"
_DEFAULT_FILE_MAX_BYTES = 16 * 1024 * 1024
_FILE_IPC_METADATA_ALLOWANCE = 64 * 1024


def _file_prefix(participant_id: str) -> bytes:
    """Return an exact, delimiter-safe participant prefix for file IPC."""
    encoded = base64.urlsafe_b64encode(participant_id.encode("utf-8")).rstrip(b"=")
    return b"file." + encoded + b"."


class HubEndpoint:
    """
    Hub-side IPC endpoint.

    The optional file addresses must be configured together. ``file_hwm``
    limits queued complete files per socket, and ``file_max_bytes`` bounds one
    payload before it enters the file fan-out lane.
    """

    def __init__(
        self,
        pull_addr: str,
        pub_addr: str,
        *,
        file_pull_addr: str | None = None,
        file_pub_addr: str | None = None,
        file_hwm: int = 2,
        file_max_bytes: int = _DEFAULT_FILE_MAX_BYTES,
    ) -> None:
        if isinstance(file_hwm, bool) or not isinstance(file_hwm, int) or file_hwm <= 0:
            raise ValueError("file_hwm must be a positive integer")
        if (
            isinstance(file_max_bytes, bool)
            or not isinstance(file_max_bytes, int)
            or file_max_bytes <= 0
        ):
            raise ValueError("file_max_bytes must be a positive integer")
        ctx = zmq.asyncio.Context.instance()

        self._pull: zmq.asyncio.Socket = ctx.socket(zmq.PULL)
        self._pull.bind(pull_addr)

        self._pub: zmq.asyncio.Socket = ctx.socket(zmq.PUB)
        self._pub.bind(pub_addr)

        self._file_pull: zmq.asyncio.Socket | None = None
        self._file_pub: zmq.asyncio.Socket | None = None
        self._file_max_bytes = file_max_bytes
        if (file_pull_addr is None) != (file_pub_addr is None):
            raise ValueError("file_pull_addr and file_pub_addr must be configured together")
        if file_pull_addr is not None and file_pub_addr is not None:
            self._file_pull = ctx.socket(zmq.PULL)
            self._file_pull.setsockopt(zmq.RCVHWM, file_hwm)
            self._file_pull.setsockopt(
                zmq.MAXMSGSIZE,
                file_max_bytes + _FILE_IPC_METADATA_ALLOWANCE,
            )
            self._file_pull.bind(file_pull_addr)
            self._file_pub = ctx.socket(zmq.PUB)
            self._file_pub.setsockopt(zmq.SNDHWM, file_hwm)
            self._file_pub.bind(file_pub_addr)

        # connector_id → ShmRingBuffer (opened on CONNECTOR_REGISTER)
        self._ring_registry: dict[str, ShmRingBuffer] = {}
        self._ring_names: dict[str, str] = {}
        # participant_id → connector_id (updated on PARTICIPANT_EVENT)
        self._participant_connector: dict[str, str] = {}
        self._participant_sessions: dict[str, str] = {}
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

        # agent_id → {participant_id → status}. An attached agent with no entry
        # for a participant has not reported for it yet.
        self._agent_status: dict[str, dict[str, str]] = {}
        # agent_id → participants it answers for; None means all of them.
        self._agent_scope: dict[str, set[str] | None] = {}
        # participant_id → last aggregate published, to suppress duplicates.
        self._published_status: dict[str, str] = {}

    # ── callback registration ─────────────────────────────────────────────────

    def on_frame(self,       cb: FrameCallback)       -> None: self._frame_cbs.append(cb)
    def on_audio(self,       cb: AudioCallback)       -> None: self._audio_cbs.append(cb)
    def on_data(self,        cb: DataCallback)        -> None: self._data_cbs.append(cb)
    def on_participant(self, cb: ParticipantCallback) -> None: self._participant_cbs.append(cb)
    def on_control(self,     cb: ControlCallback)     -> None: self._control_cbs.append(cb)

    @staticmethod
    def _release_held_slot(
        ring: ShmRingBuffer,
        view: SlotView,
        *,
        context: str,
    ) -> None:
        """Release a latest-frame view without disrupting hub lifecycle."""
        try:
            view.data.release()
        except ValueError as exc:
            logger.debug("Frame view was already released during {}: {}", context, exc)
        try:
            ring.release_slot(view.signal.slot)
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Could not release shared-memory slot {} during {}: {}",
                view.signal.slot,
                context,
                exc,
            )

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
        # Trailing "." terminates the pid segment so a connector subscribed for
        # `alice` does not also receive `alice2`'s return audio (ZMQ SUBSCRIBE is
        # a byte-prefix match). The connector subscribes with the same delimiter.
        topic = f"return_audio.{chunk.participant_id}.".encode()
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
        # Trailing "." — same pid-segment guard as send_return_audio.
        topic = f"return_audio_flush.{flush.participant_id}.".encode()
        await self._pub.send_multipart([topic, encode(MsgType.RETURN_AUDIO_FLUSH, flush)])

    def _is_connected(self, participant_id: str) -> bool:
        return participant_id in self._participant_connector

    # ── agent status aggregation ─────────────────────────────────────────────

    def _record_agent_status(self, msg: DataMessage) -> str | None:
        """Record one agent's per-participant status. Returns its agent id.

        Returns *None* when the payload does not identify an agent, which
        means an SDK too old to participate in aggregation — those updates
        are forwarded verbatim so the client is not left without a status.
        """
        try:
            payload = json.loads(msg.data)
            agent_id = payload["agent_id"]
            status   = payload["status"]
        except (ValueError, TypeError, KeyError, UnicodeDecodeError):
            return None
        self._agent_status.setdefault(agent_id, {})[msg.participant_id] = str(status)
        return str(agent_id)

    def _responsible_agents(self, participant_id: str) -> list[str]:
        """Agent ids that answer for *participant_id*."""
        return [
            agent_id
            for agent_id, scope in self._agent_scope.items()
            if scope is None or participant_id in scope
        ]

    def _aggregate_status(self, participant_id: str) -> str:
        """Fold the responsible agents' states into the one status a client sees.

        Agents scoped to other participants are excluded, and a passive
        processor never registers at all — neither can hold this client back.
        """
        responsible = self._responsible_agents(participant_id)
        if not responsible:
            return "loading"
        worst = len(_STATUS_PRECEDENCE) - 1
        for agent_id in responsible:
            status = self._agent_status.get(agent_id, {}).get(participant_id)
            if status is None:
                return _STATUS_PRECEDENCE[0]
            rank = (
                _STATUS_PRECEDENCE.index(status)
                if status in _STATUS_PRECEDENCE
                else _STATUS_PRECEDENCE.index("processing")
            )
            worst = min(worst, rank)
        return _STATUS_PRECEDENCE[worst]

    async def publish_agent_status(self, participant_id: str, *,
                                   force: bool = False) -> None:
        """Publish the aggregate agent status for *participant_id*.

        Skips the send when the aggregate has not moved, so the agents'
        periodic re-announcements do not become per-agent client traffic.
        """
        if not self._is_connected(participant_id):
            return
        status = self._aggregate_status(participant_id)
        if not force and self._published_status.get(participant_id) == status:
            return
        self._published_status[participant_id] = status
        await self.send_return_data(DataMessage(
            participant_id=participant_id,
            topic=AGENT_STATUS_TOPIC,
            pts_us=_now_us(),
            data=json.dumps({"status": status}).encode(),
        ))

    async def _republish_agent_status(self) -> None:
        for pid in list(self._participant_connector):
            await self.publish_agent_status(pid)

    # ── receive loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Receive and dispatch messages from all connectors until stop()."""
        self._running = True
        file_tasks = []
        if self._file_pull is not None:
            file_tasks = [asyncio.create_task(self._run_files(), name="hub-file-ipc")]
        try:
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
        finally:
            for task in file_tasks:
                task.cancel()
            await asyncio.gather(*file_tasks, return_exceptions=True)

    async def _run_files(self) -> None:
        """Route bounded complete-file messages independently of real-time traffic."""
        assert self._file_pull is not None
        assert self._file_pub is not None
        while self._running:
            try:
                raw = await self._file_pull.recv()
                type_id, msg = await asyncio.to_thread(decode, raw)
                if type_id != MsgType.FILE_MESSAGE:
                    logger.warning("Unknown file IPC message type {} ignored", type_id)
                    continue
                if len(msg.data) > self._file_max_bytes:
                    logger.warning(
                        "File {} dropped: {} bytes exceeds the {} byte IPC limit",
                        msg.transfer_id,
                        len(msg.data),
                        self._file_max_bytes,
                    )
                    continue
                active_session = self._participant_sessions.get(msg.participant_id)
                if active_session is None or msg.participant_session_id != active_session:
                    logger.info(
                        "File {} dropped: participant session is no longer active",
                        msg.transfer_id,
                    )
                    continue
                topic = _file_prefix(msg.participant_id) + msg.topic.encode("utf-8")
                encoded = await asyncio.to_thread(encode, MsgType.FILE_MESSAGE, msg)
                await self._file_pub.send_multipart([topic, encoded])
            except asyncio.CancelledError:
                raise
            except zmq.ZMQError as exc:
                if not self._running:
                    return
                logger.error("File IPC ZMQ error: {}", exc)
            except Exception:
                logger.exception("Error routing file message")

    async def _dispatch(self, type_id: int, msg) -> None:
        if type_id == MsgType.CONNECTOR_REGISTER:
            await self._handle_registration(msg)

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

            # Read and hold the new slot — NOT released until the next frame
            # arrives or the hub shuts down, so pixels remain readable on demand.
            try:
                view = ring.read_slot(msg)
            except (RuntimeError, ValueError) as exc:
                # A rejected signal normally refers to a freshly written READY
                # slot. Free it so malformed metadata cannot exhaust the ring,
                # but never release a slot already held by another latest view.
                held = any(
                    held_ring is ring and held_view.signal.slot == msg.slot
                    for held_ring, held_view in self._latest_slots.values()
                )
                if not held:
                    try:
                        ring.release_slot(msg.slot)
                    except (RuntimeError, ValueError) as release_exc:
                        logger.debug(
                            "Could not release rejected frame slot {}: {}",
                            msg.slot,
                            release_exc,
                        )
                logger.warning(
                    "Invalid frame signal for participant {} track {} — dropped: {}",
                    msg.participant_id,
                    msg.track_id,
                    exc,
                )
                return

            held_key = next(
                (
                    held_key
                    for held_key, (held_ring, held_view) in self._latest_slots.items()
                    if held_ring is ring and held_view.signal.slot == msg.slot
                ),
                None,
            )
            if held_key is not None:
                view.data.release()
                logger.debug(
                    "Duplicate frame signal for participant {} track {} slot {} "
                    "already held for {}/{} — ignored",
                    msg.participant_id,
                    msg.track_id,
                    msg.slot,
                    *held_key,
                )
                return

            prev = self._latest_slots.get(key)
            if prev:
                self._latest_slots.pop(key)
                self._release_held_slot(
                    *prev,
                    context=f"replacement of {msg.participant_id}/{msg.track_id}",
                )
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
                try:
                    await cb(msg)
                except Exception:
                    logger.exception("audio callback error")
            topic = f"audio.{msg.participant_id}.{msg.track_id}".encode()
            await self._pub.send_multipart([topic, encode(MsgType.AUDIO_CHUNK, msg)])

        elif type_id == MsgType.DATA_MESSAGE:
            for cb in self._data_cbs:
                try:
                    await cb(msg)
                except Exception:
                    logger.exception("data callback error")
            topic = f"data.{msg.participant_id}.{msg.topic}".encode()
            await self._pub.send_multipart([topic, encode(MsgType.DATA_MESSAGE, msg)])

        elif type_id == MsgType.PARTICIPANT_EVENT:
            # Some embedders construct a lightweight HubEndpoint instance for
            # dispatch-only testing, so initialise newly added state lazily too.
            participant_sessions = getattr(self, "_participant_sessions", None)
            if participant_sessions is None:
                participant_sessions = {}
                self._participant_sessions = participant_sessions
            if msg.joined:
                self._participant_connector[msg.participant_id] = msg.connector_id
                participant_sessions[msg.participant_id] = (
                    msg.participant_session_id
                )
            else:
                active_session = participant_sessions.get(msg.participant_id, "")
                if (
                    msg.participant_session_id
                    and active_session
                    and msg.participant_session_id != active_session
                ):
                    logger.debug(
                        "Ignoring stale departure for participant {!r}",
                        msg.participant_id,
                    )
                    return
                self._participant_connector.pop(msg.participant_id, None)
                participant_sessions.pop(msg.participant_id, None)
                self._published_status.pop(msg.participant_id, None)
                for per_participant in self._agent_status.values():
                    per_participant.pop(msg.participant_id, None)
                # Release any slots held for this participant's tracks. Without
                # this the ring fills up after enough connect/publish/disconnect
                # cycles and every subsequent frame is dropped (issue #143).
                stale = [k for k in self._latest_slots if k[0] == msg.participant_id]
                for k in stale:
                    ring, view = self._latest_slots.pop(k)
                    self._release_held_slot(
                        ring,
                        view,
                        context=f"departure of {msg.participant_id}",
                    )
            for cb in self._participant_cbs:
                try:
                    await cb(msg)
                except Exception:
                    logger.exception("participant callback error")
            await self._pub.send_multipart([b"participant", encode(MsgType.PARTICIPANT_EVENT, msg)])
            if msg.joined:
                # A joining client has no status yet; publish the aggregate
                # unconditionally so it starts from a real state instead of
                # inheriting whatever the previous occupant of this pid saw.
                await self.publish_agent_status(msg.participant_id, force=True)

        elif type_id == MsgType.CONTROL:
            for cb in self._control_cbs:
                try:
                    await cb(msg)
                except Exception:
                    logger.exception("control callback error")
            await self._pub.send_multipart([TOPIC_CONTROL, encode(MsgType.CONTROL, msg)])

        elif type_id == MsgType.RETURN_AUDIO:
            await self.send_return_audio(msg)

        elif type_id == MsgType.RETURN_DATA:
            # Agent status is per-agent state; the client's is the aggregate,
            # so it is folded here rather than forwarded straight through.
            if msg.topic == AGENT_STATUS_TOPIC:
                if self._record_agent_status(msg) is not None:
                    await self.publish_agent_status(msg.participant_id)
                    return
            await self.send_return_data(msg)

        elif type_id == MsgType.RETURN_AUDIO_FLUSH:
            await self.send_return_audio_flush(msg)

        elif type_id == MsgType.ROSTER_REQUEST:
            await self._replay_roster()

        elif type_id == MsgType.SUBSCRIPTION_PROBE:
            probe = [
                f"_probe.{msg.token}".encode(),
                encode(MsgType.SUBSCRIPTION_PROBE, msg),
            ]
            await self._pub.send_multipart(probe)
            if self._file_pub is not None:
                await self._file_pub.send_multipart(probe)

        elif type_id == MsgType.AGENT_PRESENCE:
            if msg.attached:
                self._agent_status.setdefault(msg.agent_id, {})
                self._agent_scope[msg.agent_id] = (
                    None if msg.scope is None else set(msg.scope)
                )
            else:
                self._agent_status.pop(msg.agent_id, None)
                self._agent_scope.pop(msg.agent_id, None)
            await self._republish_agent_status()

        else:
            logger.warning("Unknown message type {} — ignored", type_id)

    async def _replay_roster(self) -> None:
        """Re-publish PARTICIPANT_EVENT(joined=True) for every connected pid.

        Used by ProcessorEndpoints starting up mid-session so they can
        subscribe to clients who joined before they connected. The events
        go on the regular ``participant`` topic, so all current
        subscribers see them. ProcessorEndpoint de-duplicates the resulting
        lifecycle transitions before invoking application callbacks.
        """
        pts_us = _now_us()
        for pid, connector_id in self._participant_connector.items():
            event = ParticipantEvent(
                participant_id=pid, joined=True,
                pts_us=pts_us, connector_id=connector_id,
                participant_session_id=self._participant_sessions.get(pid, ""),
            )
            await self._pub.send_multipart([
                b"participant", encode(MsgType.PARTICIPANT_EVENT, event),
            ])

    async def _acknowledge_registration(
        self,
        reg: ConnectorRegistration,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        ack = ControlMessage(
            topic=_CONNECTOR_REGISTER_ACK_TOPIC,
            payload={
                "connector_id": reg.connector_id,
                "shm_name": reg.shm_name,
                "success": not error_code,
                "error_code": error_code,
                "error_message": error_message,
            },
        )
        await self._pub.send_multipart([
            f"connector.{reg.connector_id}.".encode(),
            encode(MsgType.CONTROL, ack),
        ])

    async def _handle_registration(self, reg: ConnectorRegistration) -> None:
        if self._ring_names.get(reg.connector_id) == reg.shm_name:
            await self._acknowledge_registration(reg)
            return
        try:
            new_ring = ShmRingBuffer(name=reg.shm_name, create=False)
        except FileNotFoundError:
            error_code = "shm_not_found"
            error_message = f"shared memory {reg.shm_name!r} does not exist"
        except ValueError as exc:
            error_code = "shm_incompatible"
            error_message = f"shared memory {reg.shm_name!r} is incompatible: {exc}"
        except Exception as exc:
            error_code = "shm_open_failed"
            error_message = f"could not open shared memory {reg.shm_name!r}: {exc}"
            logger.opt(exception=exc).error(
                "Failed to register connector {} using shm {}",
                reg.connector_id,
                reg.shm_name,
            )
        else:
            old_ring = self._ring_registry.get(reg.connector_id)
            if old_ring is not None:
                logger.warning(
                    "Connector {} re-registered — replacing ring buffer",
                    reg.connector_id,
                )
                # Drop held frames before closing the old consumer mapping. A
                # SlotView exports a memoryview into the mapping and would leave
                # a half-closed ring behind if it survived replacement (#197).
                for key in [
                    k for k, (ring, _) in self._latest_slots.items()
                    if ring is old_ring
                ]:
                    _, view = self._latest_slots.pop(key)
                    self._release_held_slot(
                        old_ring,
                        view,
                        context=f"re-registration of {reg.connector_id}",
                    )
                old_ring.close()
            self._ring_registry[reg.connector_id] = new_ring
            self._ring_names[reg.connector_id] = reg.shm_name
            logger.info("Connector {} registered (shm={})", reg.connector_id, reg.shm_name)
            await self._acknowledge_registration(reg)
            return

        logger.warning(
            "Failed to register connector {}: {}",
            reg.connector_id,
            error_message,
        )
        await self._acknowledge_registration(reg, error_code, error_message)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._pull.close(linger=0)
        self._pub.close(linger=0)
        if self._file_pull is not None:
            self._file_pull.close(linger=0)
        if self._file_pub is not None:
            self._file_pub.close(linger=0)
        for ring, view in self._latest_slots.values():
            self._release_held_slot(ring, view, context="hub shutdown")
        self._latest_slots.clear()
        for ring in self._ring_registry.values():
            ring.close()
        self._ring_registry.clear()
        self._ring_names.clear()
