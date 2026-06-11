# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LiveKit Python room client.

Connects to the LiveKit room as the hub-side observer, then:
  • Calls notify_participant_joined/left on the IPC connector endpoint as
    participants enter and exit the room.
  • Streams decoded video frames (I420) into the ring buffer via push_frame().
  • Streams decoded audio (float32) via push_audio().
  • Forwards data-channel packets via push_data().

The client never publishes media — it is subscribe-only.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
from livekit import rtc
from loguru import logger

from xr_media_hub.ipc import (
    AudioChunk,
    ConnectorEndpoint,
    DataMessage,
    PixelFormat,
    ReturnAudioFlush,
)

from ._token import make_client_token
from .config import LiveKitConnectorConfig


def _now_us() -> int:
    return time.time_ns() // 1_000


class _ReturnAudioPipe:
    """Per-participant pacing pipe for return audio.

    Decouples the connector's IPC recv loop from LiveKit's ``capture_frame``.
    Without it, when the agent floods many TTS chunks back-to-back, the
    connector's serial recv loop blocks on capture_frame's internal-queue
    backpressure while a flush message sits FIFO-stuck behind dozens of
    audio chunks in the ZMQ SUB buffer — by the time flush is delivered,
    the audio is already past us.

    With it, ``push`` is a non-blocking ``put_nowait`` so the connector
    loop stays responsive; a background task drains the queue into
    ``capture_frame`` at audio rate; ``flush`` is O(1) and drops both
    layers instantly.  Only the client's jitter buffer (~100 ms) remains
    irreducibly outside our control.
    """

    def __init__(self, src: rtc.AudioSource) -> None:
        self._src   = src
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        self._task  = asyncio.create_task(self._drain(), name="return_audio_pipe")

    def push(self, frame: rtc.AudioFrame) -> None:
        self._queue.put_nowait(frame)

    def flush(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        self._src.clear_queue()

    async def _drain(self) -> None:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            try:
                await self._src.capture_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("capture_frame failed")

    async def close(self) -> None:
        # Cancel rather than enqueue a sentinel — a bounded queue could drop
        # the None and leak the drainer forever.
        if not self._task.done():
            self._task.cancel()
        try:
            await self._task
        # CancelledError is the expected success path; any other drainer error
        # is irrelevant once the track is closing.
        except (asyncio.CancelledError, Exception):
            pass


class RoomClient:
    """
    Subscribe-only LiveKit room participant.

    Feeds decoded media into a ConnectorEndpoint so the hub receives it via IPC.
    """

    def __init__(self, cfg: LiveKitConnectorConfig, ep: ConnectorEndpoint) -> None:
        self._cfg  = cfg
        self._ep   = ep
        self._room = rtc.Room()
        # track SID → streaming task; lets us cancel exactly the right task on unsubscribe.
        self._track_tasks: dict[str, asyncio.Task] = {}
        # Tasks spawned by sync event callbacks; cancelled on disconnect().
        self._pending_tasks: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        # Per-participant return audio: pid → (AudioSource, LocalTrackPublication, ReturnPipe).
        # Lazy-published on first send_return_audio for a pid; subscribe permissions
        # restrict each track so only the target participant can hear it.
        # The pipe paces audio into LiveKit at audio rate, so flush_return_audio
        # can drop in-flight TTS instantly even after a burst of chunks.
        self._return_audio: dict[
            str, tuple[rtc.AudioSource, rtc.LocalTrackPublication, _ReturnAudioPipe]
        ] = {}

        # ── room event handlers ───────────────────────────────────────────────

        @self._room.on("participant_connected")
        def _on_joined(participant: rtc.RemoteParticipant) -> None:
            self._spawn(self._handle_joined(participant))

        @self._room.on("participant_disconnected")
        def _on_left(participant: rtc.RemoteParticipant) -> None:
            self._spawn(self._handle_left(participant))

        @self._room.on("track_subscribed")
        def _on_track(
            track: rtc.Track,
            _pub: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            self._maybe_start_track(track, participant.identity)

        @self._room.on("track_unsubscribed")
        def _on_track_end(
            track: rtc.Track,
            _pub: rtc.RemoteTrackPublication,
            _participant: rtc.RemoteParticipant,
        ) -> None:
            self._cancel_track_task(track.sid)

        @self._room.on("data_received")
        def _on_data(packet: rtc.DataPacket) -> None:
            if packet.participant is None:
                return
            self._spawn(
                self._ep.push_data(
                    DataMessage(
                        participant_id=packet.participant.identity,
                        topic=packet.topic or "",
                        pts_us=_now_us(),
                        data=packet.data,
                    )
                )
            )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        await self._room.connect(
            self._cfg.lk_internal_url,
            make_client_token(self._cfg, identity=self._cfg.identity),
            options=rtc.RoomOptions(auto_subscribe=True, connect_timeout=15.0),
        )
        logger.info(
            "Room client connected: url={}  room={!r}  identity={!r}",
            self._cfg.lk_internal_url, self._cfg.room_name, self._cfg.identity,
        )

        # Notify IPC about participants already in the room when we joined.
        for participant in self._room.remote_participants.values():
            await self._handle_joined(participant)
            for pub in participant.track_publications.values():
                if pub.track is not None and pub.subscribed:
                    self._maybe_start_track(pub.track, participant.identity)

    def _maybe_start_track(self, track: rtc.Track, identity: str) -> None:
        """Start a stream task for a video/audio track; ignore other kinds."""
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self._start_track_task(
                track.sid, self._stream_video(track, identity, track.sid),
            )
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            self._start_track_task(
                track.sid, self._stream_audio(track, identity, track.sid),
            )

    async def run(self) -> None:
        """Wait until stop() is called."""
        await self._stop.wait()

    def stop(self) -> None:
        self._stop.set()

    def _start_track_task(self, sid: str, coro) -> None:
        # Cancel any existing task for this SID before starting a new one.
        self._cancel_track_task(sid)
        self._track_tasks[sid] = asyncio.create_task(coro, name=f"track-{sid}")

    def _cancel_track_task(self, sid: str) -> None:
        t = self._track_tasks.pop(sid, None)
        if t and not t.done():
            t.cancel()

    def _spawn(self, coro) -> None:
        """Track a fire-and-forget task so disconnect() can cancel orphans."""
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        task.add_done_callback(self._log_task_exception)

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        # Surface failures in push_data/join/leave handlers instead of letting
        # them stay silent until disconnect retrieves the exception.
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.opt(exception=exc).error("spawned room-client task failed")

    async def disconnect(self) -> None:
        for t in self._track_tasks.values():
            t.cancel()
        await asyncio.gather(*self._track_tasks.values(), return_exceptions=True)
        self._track_tasks.clear()
        pending = list(self._pending_tasks)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._pending_tasks.clear()
        # Close pacing pipes before dropping the entries so drainer tasks exit cleanly.
        await asyncio.gather(
            *(pipe.close() for _src, _pub, pipe in self._return_audio.values()),
            return_exceptions=True,
        )
        self._return_audio.clear()
        await self._room.disconnect()

    async def send_return_data(self, msg: DataMessage) -> None:
        """Publish data to the target participant via LiveKit data channel."""
        try:
            await self._room.local_participant.publish_data(
                msg.data,
                reliable=True,
                topic=msg.topic or "",
                destination_identities=[msg.participant_id],
            )
        except Exception:
            logger.exception("send_return_data failed")

    async def send_return_audio(self, chunk: AudioChunk) -> None:
        """Hand a return-audio chunk to the participant's pacing pipe.

        Non-blocking: the pipe absorbs the chunk and a background task
        feeds it into LiveKit at audio rate.  Keeps the connector's
        recv loop responsive so flush messages are not stuck FIFO
        behind a burst of chunks.
        """
        pid   = chunk.participant_id
        entry = self._return_audio.get(pid)
        if entry is None:
            entry = await self._publish_return_track(pid, chunk.sample_rate, chunk.channels)
            self._return_audio[pid] = entry
            self._refresh_return_track_permissions()
        _src, _pub, pipe = entry

        pcm_f32 = np.frombuffer(chunk.data, dtype=np.float32)
        pcm_i16 = (np.clip(pcm_f32, -1.0, 1.0) * 32767).astype(np.int16)
        frame = rtc.AudioFrame(
            data=pcm_i16.tobytes(),
            samples_per_channel=chunk.samples,
            sample_rate=chunk.sample_rate,
            num_channels=chunk.channels,
        )
        pipe.push(frame)

    async def flush_return_audio(self, flush: ReturnAudioFlush) -> None:
        """Drop every audio frame currently buffered for *flush.participant_id*.

        Clears both the pacing-pipe queue and LiveKit's internal queue;
        only the client's jitter buffer (~100 ms) plays out afterwards.
        """
        entry = self._return_audio.get(flush.participant_id)
        if entry is None:
            return
        _src, _pub, pipe = entry
        pipe.flush()

    async def _publish_return_track(
        self, pid: str, sample_rate: int, channels: int,
    ) -> tuple[rtc.AudioSource, rtc.LocalTrackPublication, _ReturnAudioPipe]:
        src   = rtc.AudioSource(sample_rate=sample_rate, num_channels=channels)
        track = rtc.LocalAudioTrack.create_audio_track(f"xr-hub-return-{pid}", src)
        pub   = await self._room.local_participant.publish_track(track)
        pipe  = _ReturnAudioPipe(src)
        logger.info("Return audio track published: pid={!r}  sid={!r}", pid, pub.sid)
        return src, pub, pipe

    def _refresh_return_track_permissions(self) -> None:
        """
        Each participant may subscribe only to their own return track.
        Recomputed whenever the per-pid track set changes.
        """
        perms = [
            rtc.ParticipantTrackPermission(
                participant_identity=pid,
                allow_all=False,
                allowed_track_sids=[pub.sid],
            )
            for pid, (_src, pub, _pipe) in self._return_audio.items()
        ]
        self._room.local_participant.set_track_subscription_permissions(
            allow_all_participants=False,
            participant_permissions=perms,
        )

    # ── participant events ────────────────────────────────────────────────────

    async def _handle_joined(self, participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant joined: {!r}", participant.identity)
        await self._ep.notify_participant_joined(participant.identity, _now_us())

    async def _handle_left(self, participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant left: {!r}", participant.identity)
        await self._ep.notify_participant_left(participant.identity, _now_us())
        entry = self._return_audio.pop(participant.identity, None)
        if entry is not None:
            _src, pub, pipe = entry
            await pipe.close()
            try:
                await self._room.local_participant.unpublish_track(pub.sid)
            except Exception:
                logger.exception("unpublish_track failed for {!r}", participant.identity)
            self._refresh_return_track_permissions()

    # ── media streams ─────────────────────────────────────────────────────────

    async def _stream_video(
        self, track: rtc.Track, identity: str, track_id: str
    ) -> None:
        logger.info("Video stream started: participant={!r}  track={!r}", identity, track_id)
        video_stream = rtc.VideoStream(track, format=rtc.VideoBufferType.I420)
        try:
            async for event in video_stream:
                frame = event.frame
                try:
                    await self._ep.push_frame(
                        data=bytes(frame.data),
                        width=frame.width,
                        height=frame.height,
                        fmt=PixelFormat.I420,
                        pts_us=_now_us(),
                        participant_id=identity,
                        track_id=track_id,
                    )
                except RuntimeError:
                    logger.warning(
                        "Ring buffer full — dropped frame from {!r}/{!r}",
                        identity, track_id,
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "Video stream error: participant={!r}  track={!r}", identity, track_id,
            )
        finally:
            logger.info(
                "Video stream ended: participant={!r}  track={!r}", identity, track_id,
            )
            await video_stream.aclose()

    async def _stream_audio(
        self, track: rtc.Track, identity: str, track_id: str
    ) -> None:
        logger.info("Audio stream started: participant={!r}  track={!r}", identity, track_id)
        audio_stream = rtc.AudioStream(track)
        try:
            async for event in audio_stream:
                frame = event.frame
                # LiveKit delivers int16 PCM; AudioChunk expects float32 LE interleaved.
                pcm_f32 = (
                    np.frombuffer(bytes(frame.data), dtype=np.int16)
                    .astype(np.float32)
                    / 32768.0
                )
                await self._ep.push_audio(
                    AudioChunk(
                        pts_us=_now_us(),
                        sample_rate=frame.sample_rate,
                        channels=frame.num_channels,
                        samples=frame.samples_per_channel,
                        data=pcm_f32.tobytes(),
                        participant_id=identity,
                        track_id=track_id,
                    )
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "Audio stream error: participant={!r}  track={!r}", identity, track_id,
            )
        finally:
            logger.info(
                "Audio stream ended: participant={!r}  track={!r}", identity, track_id,
            )
            await audio_stream.aclose()
