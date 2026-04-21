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
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from livekit import rtc

from xr_media_hub.ipc._connector import ConnectorEndpoint
from xr_media_hub.ipc._types import AudioChunk, DataMessage, PixelFormat

from ._token import make_client_token
from .config import LiveKitConnectorConfig

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def _now_us() -> int:
    return time.time_ns() // 1_000


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
        self._stop = asyncio.Event()

        # ── room event handlers ───────────────────────────────────────────────

        @self._room.on("participant_connected")
        def _on_joined(participant: rtc.RemoteParticipant) -> None:
            asyncio.ensure_future(self._handle_joined(participant))

        @self._room.on("participant_disconnected")
        def _on_left(participant: rtc.RemoteParticipant) -> None:
            asyncio.ensure_future(self._handle_left(participant))

        @self._room.on("track_subscribed")
        def _on_track(
            track: rtc.Track,
            _pub: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._start_track_task(
                    track.sid,
                    self._stream_video(track, participant.identity, track.sid),
                )
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                self._start_track_task(
                    track.sid,
                    self._stream_audio(track, participant.identity, track.sid),
                )

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
            asyncio.ensure_future(
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
        log.info(
            "Room client connected: url=%s  room=%r  identity=%r",
            self._cfg.lk_internal_url, self._cfg.room_name, self._cfg.identity,
        )

        # Notify IPC about participants already in the room when we joined.
        for participant in self._room.remote_participants.values():
            await self._handle_joined(participant)
            for pub in participant.track_publications.values():
                if pub.track is not None and pub.subscribed:
                    if pub.track.kind == rtc.TrackKind.KIND_VIDEO:
                        self._start_track_task(
                            pub.track.sid,
                            self._stream_video(
                                pub.track, participant.identity, pub.track.sid
                            ),
                        )
                    elif pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                        self._start_track_task(
                            pub.track.sid,
                            self._stream_audio(
                                pub.track, participant.identity, pub.track.sid
                            ),
                        )

    async def run(self) -> None:
        """Wait until stop() is called."""
        await self._stop.wait()

    def stop(self) -> None:
        self._stop.set()

    def _start_track_task(self, sid: str, coro) -> None:
        # Cancel any existing task for this SID before starting a new one.
        self._cancel_track_task(sid)
        self._track_tasks[sid] = asyncio.ensure_future(coro)

    def _cancel_track_task(self, sid: str) -> None:
        t = self._track_tasks.pop(sid, None)
        if t and not t.done():
            t.cancel()

    async def disconnect(self) -> None:
        for t in self._track_tasks.values():
            t.cancel()
        await asyncio.gather(*self._track_tasks.values(), return_exceptions=True)
        self._track_tasks.clear()
        await self._room.disconnect()

    # ── participant events ────────────────────────────────────────────────────

    async def _handle_joined(self, participant: rtc.RemoteParticipant) -> None:
        log.info("Participant joined: %r", participant.identity)
        await self._ep.notify_participant_joined(participant.identity, _now_us())

    async def _handle_left(self, participant: rtc.RemoteParticipant) -> None:
        log.info("Participant left: %r", participant.identity)
        await self._ep.notify_participant_left(participant.identity, _now_us())

    # ── media streams ─────────────────────────────────────────────────────────

    async def _stream_video(
        self, track: rtc.Track, identity: str, track_id: str
    ) -> None:
        log.info("Video stream started: participant=%r  track=%r", identity, track_id)
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
                    log.warning(
                        "Ring buffer full — dropped frame from %r/%r", identity, track_id
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Video stream error: participant=%r  track=%r", identity, track_id)
        finally:
            log.info("Video stream ended: participant=%r  track=%r", identity, track_id)
            await video_stream.aclose()

    async def _stream_audio(
        self, track: rtc.Track, identity: str, track_id: str
    ) -> None:
        log.info("Audio stream started: participant=%r  track=%r", identity, track_id)
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
            log.exception("Audio stream error: participant=%r  track=%r", identity, track_id)
        finally:
            log.info("Audio stream ended: participant=%r  track=%r", identity, track_id)
            await audio_stream.aclose()
