# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe participant capture bundles backed by NVENC and PCM files."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from xr_ai_hub import AudioChunk, DataMessage, FrameData

from ._frontends import _CaptureFrontend, _make_frontend, _RawCaptureFrontend
from ._matroska import VideoPacket
from .config import CaptureConfig

_MAX_SAFE_NAME = 96


_MAX_DATA_FEED_TEXT = 1_024
_CAPTURE_MARKER_NAME = ".xr-ai-media-capture"
_CAPTURE_MARKER_CONTENT = "xr-ai media capture session v1\n"


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    if cleaned == value and 0 < len(cleaned) <= _MAX_SAFE_NAME:
        return cleaned
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    prefix = (cleaned or "unnamed")[:_MAX_SAFE_NAME - len(digest) - 1]
    return f"{prefix}_{digest}"


def _target_parts(target: str | None) -> tuple[str, ...]:
    """Validate a wrapper-owned namespace beneath the configured capture root."""
    if target is None:
        return ()
    if not target or target.startswith("/"):
        raise ValueError("capture target must be a non-empty relative path")
    parts = tuple(target.split("/"))
    if any(part in {".", ".."} or _safe_name(part) != part for part in parts):
        raise ValueError(
            "capture target components may contain only letters, numbers, '.', '-', and '_'"
        )
    return parts


def _write_json_line(stream, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def _encoded_packets(packets: object) -> list[dict]:
    if not isinstance(packets, list):
        raise TypeError(f"unexpected PyNvVideoCodec packet collection: {type(packets).__name__}")
    output: list[dict] = []
    for packet in packets:
        if not isinstance(packet, dict) or not isinstance(packet.get("data"), bytes):
            raise TypeError(f"unexpected PyNvVideoCodec encoded packet: {packet!r}")
        output.append(packet)
    return output


class _StereoWaveWriter:
    """Random-access stereo PCM writer aligned to the participant clock."""

    _HEADER_BYTES = 44
    # AudioChunk timestamps are wall-clock arrival times, not sample-clock
    # positions. Ignore scheduler jitter and short producer stalls, but retain
    # pauses long enough to represent a genuine break in a conversation.
    _REAL_GAP_US = 250_000

    def __init__(self, path: Path, *, start_us: int, sample_rate: int) -> None:
        self.path = path
        self.start_us = start_us
        self.sample_rate = sample_rate
        self._stream = path.open("w+b")
        self._stream.write(b"\0" * self._HEADER_BYTES)
        self._max_frames = 0
        self._timeline: dict[str, _AudioTimeline] = {}

    def add(self, direction: str, chunk: AudioChunk) -> None:
        channel_index = 0 if direction == "device" else 1
        if chunk.sample_rate <= 0:
            raise ValueError("audio sample_rate must be positive")
        source = np.frombuffer(chunk.data, dtype=np.float32)
        expected = chunk.samples * chunk.channels
        if source.size != expected or chunk.samples <= 0 or chunk.channels <= 0:
            raise ValueError(
                f"invalid audio chunk: expected {expected} samples, got {source.size}"
            )
        mono = source.reshape(chunk.samples, chunk.channels).mean(axis=1)
        if chunk.sample_rate != self.sample_rate:
            target_count = max(1, round(mono.size * self.sample_rate / chunk.sample_rate))
            if mono.size == 1:
                mono = np.repeat(mono, target_count)
            else:
                mono = np.interp(
                    np.linspace(0, mono.size - 1, target_count),
                    np.arange(mono.size),
                    mono,
                )
        pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2")
        timeline = self._timeline.get(direction)
        if timeline is None:
            frame_offset = max(
                0,
                round((chunk.pts_us - self.start_us) * self.sample_rate / 1_000_000),
            )
        else:
            frame_offset = timeline.next_frame
            gap_us = chunk.pts_us - timeline.source_end_us
            # Device timestamps are decoder-arrival times; even a long event-
            # loop stall can be followed by queued, sample-contiguous frames.
            # Agent chunks, however, stop between responses, so retain only
            # their clearly non-jitter-sized gaps.
            if direction == "agent" and gap_us >= self._REAL_GAP_US:
                frame_offset += round(gap_us * self.sample_rate / 1_000_000)
        source_duration_us = round(chunk.samples * 1_000_000 / chunk.sample_rate)
        self._timeline[direction] = _AudioTimeline(
            source_end_us=chunk.pts_us + source_duration_us,
            next_frame=frame_offset + pcm.size,
        )
        byte_offset = self._HEADER_BYTES + frame_offset * 4
        byte_count = pcm.size * 4
        self._stream.seek(0, 2)
        missing = byte_offset + byte_count - self._stream.tell()
        if missing > 0:
            # Extend with a sparse zero-filled gap. A paused session can put
            # the next chunk minutes later; allocating that entire gap as one
            # bytes object would make capture memory scale with silence.
            self._stream.seek(byte_offset + byte_count - 1)
            self._stream.write(b"\0")
        self._stream.seek(byte_offset)
        existing = self._stream.read(byte_count)
        stereo = np.frombuffer(existing, dtype="<i2").copy().reshape(-1, 2)
        stereo[:, channel_index] = pcm
        self._stream.seek(byte_offset)
        self._stream.write(stereo.tobytes())
        self._max_frames = max(self._max_frames, frame_offset + pcm.size)

    def close(self) -> None:
        data_bytes = self._max_frames * 4
        byte_rate = self.sample_rate * 4
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_bytes,
            b"WAVE",
            b"fmt ",
            16,
            1,
            2,
            self.sample_rate,
            byte_rate,
            4,
            16,
            b"data",
            data_bytes,
        )
        self._stream.seek(0)
        self._stream.write(header)
        self._stream.truncate(self._HEADER_BYTES + data_bytes)
        self._stream.close()


@dataclass(frozen=True, slots=True)
class _AudioTimeline:
    source_end_us: int
    next_frame: int


class _H264TrackWriter:
    """One NVENC stream per participant video track."""

    def __init__(
        self,
        *,
        nvc: object,
        root: Path,
        track_id: str,
        stream_name: str,
        config: CaptureConfig,
        frontend: _CaptureFrontend,
    ) -> None:
        self._nvc = nvc
        self._root = root
        self._track_id = track_id
        self._stream_name = stream_name
        self._config = config
        self._frontend = frontend
        self._encoder = None
        self._stream = None
        self._width = 0
        self._height = 0
        self._last_pts_us = 0
        self._segment_index = 0
        self._active: dict | None = None
        self._packets: list[VideoPacket] = []
        self._submitted_pts: deque[int] = deque()
        self.segments: list[dict] = []

    def write(
        self,
        frame: FrameData,
        caption: str,
        data_feed: tuple[str, ...],
    ) -> tuple[int, int, str] | None:
        min_interval_us = round(1_000_000 / self._config.sample_fps)
        if self._last_pts_us and frame.pts_us - self._last_pts_us < min_interval_us:
            return None
        rendered = self._frontend.render_frame(
            frame,
            caption=caption,
            data_feed=data_feed,
        )
        width = rendered.width
        height = rendered.height
        if self._encoder is None or (width, height) != (self._width, self._height):
            self._start_segment(width, height, frame.pts_us)
        picture_params = self._nvc.NV_ENC_PIC_PARAMS()
        picture_params.inputTimeStamp = frame.pts_us
        self._submitted_pts.append(frame.pts_us)
        for packet in _encoded_packets(self._encoder.Encode(rendered.pixels, picture_params)):
            self._write_packet(packet)
        self._last_pts_us = frame.pts_us
        self._active["end_us"] = frame.pts_us
        self._active["num_frames"] += 1
        return width, height, str(self._active["path"])

    def _start_segment(self, width: int, height: int, pts_us: int) -> None:
        self._finish_segment()
        self._width = width
        self._height = height
        name = f"{self._stream_name}_{_safe_name(self._track_id)}_{self._segment_index:03d}.264"
        self._segment_index += 1
        self._stream = (self._root / name).open("wb")
        try:
            self._encoder = self._nvc.CreateEncoder(
                width,
                height,
                "NV12",
                True,
                gpu_id=self._config.gpu_id,
                codec="h264",
                preset="P4",
                tuning_info="high_quality",
                rc="vbr",
                fps=int(round(self._config.sample_fps)),
                bitrate=self._config.bitrate,
                maxbitrate=self._config.bitrate,
                bf=0,
                repeat_sps_pps=1,
            )
        except Exception:
            self._stream.close()
            (self._root / name).unlink(missing_ok=True)
            self._stream = None
            raise
        self._active = {
            "path": f"video/{name}",
            "start_us": pts_us,
            "end_us": pts_us,
            "num_frames": 0,
            "width": width,
            "height": height,
            "fps": self._config.sample_fps,
        }
        self._packets = []
        self._submitted_pts.clear()

    def _write_packet(self, packet: dict) -> None:
        payload = packet["data"]
        offset = self._stream.tell()
        self._stream.write(payload)
        picture_type = int(packet.get("picture_type", 0))
        if not self._submitted_pts:
            raise ValueError("NVENC emitted more packets than submitted frames")
        self._packets.append(
            VideoPacket(
                offset=offset,
                size=len(payload),
                pts_us=self._submitted_pts.popleft(),
                key_frame=picture_type in (2, 3),
            )
        )

    def _finish_segment(self) -> None:
        if self._encoder is None:
            return
        try:
            for packet in _encoded_packets(self._encoder.EndEncode()):
                self._write_packet(packet)
            if self._submitted_pts:
                raise ValueError(
                    f"NVENC omitted {len(self._submitted_pts)} submitted frames"
                )
        except Exception as exc:
            logger.warning("media capture NVENC flush failed track={!r}: {}", self._track_id, exc)
        finally:
            self._stream.close()
            self._encoder = None
            self._stream = None
        if self._active is not None:
            path = self._root.parent / self._active["path"]
            self._active["size_bytes"] = path.stat().st_size
            self._active["_packets"] = self._packets
            self.segments.append(self._active)
            self._active = None
            self._packets = []
            self._submitted_pts.clear()

    def close(self) -> None:
        self._finish_segment()


@dataclass
class _ParticipantSession:
    participant_id: str
    start_us: int
    root: Path
    trigger: str
    target: str | None
    metadata: dict[str, Any]
    events: object
    transcripts: object
    observations: object
    video_index: object
    audio_index: object
    raw_audio: dict[str, object]
    conversation: _StereoWaveWriter
    video: dict[str, _H264TrackWriter] = field(default_factory=dict)
    projected_video: dict[str, _H264TrackWriter] = field(default_factory=dict)
    caption: str = ""
    caption_expires_us: int = 0
    data_feed: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    dropped_video_frames: int = 0
    event_count: int = 0
    transcript_count: int = 0
    observation_count: int = 0
    frame_count: int = 0
    audio_chunk_count: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionRecorder:
    """Own capture artifacts; callers may invoke methods from worker threads."""

    def __init__(self, config: CaptureConfig) -> None:
        try:
            import PyNvVideoCodec as nvc
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeError(f"PyNvVideoCodec is required for media capture: {exc}") from exc
        self._nvc = nvc
        self._config = config
        self._frontend = _make_frontend(
            profile=config.profile,
            overlay_lines=config.overlay_lines,
        )
        self._raw_frontend = _RawCaptureFrontend()
        self._root = Path(config.out_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, _ParticipantSession] = {}
        self._lock = threading.RLock()
        self._prune_artifacts()

    def has_session(self, participant_id: str) -> bool:
        with self._lock:
            return participant_id in self._sessions

    def begin_session(
        self,
        participant_id: str,
        pts_us: int,
        trigger: str = "participant",
        target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        target_parts = _target_parts(target)
        normalized_target = "/".join(target_parts) or None
        with self._lock:
            if participant_id in self._sessions:
                return
            start_us = max(1, pts_us)
            target_root = self._ensure_target_root(target_parts)
            base = target_root / f"{start_us}_{_safe_name(participant_id)}"
            root = base
            suffix = 2
            while root.exists():
                root = Path(f"{base}_{suffix}")
                suffix += 1
            root.mkdir(mode=0o700)
            (root / _CAPTURE_MARKER_NAME).write_text(
                _CAPTURE_MARKER_CONTENT,
                encoding="utf-8",
            )
            (root / "video").mkdir(mode=0o700)
            (root / "audio").mkdir(mode=0o700)
            events = (root / "events.jsonl").open("a", encoding="utf-8")
            transcripts = (root / "transcript.jsonl").open("a", encoding="utf-8")
            observations = (root / "observations.jsonl").open("a", encoding="utf-8")
            video_index = (root / "video" / "frames.jsonl").open("a", encoding="utf-8")
            audio_index = (root / "audio" / "chunks.jsonl").open("a", encoding="utf-8")
            raw_audio = {
                direction: (root / "audio" / f"{direction}.f32le").open("ab")
                for direction in ("device", "agent")
            }
            session = _ParticipantSession(
                participant_id=participant_id,
                start_us=start_us,
                root=root,
                trigger=trigger,
                target=normalized_target,
                metadata=dict(metadata or {}),
                events=events,
                transcripts=transcripts,
                observations=observations,
                video_index=video_index,
                audio_index=audio_index,
                raw_audio=raw_audio,
                conversation=_StereoWaveWriter(
                    root / "audio" / "conversation.wav",
                    start_us=start_us,
                    sample_rate=self._config.audio_sample_rate,
                ),
            )
            self._sessions[participant_id] = session
        self._event(session, "recording", start_us, state="started", trigger=trigger)
        logger.info("media capture started participant={!r} path={}", participant_id, root)

    def _ensure_target_root(self, parts: tuple[str, ...]) -> Path:
        target_root = self._root
        for part in parts:
            target_root = target_root / part
            try:
                target_root.mkdir(mode=0o700)
            except FileExistsError:
                if target_root.is_symlink() or not target_root.is_dir():
                    raise ValueError(
                        f"capture target component is not a directory: {part!r}"
                    ) from None
        return target_root

    def _session(self, participant_id: str, pts_us: int) -> _ParticipantSession:
        del pts_us
        with self._lock:
            session = self._sessions.get(participant_id)
        if session is None:
            raise RuntimeError(f"no active capture session for {participant_id!r}")
        return session

    def record_audio(self, direction: str, chunk: AudioChunk) -> None:
        session = self._session(chunk.participant_id, chunk.pts_us)
        with session.lock:
            stream = session.raw_audio[direction]
            offset = stream.tell()
            stream.write(chunk.data)
            stream.flush()
            session.audio_chunk_count += 1
            duration_us = round(chunk.samples * 1_000_000 / chunk.sample_rate)
            _write_json_line(session.audio_index, {
                "chunk_id": session.audio_chunk_count,
                "direction": direction,
                "pts_us": chunk.pts_us,
                "relative_us": chunk.pts_us - session.start_us,
                "end_pts_us": chunk.pts_us + duration_us,
                "duration_us": duration_us,
                "sample_rate": chunk.sample_rate,
                "channels": chunk.channels,
                "samples": chunk.samples,
                "sample_format": "float32_le",
                "offset": offset,
                "size_bytes": len(chunk.data),
                "track_id": chunk.track_id,
            })
            session.conversation.add(direction, chunk)

    def record_data(self, direction: str, message: DataMessage) -> None:
        session = self._session(message.participant_id, message.pts_us)
        try:
            text = message.data.decode("utf-8")
            payload = {"text": text}
        except UnicodeDecodeError:
            text = ""
            payload = {"data_base64": base64.b64encode(message.data).decode("ascii")}
        self._event(
            session,
            "data",
            message.pts_us,
            direction=direction,
            topic=message.topic,
            **payload,
        )
        if text.strip():
            normalized = " ".join(text.split())[:_MAX_DATA_FEED_TEXT]
            with session.lock:
                session.data_feed.append(
                    f"{direction.upper()} {message.topic}: {normalized}"
                )

    def record_voice_caption(self, source: str, message: DataMessage) -> None:
        session = self._session(message.participant_id, message.pts_us)
        text = message.data.decode("utf-8", errors="replace").strip()
        with session.lock:
            session.transcript_count += 1
            transcript = {
                "transcript_id": session.transcript_count,
                "pts_us": message.pts_us,
                "relative_us": message.pts_us - session.start_us,
                "source": source,
                "kind": "speech_recognition" if source == "user" else "speech_synthesis",
                "text": text,
            }
            _write_json_line(session.transcripts, transcript)
            self._event(
                session,
                "voice_caption",
                message.pts_us,
                source=source,
                transcript_id=session.transcript_count,
                text=text,
            )
            session.caption = f"{source.upper()}: {text}" if text else ""
            session.caption_expires_us = (
                max(message.pts_us, time.time_ns() // 1_000)
                + round(self._config.overlay_seconds * 1_000_000)
            )

    def record_observation(
        self,
        message: DataMessage,
        observation: dict[str, Any],
    ) -> None:
        session = self._session(message.participant_id, message.pts_us)
        frame_pts_us = observation.get("frame_pts_us")
        if not isinstance(frame_pts_us, int):
            frame_pts_us = None
        with session.lock:
            session.observation_count += 1
            record = {
                "observation_id": session.observation_count,
                "generated_at_us": message.pts_us,
                "generated_relative_us": message.pts_us - session.start_us,
                "frame_pts_us": frame_pts_us,
                "frame_relative_us": (
                    frame_pts_us - session.start_us
                    if frame_pts_us is not None
                    else None
                ),
                "observation": observation,
            }
            _write_json_line(session.observations, record)
            self._event(
                session,
                "observation",
                message.pts_us,
                observation_id=session.observation_count,
                frame_pts_us=frame_pts_us,
            )

    def record_flush(self, participant_id: str, pts_us: int) -> None:
        session = self._session(participant_id, pts_us)
        self._event(session, "audio_flush", pts_us, direction="agent")

    def caption_for(self, participant_id: str, pts_us: int) -> str:
        session = self._session(participant_id, pts_us)
        with session.lock:
            if pts_us > session.caption_expires_us:
                return ""
            return session.caption

    def record_video(self, frame: FrameData) -> None:
        session = self._session(frame.participant_id, frame.pts_us)
        with session.lock:
            writer = session.video.get(frame.track_id)
            if writer is None:
                writer = _H264TrackWriter(
                    nvc=self._nvc,
                    root=session.root / "video",
                    track_id=frame.track_id,
                    stream_name="raw",
                    config=self._config,
                    frontend=self._raw_frontend,
                )
                session.video[frame.track_id] = writer
            caption = "" if frame.pts_us > session.caption_expires_us else session.caption
            data_feed = tuple(session.data_feed)
            projected_writer = session.projected_video.get(frame.track_id)
            if self._frontend.name == "demo" and projected_writer is None:
                projected_writer = _H264TrackWriter(
                    nvc=self._nvc,
                    root=session.root / "video",
                    track_id=frame.track_id,
                    stream_name="demo",
                    config=self._config,
                    frontend=self._frontend,
                )
                session.projected_video[frame.track_id] = projected_writer
        raw_frame = writer.write(frame, "", ())
        if raw_frame is None:
            return
        raw_width, raw_height, raw_segment_path = raw_frame
        projected_frame = (
            projected_writer.write(frame, caption, data_feed)
            if projected_writer is not None
            else raw_frame
        )
        if projected_frame is None:
            raise RuntimeError("capture projections accepted different frame timestamps")
        projected_width, projected_height, projected_segment_path = projected_frame
        with session.lock:
            session.frame_count += 1
            _write_json_line(session.video_index, {
                "frame_id": session.frame_count,
                "pts_us": frame.pts_us,
                "relative_us": frame.pts_us - session.start_us,
                "sequence": frame.seq,
                "track_id": frame.track_id,
                "pixel_format": frame.fmt.name.lower(),
                "source_width": frame.width,
                "source_height": frame.height,
                "raw_width": raw_width,
                "raw_height": raw_height,
                "raw_encoder_segment_id": Path(raw_segment_path).stem,
                "projection_width": projected_width,
                "projection_height": projected_height,
                "projection_encoder_segment_id": Path(projected_segment_path).stem,
            })

    def note_video_drop(self, participant_id: str, pts_us: int) -> None:
        session = self._session(participant_id, pts_us)
        with session.lock:
            session.dropped_video_frames += 1

    def close_video_track(self, participant_id: str, track_id: str) -> None:
        with self._lock:
            session = self._sessions.get(participant_id)
        if session is None:
            return
        with session.lock:
            writer = session.video.get(track_id)
            projected_writer = session.projected_video.get(track_id)
        if writer is not None:
            writer.close()
        if projected_writer is not None:
            projected_writer.close()

    def end_session(
        self,
        participant_id: str,
        pts_us: int,
        incomplete_reason: str | None = None,
    ) -> None:
        with self._lock:
            session = self._sessions.pop(participant_id, None)
        if session is None:
            return
        pts_us = max(session.start_us, pts_us)
        with session.lock:
            complete = incomplete_reason is None
            self._event(
                session,
                "recording",
                pts_us,
                state="stopped",
                complete=complete,
                incomplete_reason=incomplete_reason,
            )
            for writer in session.video.values():
                writer.close()
            for writer in session.projected_video.values():
                writer.close()
            session.conversation.close()
            video_tracks = self._mux_video(session, pts_us)
            for stream in session.raw_audio.values():
                stream.close()
            session.video_index.close()
            session.audio_index.close()
            session.transcripts.close()
            session.observations.close()
            session.events.close()
            manifest = {
                "version": 2,
                "participant_id": participant_id,
                "profile": self._frontend.name,
                "session_mode": self._config.session_mode,
                "trigger": session.trigger,
                "target": session.target,
                "metadata": session.metadata,
                "complete": complete,
                "incomplete_reason": incomplete_reason,
                "start_us": session.start_us,
                "end_us": pts_us,
                "duration_us": pts_us - session.start_us,
                "clock": {
                    "timebase": "microseconds",
                    "origin": "unix_epoch",
                    "session_start_us": session.start_us,
                },
                "video_tracks": video_tracks,
                "video_frame_index": "video/frames.jsonl",
                "audio": {
                    "conversation": "audio/conversation.wav",
                    "channels": {"left": "device", "right": "agent"},
                    "sample_rate": self._config.audio_sample_rate,
                    "sample_format": "pcm_s16le",
                    "raw_index": "audio/chunks.jsonl",
                    "device_raw": "audio/device.f32le",
                    "agent_raw": "audio/agent.f32le",
                },
                "events": "events.jsonl",
                "transcript": "transcript.jsonl",
                "observations": "observations.jsonl",
                "counts": {
                    "video_frames": session.frame_count,
                    "audio_chunks": session.audio_chunk_count,
                    "transcripts": session.transcript_count,
                    "observations": session.observation_count,
                    "events": session.event_count,
                },
                "dropped_video_frames": session.dropped_video_frames,
            }
            manifest_path = session.root / "manifest.json"
            pending_manifest = manifest_path.with_suffix(".json.pending")
            try:
                pending_manifest.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                pending_manifest.replace(manifest_path)
            except Exception:
                pending_manifest.unlink(missing_ok=True)
                raise
        logger.info("media capture completed participant={!r} path={}", participant_id, session.root)
        self._prune_artifacts()

    def _mux_video(
        self,
        session: _ParticipantSession,
        end_us: int,
    ) -> dict[str, list[dict]]:
        raw_sources = self._video_sources(session.video)
        if not raw_sources:
            return {}
        raw_path = session.root / "video" / "session.264"
        raw_packets = self._join_sources(session.root, raw_sources, raw_path)
        projected_sources = self._video_sources(session.projected_video)
        display_sources = projected_sources or raw_sources
        projection_path: Path | None = None
        if projected_sources:
            projection_path = session.root / "video" / "projection.264"
            display_packets = self._join_sources(
                session.root,
                projected_sources,
                projection_path,
            )
        else:
            display_packets = raw_packets

        source_dimensions = sorted({
            (segment["width"], segment["height"])
            for _, segment in raw_sources
        })
        encoded_dimensions = sorted({
            (segment["width"], segment["height"])
            for _, segment in display_sources
        })
        display_segment = max(
            (segment for _, segment in display_sources),
            key=lambda segment: segment["width"] * segment["height"],
        )
        track_ids = list(dict.fromkeys(track_id for track_id, _ in display_sources))
        combined = {
            "start_us": min(segment["start_us"] for _, segment in display_sources),
            "end_us": max(segment["end_us"] for _, segment in display_sources),
            "num_frames": sum(segment["num_frames"] for _, segment in display_sources),
            "width": display_segment["width"],
            "height": display_segment["height"],
            "fps": self._config.sample_fps,
            "raw_path": "video/session.264",
            "raw_size_bytes": raw_path.stat().st_size,
            "source_track_ids": track_ids,
            "source_dimensions": [
                {"width": width, "height": height}
                for width, height in source_dimensions
            ],
            "encoded_dimensions": [
                {"width": width, "height": height}
                for width, height in encoded_dimensions
            ],
        }

        wave_path = session.root / "audio" / "conversation.wav"
        sample_rate = self._config.audio_sample_rate
        start_frame = round(
            (combined["start_us"] - session.start_us)
            * sample_rate
            / 1_000_000
        )
        end_frame = round(
            (end_us - session.start_us)
            * sample_rate
            / 1_000_000
        )
        try:
            artifact = self._frontend.finalize_video(
                session_root=session.root,
                raw_path=projection_path or raw_path,
                wave_path=wave_path,
                audio_start_frame=start_frame,
                audio_end_frame=end_frame,
                fps=combined["fps"],
                packets=display_packets,
                width=combined["width"],
                height=combined["height"],
            )
        except Exception as exc:
            logger.warning("media capture video finalization failed path={}: {}", raw_path, exc)
            combined["path"] = combined["raw_path"]
            combined["size_bytes"] = combined["raw_size_bytes"]
            combined["audio_embedded"] = False
        else:
            combined["path"] = artifact.path
            combined["size_bytes"] = artifact.size_bytes
            combined["audio_embedded"] = artifact.audio_embedded
        finally:
            if projection_path is not None:
                projection_path.unlink(missing_ok=True)

        manifest_track = track_ids[0] if len(track_ids) == 1 else "session"
        return {manifest_track: [combined]}

    @staticmethod
    def _video_sources(
        writers: dict[str, _H264TrackWriter],
    ) -> list[tuple[str, dict]]:
        return sorted(
            (
                (track_id, segment)
                for track_id, writer in writers.items()
                for segment in writer.segments
            ),
            key=lambda item: (item[1]["start_us"], item[0]),
        )

    @staticmethod
    def _join_sources(
        session_root: Path,
        sources: list[tuple[str, dict]],
        output_path: Path,
    ) -> list[VideoPacket]:
        pending_path = output_path.with_suffix(".264.pending")
        packets: list[VideoPacket] = []
        try:
            with pending_path.open("wb") as output:
                for _, segment in sources:
                    source_path = session_root / segment["path"]
                    offset = output.tell()
                    with source_path.open("rb") as source:
                        shutil.copyfileobj(source, output)
                    packets.extend(
                        VideoPacket(
                            offset=offset + packet.offset,
                            size=packet.size,
                            pts_us=packet.pts_us,
                            key_frame=packet.key_frame,
                        )
                        for packet in segment["_packets"]
                    )
            pending_path.replace(output_path)
        except Exception:
            pending_path.unlink(missing_ok=True)
            raise
        for _, segment in sources:
            (session_root / segment["path"]).unlink(missing_ok=True)
        return sorted(packets, key=lambda packet: packet.pts_us)

    def _event(self, session: _ParticipantSession, kind: str, pts_us: int, **fields) -> None:
        with session.lock:
            session.event_count += 1
            _write_json_line(session.events, {
                "event_id": session.event_count,
                "kind": kind,
                "pts_us": pts_us,
                "relative_us": pts_us - session.start_us,
                **fields,
            })

    def close(self) -> None:
        end_us = time.time_ns() // 1_000
        for participant_id in list(self._sessions):
            self.end_session(participant_id, end_us)

    def _prune_artifacts(self) -> None:
        cap = self._config.max_total_bytes
        if cap <= 0:
            return
        active = {session.root for session in self._sessions.values()}
        artifacts = []
        for directory, dirnames, filenames in os.walk(self._root, followlinks=False):
            parent = Path(directory)
            dirnames[:] = [
                name for name in dirnames if not (parent / name).is_symlink()
            ]
            if _CAPTURE_MARKER_NAME not in filenames:
                continue
            path = parent
            if path not in active and self._is_capture_artifact(path):
                artifacts.append(path)
        sizes = {
            path: sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            for path in artifacts
        }
        total = sum(sizes.values())
        for path in sorted(artifacts, key=lambda item: item.stat().st_mtime_ns):
            if total <= cap:
                break
            total -= sizes[path]
            shutil.rmtree(path)
            logger.info("media capture evicted session artifact {}", path)

    @staticmethod
    def _is_capture_artifact(path: Path) -> bool:
        marker = path / _CAPTURE_MARKER_NAME
        if not marker.is_file() or marker.is_symlink():
            return False
        try:
            return marker.read_text(encoding="utf-8") == _CAPTURE_MARKER_CONTENT
        except OSError:
            return False
