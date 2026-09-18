# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe participant capture bundles backed by NVENC and PCM files."""
from __future__ import annotations

import base64
import json
import os
import shutil
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from xr_ai_hub import AudioChunk, DataMessage, FrameData

from ._frontends import _RawCaptureFrontend
from ._video import _H264TrackWriter, _join_sources, _safe_name, _video_sources
from .config import CaptureConfig

_CAPTURE_MARKER_NAME = ".xr-ai-media-capture"
_CAPTURE_MARKER_CONTENT = "xr-ai media capture session v1\n"


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
            payload = {"data_base64": base64.b64encode(message.data).decode("ascii")}
        self._event(
            session,
            "data",
            message.pts_us,
            direction=direction,
            topic=message.topic,
            **payload,
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
        raw_frame = writer.write(frame, "", ())
        if raw_frame is None:
            with session.lock:
                session.dropped_video_frames += 1
            return
        raw_width, raw_height, raw_segment_path = raw_frame
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
        if writer is not None:
            writer.close()

    def end_session(
        self,
        participant_id: str,
        pts_us: int,
        incomplete_reason: str | None = None,
    ) -> Path | None:
        with self._lock:
            session = self._sessions.pop(participant_id, None)
        if session is None:
            return None
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
            session.conversation.close()
            video_tracks = self._mux_video(session)
            for stream in session.raw_audio.values():
                stream.close()
            session.video_index.close()
            session.audio_index.close()
            session.transcripts.close()
            session.observations.close()
            session.events.close()
            manifest = {
                "version": 3,
                "participant_id": participant_id,
                "profile": self._config.profile,
                "capture_profile": "raw",
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
                "renderings": {},
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
        return session.root

    def _mux_video(
        self,
        session: _ParticipantSession,
    ) -> dict[str, list[dict]]:
        raw_sources = _video_sources(session.video)
        if not raw_sources:
            return {}
        raw_path = session.root / "video" / "session.264"
        raw_packets = _join_sources(session.root, raw_sources, raw_path)
        source_dimensions = sorted({
            (segment["width"], segment["height"])
            for _, segment in raw_sources
        })
        encoded_dimensions = sorted({
            (segment["width"], segment["height"])
            for _, segment in raw_sources
        })
        display_segment = max(
            (segment for _, segment in raw_sources),
            key=lambda segment: segment["width"] * segment["height"],
        )
        track_ids = list(dict.fromkeys(track_id for track_id, _ in raw_sources))
        packet_index_path = session.root / "video" / "packets.jsonl"
        with packet_index_path.open("w", encoding="utf-8") as packet_index:
            for packet_id, indexed in enumerate(raw_packets, start=1):
                packet = indexed.packet
                _write_json_line(packet_index, {
                    "packet_id": packet_id,
                    "track_id": indexed.track_id,
                    "offset": packet.offset,
                    "size_bytes": packet.size,
                    "pts_us": packet.pts_us,
                    "relative_us": packet.pts_us - session.start_us,
                    "key_frame": packet.key_frame,
                })
        combined = {
            "start_us": min(segment["start_us"] for _, segment in raw_sources),
            "end_us": max(segment["end_us"] for _, segment in raw_sources),
            "num_frames": sum(segment["num_frames"] for _, segment in raw_sources),
            "width": display_segment["width"],
            "height": display_segment["height"],
            "fps": self._config.sample_fps,
            "raw_path": "video/session.264",
            "raw_size_bytes": raw_path.stat().st_size,
            "packet_index": "video/packets.jsonl",
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

        combined["path"] = combined["raw_path"]
        combined["size_bytes"] = combined["raw_size_bytes"]
        combined["audio_embedded"] = False

        manifest_track = track_ids[0] if len(track_ids) == 1 else "session"
        return {manifest_track: [combined]}

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

    def close(self) -> list[Path]:
        end_us = time.time_ns() // 1_000
        completed = []
        for participant_id in list(self._sessions):
            root = self.end_session(participant_id, end_us)
            if root is not None:
                completed.append(root)
        return completed

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
