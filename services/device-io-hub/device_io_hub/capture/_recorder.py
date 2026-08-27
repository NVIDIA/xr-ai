# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe participant capture bundles backed by NVENC and PCM files."""
from __future__ import annotations

import base64
import json
import shutil
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from xr_ai_hub import AudioChunk, DataMessage, FrameData

from ._compositor import compose_caption
from ._matroska import VideoPacket, mux_h264_pcm
from .config import CaptureConfig


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return cleaned or "unnamed"


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
        config: CaptureConfig,
    ) -> None:
        self._nvc = nvc
        self._root = root
        self._track_id = track_id
        self._config = config
        self._encoder = None
        self._stream = None
        self._width = 0
        self._height = 0
        self._last_pts_us = 0
        self._segment_index = 0
        self._active: dict | None = None
        self._packets: list[VideoPacket] = []
        self.segments: list[dict] = []

    def write(self, frame: FrameData, caption: str) -> None:
        min_interval_us = round(1_000_000 / self._config.sample_fps)
        if self._last_pts_us and frame.pts_us - self._last_pts_us < min_interval_us:
            return
        nv12, width, height = compose_caption(
            frame,
            caption,
            max_lines=self._config.overlay_lines,
        )
        if self._encoder is None or (width, height) != (self._width, self._height):
            self._start_segment(width, height, frame.pts_us)
        picture_params = self._nvc.NV_ENC_PIC_PARAMS()
        picture_params.inputTimeStamp = frame.pts_us
        for packet in _encoded_packets(self._encoder.Encode(nv12, picture_params)):
            self._write_packet(packet, fallback_pts_us=frame.pts_us)
        self._last_pts_us = frame.pts_us
        self._active["end_us"] = frame.pts_us
        self._active["num_frames"] += 1

    def _start_segment(self, width: int, height: int, pts_us: int) -> None:
        self._finish_segment()
        self._width = width
        self._height = height
        name = f"{_safe_name(self._track_id)}_{self._segment_index:03d}.264"
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

    def _write_packet(self, packet: dict, *, fallback_pts_us: int) -> None:
        payload = packet["data"]
        offset = self._stream.tell()
        self._stream.write(payload)
        picture_type = int(packet.get("picture_type", 0))
        self._packets.append(
            VideoPacket(
                offset=offset,
                size=len(payload),
                pts_us=int(packet.get("timestamp", fallback_pts_us)),
                key_frame=picture_type in (2, 3),
            )
        )

    def _finish_segment(self) -> None:
        if self._encoder is None:
            return
        try:
            for packet in _encoded_packets(self._encoder.EndEncode()):
                self._write_packet(packet, fallback_pts_us=self._last_pts_us)
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

    def close(self) -> None:
        self._finish_segment()


@dataclass
class _ParticipantSession:
    participant_id: str
    start_us: int
    root: Path
    events: object
    audio_index: object
    raw_audio: dict[str, object]
    conversation: _StereoWaveWriter
    video: dict[str, _H264TrackWriter] = field(default_factory=dict)
    caption: str = ""
    caption_expires_us: int = 0
    dropped_video_frames: int = 0
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
        self._root = Path(config.out_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, _ParticipantSession] = {}
        self._lock = threading.RLock()

    def begin_session(self, participant_id: str, pts_us: int) -> None:
        with self._lock:
            if participant_id in self._sessions:
                return
            start_us = max(1, pts_us)
            base = self._root / f"{start_us}_{_safe_name(participant_id)}"
            root = base
            suffix = 2
            while root.exists():
                root = Path(f"{base}_{suffix}")
                suffix += 1
            (root / "video").mkdir(parents=True)
            (root / "audio").mkdir()
            events = (root / "events.jsonl").open("a", encoding="utf-8")
            audio_index = (root / "audio" / "chunks.jsonl").open("a", encoding="utf-8")
            raw_audio = {
                direction: (root / "audio" / f"{direction}.f32le").open("ab")
                for direction in ("device", "agent")
            }
            session = _ParticipantSession(
                participant_id=participant_id,
                start_us=start_us,
                root=root,
                events=events,
                audio_index=audio_index,
                raw_audio=raw_audio,
                conversation=_StereoWaveWriter(
                    root / "audio" / "conversation.wav",
                    start_us=start_us,
                    sample_rate=self._config.audio_sample_rate,
                ),
            )
            self._sessions[participant_id] = session
        self._event(session, "participant", start_us, state="joined")
        logger.info("media capture started participant={!r} path={}", participant_id, root)

    def _session(self, participant_id: str, pts_us: int) -> _ParticipantSession:
        with self._lock:
            session = self._sessions.get(participant_id)
        if session is None:
            self.begin_session(participant_id, pts_us)
            with self._lock:
                session = self._sessions[participant_id]
        return session

    def record_audio(self, direction: str, chunk: AudioChunk) -> None:
        session = self._session(chunk.participant_id, chunk.pts_us)
        with session.lock:
            stream = session.raw_audio[direction]
            offset = stream.tell()
            stream.write(chunk.data)
            stream.flush()
            _write_json_line(session.audio_index, {
                "direction": direction,
                "pts_us": chunk.pts_us,
                "sample_rate": chunk.sample_rate,
                "channels": chunk.channels,
                "samples": chunk.samples,
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
        if direction == "agent" and message.topic in self._config.overlay_topics:
            caption = self._caption_text(text)
            with session.lock:
                session.caption = f"AGENT: {caption}" if caption else ""
                session.caption_expires_us = (
                    message.pts_us + round(self._config.overlay_seconds * 1_000_000)
                )

    @staticmethod
    def _caption_text(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        try:
            value = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return stripped
        if isinstance(value, dict):
            for key in ("text", "message", "content", "title"):
                if isinstance(value.get(key), str):
                    return value[key].strip()
        if isinstance(value, str):
            return value.strip()
        return stripped

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
                    config=self._config,
                )
                session.video[frame.track_id] = writer
            caption = "" if frame.pts_us > session.caption_expires_us else session.caption
        writer.write(frame, caption)

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

    def end_session(self, participant_id: str, pts_us: int) -> None:
        with self._lock:
            session = self._sessions.pop(participant_id, None)
        if session is None:
            return
        pts_us = max(session.start_us, pts_us)
        with session.lock:
            self._event(session, "participant", pts_us, state="left")
            for writer in session.video.values():
                writer.close()
            session.conversation.close()
            self._mux_video(session, pts_us)
            for stream in session.raw_audio.values():
                stream.close()
            session.audio_index.close()
            session.events.close()
            manifest = {
                "version": 1,
                "participant_id": participant_id,
                "start_us": session.start_us,
                "end_us": pts_us,
                "video_tracks": {
                    track_id: writer.segments
                    for track_id, writer in session.video.items()
                },
                "audio": {
                    "conversation": "audio/conversation.wav",
                    "channels": {"left": "device", "right": "agent"},
                    "sample_rate": self._config.audio_sample_rate,
                    "raw_index": "audio/chunks.jsonl",
                    "device_raw": "audio/device.f32le",
                    "agent_raw": "audio/agent.f32le",
                },
                "events": "events.jsonl",
                "dropped_video_frames": session.dropped_video_frames,
            }
            (session.root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        logger.info("media capture completed participant={!r} path={}", participant_id, session.root)
        self._prune_completed()

    def _mux_video(self, session: _ParticipantSession, end_us: int) -> None:
        wave_path = session.root / "audio" / "conversation.wav"
        sample_rate = self._config.audio_sample_rate
        for writer in session.video.values():
            for index, segment in enumerate(writer.segments):
                raw_path = session.root / segment["path"]
                muxed_path = raw_path.with_suffix(".mkv")
                next_start_us = (
                    writer.segments[index + 1]["start_us"]
                    if index + 1 < len(writer.segments)
                    else end_us
                )
                start_frame = round(
                    (segment["start_us"] - session.start_us)
                    * sample_rate
                    / 1_000_000
                )
                end_frame = round(
                    (next_start_us - session.start_us)
                    * sample_rate
                    / 1_000_000
                )
                packets = segment.pop("_packets")
                try:
                    mux_h264_pcm(
                        output_path=muxed_path,
                        h264_path=raw_path,
                        packets=packets,
                        wave_path=wave_path,
                        audio_start_frame=start_frame,
                        audio_end_frame=end_frame,
                        width=segment["width"],
                        height=segment["height"],
                        fps=segment["fps"],
                    )
                except Exception as exc:
                    muxed_path.unlink(missing_ok=True)
                    logger.warning("media capture A/V mux failed path={}: {}", raw_path, exc)
                    segment["audio_embedded"] = False
                    continue
                segment["raw_path"] = segment["path"]
                segment["raw_size_bytes"] = segment["size_bytes"]
                segment["path"] = str(muxed_path.relative_to(session.root))
                segment["size_bytes"] = muxed_path.stat().st_size
                segment["audio_embedded"] = True

    def _event(self, session: _ParticipantSession, kind: str, pts_us: int, **fields) -> None:
        with session.lock:
            _write_json_line(session.events, {"kind": kind, "pts_us": pts_us, **fields})

    def close(self) -> None:
        end_us = time.time_ns() // 1_000
        for participant_id in list(self._sessions):
            self.end_session(participant_id, end_us)

    def _prune_completed(self) -> None:
        cap = self._config.max_total_bytes
        if cap <= 0:
            return
        completed = [
            path for path in self._root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        ]
        sizes = {
            path: sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            for path in completed
        }
        total = sum(sizes.values())
        for path in sorted(completed, key=lambda item: item.stat().st_mtime_ns):
            if total <= cap:
                break
            total -= sizes[path]
            shutil.rmtree(path)
            logger.info("media capture evicted completed session {}", path)
