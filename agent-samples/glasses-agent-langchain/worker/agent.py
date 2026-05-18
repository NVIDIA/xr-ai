# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GlassesAgent — always-on AI assistant for smart glasses.

Lifecycle
---------
1. Connects to the XR hub via ProcessorEndpoint.
2. Launches background tasks:
   - _background_vlm_loop: every cfg.vlm_interval_s seconds, grabs a frame,
     runs a brief VLM description, adds to AgentMemory.
   - _memory_condenser_loop: every condenser_interval_s seconds, condenses
     recent observations into a scene summary via LLM.
3. Audio from participants is routed through per-participant VadDetector.
4. When an utterance finalizes: STT → QueryProcessor.handle().
5. Data messages are passed directly to QueryProcessor as text queries.
6. Responses go back as TTS audio + data messages to the participant.

Default participant
-------------------
If no participant has joined yet, the "web-client" default is used for
background VLM calls and initial responses.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import wave

import httpx
import numpy as np

from xr_ai_agent import (
    AudioChunk, DataMessage, FrameSignal, ParticipantEvent, ProcessorEndpoint
)

from config import WorkerConfig
from memory import AgentMemory, Observation, RecordedFrame, TranscriptClient
from processors import QueryProcessor
from vad import VadDetector

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"

log        = logging.getLogger("glasses_agent_langchain.agent")
_trace_log = logging.getLogger("glasses_agent_langchain.trace")  # shared with glasses_agent_langchain_worker

_DEFAULT_PID = "web-client"


def _now_us() -> int:
    return time.time_ns() // 1_000


def _chunks_to_wav(int16_pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw int16 PCM bytes in a WAV container for the STT server.

    VadDetector already converts float32 → int16 before calling on_utterance,
    so no further conversion is needed here.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_pcm)
    return buf.getvalue()


def _wav_to_chunks(wav_bytes: bytes, participant_id: str) -> list[AudioChunk]:
    """Decode WAV blob into 20 ms float32 AudioChunks for the return path."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_frames = max(1, sr // 50)  # 20 ms
    pts = _now_us()
    out: list[AudioChunk] = []
    for i in range(0, len(arr), chunk_frames * ch):
        seg = arr[i : i + chunk_frames * ch]
        if not len(seg):
            break
        out.append(AudioChunk(
            pts_us=pts, sample_rate=sr, channels=ch,
            samples=len(seg) // ch, data=seg.tobytes(),
            participant_id=participant_id,
        ))
        pts += 20_000
    return out


class GlassesAgent:
    """Always-on smart-glasses AI agent.

    Parameters
    ----------
    cfg:               WorkerConfig
    memory:            AgentMemory — shared observation + demo store.
    transcript_client: TranscriptClient — persists observations to transcript-mcp.
    query_processor:   QueryProcessor — handles transcribed utterances.
    stt_url:           STT server base URL (OpenAI-compatible /v1/audio/transcriptions).
    tts_url:           TTS server base URL (OpenAI-compatible /v1/audio/speech).
    vlm_mcp_url:       vlm-mcp base URL — used by the background VLM loop.
    video_mcp_url:     video-mcp base URL — used by the background VLM loop.
    """

    def __init__(
        self,
        cfg:               WorkerConfig,
        memory:            AgentMemory,
        transcript_client: TranscriptClient,
        query_processor:   QueryProcessor,
        *,
        stt_url:      str,
        tts_url:      str,
        vlm_client,         # fastmcp.Client already open (shared with QueryProcessor)
        video_client,       # fastmcp.Client already open (shared with QueryProcessor)
    ) -> None:
        self._cfg               = cfg
        self._memory            = memory
        self._transcript        = transcript_client
        self._qproc             = query_processor
        self._stt_url           = stt_url.rstrip("/")
        self._tts_url           = tts_url.rstrip("/")
        self._vlm_client        = vlm_client
        self._video_client      = video_client

        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)
        self._ep.on_frame(self._on_frame)

        # Per-participant VAD detectors.
        self._vad: dict[str, VadDetector] = {}

        # Latest FrameSignals per (pid, track_id).
        self._latest: dict[tuple[str, str], FrameSignal] = {}

        # Track connected participants.
        self._participants: set[str] = set()

        # Background tasks.
        self._bg_task:        asyncio.Task | None = None
        self._condenser_task: asyncio.Task | None = None

        # Flag to pause background VLM loop when a user query is in flight.
        self._user_query_active = False

        # Last observed frame timestamp and description (observation loop).
        # Instance variables so they can be reset on participant reconnect.
        self._obs_last_ts:   int = 0
        self._obs_last_desc: str = ""

        # Last frame timestamp captured during recording (skip-dedup).
        self._rec_last_ts:    int = 0
        self._rec_warmup_end: int = 0

        # Shared HTTP client for STT / TTS / LLM calls.
        self._http = httpx.AsyncClient(timeout=120.0)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Connect to hub, start background tasks, run until stopped."""
        self._bg_task        = asyncio.create_task(
            self._background_vlm_loop(), name="glasses-bg-vlm"
        )
        self._condenser_task = asyncio.create_task(
            self._memory_condenser_loop(), name="glasses-condenser"
        )
        try:
            await self._ep.run()
        finally:
            for task in (self._bg_task, self._condenser_task):
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()

    # ── audio path: VAD → STT → QueryProcessor ───────────────────────────────

    async def _on_audio(self, chunk: AudioChunk) -> None:
        pid = chunk.participant_id or _DEFAULT_PID
        vad = self._get_vad(pid)
        await vad.feed(chunk.data, chunk.sample_rate, chunk.samples)

    def _get_vad(self, pid: str) -> VadDetector:
        if pid not in self._vad:
            self._vad[pid] = VadDetector(
                on_utterance    = lambda pcm, sr: self._handle_utterance(pid, pcm, sr),
                silence_threshold = self._cfg.silence_threshold,
                silence_duration  = self._cfg.silence_duration,
                min_speech        = self._cfg.min_speech,
                silero_threshold  = self._cfg.silero_threshold,
                vad_noise_mult    = self._cfg.vad_noise_mult,
            )
        return self._vad[pid]

    async def _handle_utterance(self, pid: str, pcm_bytes: bytes, sample_rate: int) -> None:
        """STT-transcribe a finalized utterance, then dispatch to QueryProcessor."""
        ref_us = _now_us()
        wav    = _chunks_to_wav(pcm_bytes, sample_rate)
        try:
            resp = await self._http.post(
                self._stt_url + "/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"response_format": "json"},
            )
            if resp.is_error:
                log.error("stt %s: %s", resp.status_code, resp.text[:200])
                return
            text = resp.json().get("text", "").strip()
        except Exception as exc:
            log.error("stt request failed pid=%r: %s", pid, exc)
            return

        if not text:
            log.info("stt returned empty for pid=%r", pid)
            return

        log.info("stt  pid=%r  %r", pid, text[:80])
        self._user_query_active = True
        try:
            await self._qproc.handle(text, pid, ref_us)
        finally:
            self._user_query_active = False

    # ── data path: text → QueryProcessor ─────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        pid    = msg.participant_id or _DEFAULT_PID
        ref_us = msg.pts_us or _now_us()
        log.info("data query  pid=%r  %r", pid, text[:80])
        self._user_query_active = True
        try:
            await self._qproc.handle(text, pid, ref_us)
        finally:
            self._user_query_active = False

    # ── participant tracking ──────────────────────────────────────────────────

    async def _on_participant(self, event: ParticipantEvent) -> None:
        pid = event.participant_id
        if event.joined:
            log.info("participant joined  pid=%r", pid)
            self._participants.add(pid)
            # Reset observation state so the new stream is processed fresh.
            self._obs_last_ts   = 0
            self._obs_last_desc = ""
        else:
            log.info("participant left  pid=%r", pid)
            self._participants.discard(pid)
            self._vad.pop(pid, None)
            for k in [k for k in self._latest if k[0] == pid]:
                del self._latest[k]

    async def _on_frame(self, sig: FrameSignal) -> None:
        self._latest[(sig.participant_id, sig.track_id)] = sig

    def _active_pid(self) -> str:
        """Return the most recently active participant, or the default."""
        if self._participants:
            return next(iter(self._participants))
        return _DEFAULT_PID

    def _latest_signal(self, pid: str) -> FrameSignal | None:
        candidates = [v for k, v in self._latest.items() if k[0] == pid]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.pts_us)

    # ── background VLM loop ───────────────────────────────────────────────────

    async def _background_vlm_loop(self) -> None:
        """Observe the scene every vlm_interval_s seconds.

        Recording — dense VLM capture, every frame saved to RecordedFrame buffer
                    and to a per-demo JSONL log on disk. No filtering or dedup —
                    analysis happens after recording stops.

        Normal    — change-detection VLM for the observation/memory timeline.
        """
        log.info("background VLM loop started  gap=%.1fs", self._cfg.vlm_interval_s)
        skipped:          int  = 0
        was_recording:    bool = False
        while True:
            try:
                # Run even when self._participants is empty: the client may have
                # connected before the worker started and the join event was missed.
                # _active_pid() falls back to _DEFAULT_PID ("web-client") so the
                # VLM call still works; get_latest_frame returns an error dict if
                # no stream is active, which is handled gracefully.
                if not self._user_query_active:
                    pid = self._active_pid()
                    is_recording = self._memory.recording is not None

                    if is_recording and not was_recording:
                        self._rec_last_ts    = 0
                        self._rec_warmup_end = _now_us() + 2_000_000  # 2s warmup
                    was_recording = is_recording

                    if is_recording and _now_us() < self._rec_warmup_end:
                        pass  # skip first 2 s — user is still positioning
                    elif is_recording:
                        # ── Dense recording capture ───────────────────────────
                        frame = await self._capture_recording_frame(
                            pid, skip_ts=self._rec_last_ts
                        )
                        if frame:
                            self._rec_last_ts = frame.timestamp_us
                            self._memory.add_recorded_frame(frame)
                            _trace_log.info("REC_FRAME  %d  %s",
                                            frame.frame_idx + 1, frame.description[:80])
                    else:
                        # ── Normal change-detection for memory/context ────────
                        obs = await self._observe_frame(pid, previous=self._obs_last_desc,
                                                        skip_ts=self._obs_last_ts)
                        if obs:
                            self._obs_last_desc = obs.description
                            self._obs_last_ts   = obs.timestamp_us
                            skipped             = 0
                            self._memory.add_observation(obs)
                            _trace_log.info("OBS  %s", obs.description)
                            log.info(
                                "\n  ┌─ observation ──────────────────────────────\n"
                                "  │ %s\n"
                                "  └────────────────────────────────────────────",
                                obs.description,
                            )
                            asyncio.create_task(
                                self._transcript.add_entry(
                                    self._cfg.transcript_source + ":observations",
                                    obs.timestamp_us,
                                    obs.description,
                                )
                            )
                        else:
                            skipped += 1
                            if skipped % 30 == 0:
                                log.info("[vlm] observing — no change in last %ds", skipped)

                await asyncio.sleep(self._cfg.vlm_interval_s)
            except asyncio.CancelledError:
                log.info("background VLM loop cancelled")
                return
            except Exception:
                log.exception("background VLM loop error")

    _UNCHANGED_RESPONSES = frozenset({
        "unchanged", "no change", "nothing new", "same", "no changes",
        "nothing has changed", "nothing changed", "no significant change",
    })

    async def _observe_frame(self, pid: str, previous: str = "",
                             skip_ts: int = 0) -> Observation | None:
        """Get the latest frame and caption what is NEW compared to *previous*."""
        import os
        frame_data = await self._call_video("get_latest_frame", {"participant_id": pid})
        if not isinstance(frame_data, dict) or "path" not in frame_data:
            return None
        frame_path = frame_data["path"]
        frame_ts   = int(frame_data.get("timestamp_us", _now_us()))

        if not frame_path or not os.path.isfile(frame_path):
            return None

        # Same frame as last time — hub hasn't delivered a new one yet.
        if skip_ts and frame_ts == skip_ts:
            return None

        if previous:
            prev_clean = previous.split("compared to")[0].strip().rstrip(",")
            question = (
                f"Previous observation: {prev_clean}\n\n"
                "Compare this new frame to the previous observation. "
                "Has something visibly changed — a new action, movement, or object state?\n"
                "If yes: one short phrase describing the change. "
                "Include WHERE in the frame (left/right/center) and position "
                "relative to nearby objects if relevant.\n"
                "If nothing changed: respond with exactly: unchanged"
            )
        else:
            question = (
                "In one short phrase, describe the most notable action or movement "
                "happening in this frame. Include spatial position if relevant "
                "(left/right/center of frame, above/below/beside what object)."
            )

        description = await self._call_vlm(
            "ask_image", {"question": question, "image_path": frame_path}
        )
        if isinstance(description, dict):
            description = (description.get("result") or description.get("text")
                          or next(iter(description.values()), ""))
        if not isinstance(description, str) or not description.strip():
            return None
        desc = description.strip()
        if desc.lower().rstrip(".!") in self._UNCHANGED_RESPONSES:
            return None
        # Dedup: if the model regenerated the same text as before, skip.
        if previous and desc == previous:
            return None

        return Observation(
            timestamp_us = frame_ts,
            description  = description.strip(),
            image_path   = frame_path,
        )

    async def _capture_recording_frame(
        self, pid: str, skip_ts: int = 0
    ) -> RecordedFrame | None:
        """Capture one frame's detailed VLM analysis during recording.

        Returns a RecordedFrame with a rich 2-4 sentence description, or None
        if no new frame is available. Also appends to the per-demo JSONL log.
        No dedup, no filtering — everything is captured; analysis happens later.
        """
        import os, json as _json, re as _re
        frame_data = await self._call_video("get_latest_frame", {"participant_id": pid})
        if not isinstance(frame_data, dict) or "path" not in frame_data:
            _trace_log.info("REC_NO_FRAME  %s", str(frame_data)[:120] if frame_data else "None")
            return None
        frame_path = frame_data["path"]
        frame_ts   = int(frame_data.get("timestamp_us", _now_us()))
        if not frame_path or not os.path.isfile(frame_path):
            _trace_log.info("REC_NO_FILE  path=%s", frame_path)
            return None
        if skip_ts and frame_ts == skip_ts:
            return None  # same frame as last cycle — normal, no log needed

        # Get recording context before the VLM call (race-safe: check again after).
        recording = self._memory.recording
        if recording is None:
            return None

        # Copy frame to a stable path so it persists after get_latest_frame
        # overwrites the temp file on the next cycle.
        import shutil as _shutil
        frame_idx_now = len(recording.recorded_frames)
        run_dir_now   = os.environ.get("XR_RUN_DIR", "/tmp")
        safe_now      = __import__("re").sub(r"[^a-zA-Z0-9_-]", "_", recording.name)[:40]
        frames_dir    = os.path.join(run_dir_now, "recordings",
                                     f"{safe_now}_{recording.started_at_us}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        stable_path = os.path.join(frames_dir, f"frame_{frame_idx_now:04d}.png")
        try:
            _shutil.copy2(frame_path, stable_path)
            frame_path = stable_path
        except Exception as exc:
            log.warning("frame copy failed: %s", exc)

        # Build context from demo name + most recent voice note.
        ctx = f"The user is recording a demonstration of: {recording.name!r}."
        last_note = recording.voice_notes[-1].text if recording.voice_notes else ""
        if last_note:
            ctx += f'\n They just said: "{last_note}".'

        question = (
            f"{ctx}\n\n"
            "Describe ONLY what you observe in this image right now. "
            "Do not predict, suggest, or describe what should happen next.\n"
            "Include:\n"
            "- Which hand(s) are active and the exact action being performed\n"
            "- What object is being held or touched (describe by color, shape, size "
            "if its name is uncertain)\n"
            "- Precise spatial position: left/right/center, above/below/beside which "
            "other visible object\n"
            "2-3 factual sentences. Describe the current moment only."
        )
        description = await self._call_vlm(
            "ask_image", {"question": question, "image_path": frame_path}
        )

        # ask_image returns a str, but _parse_mcp_result may JSON-decode it
        # into {"result": "..."} if the content happens to be valid JSON.
        if isinstance(description, dict):
            description = (description.get("result") or description.get("text")
                          or next(iter(description.values()), ""))
        if not isinstance(description, str) or not description.strip():
            _trace_log.info("REC_NO_DESC  vlm returned empty  raw=%s", str(description)[:80])
            return None

        # Re-check in case recording stopped while the VLM call was in flight.
        if self._memory.recording is None:
            _trace_log.info("REC_RACE  recording stopped mid-capture")
            return None

        frame_idx = len(recording.recorded_frames)
        desc      = description.strip()

        # Write to per-demo JSONL log (one JSON per line, crash-safe).
        run_dir  = os.environ.get("XR_RUN_DIR", "/tmp")
        safe     = _re.sub(r"[^a-zA-Z0-9_-]", "_", recording.name)[:40]
        log_dir  = os.path.join(run_dir, "recordings")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{safe}_{recording.started_at_us}.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps({
                    "frame_idx":  frame_idx,
                    "ts_us":      frame_ts,
                    "image_path": frame_path,
                    "description": desc,
                }) + "\n")
        except Exception as exc:
            log.warning("recording log write failed: %s", exc)

        return RecordedFrame(
            frame_idx    = frame_idx,
            timestamp_us = frame_ts,
            image_path   = frame_path,
            description  = desc,
        )

    def _parse_mcp_result(self, result) -> dict | str | None:
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured
        items = getattr(result, "content", None) or []
        if items and hasattr(items[0], "text"):
            try:
                return json.loads(items[0].text)
            except Exception:
                return items[0].text
        return None

    async def _call_vlm(self, tool: str, args: dict) -> dict | str | None:
        """Call a vlm-mcp tool using the persistent client."""
        if self._vlm_client is None:
            return None
        try:
            return self._parse_mcp_result(
                await self._vlm_client.call_tool(tool, args)
            )
        except Exception as exc:
            log.error("vlm-mcp %s failed: %s", tool, exc)
            return None

    async def _call_video(self, tool: str, args: dict) -> dict | str | None:
        """Call a video-mcp tool using the persistent client."""
        if self._video_client is None:
            return None
        try:
            return self._parse_mcp_result(
                await self._video_client.call_tool(tool, args)
            )
        except Exception as exc:
            log.error("video-mcp %s failed: %s", tool, exc)
            return None

    # ── memory condenser loop ─────────────────────────────────────────────────

    async def _memory_condenser_loop(self) -> None:
        """Every cfg.condenser_interval_s seconds, condense observations into a summary."""
        log.info("memory condenser started  interval=%.1fs", self._cfg.condenser_interval_s)
        while True:
            try:
                await asyncio.sleep(self._cfg.condenser_interval_s)
                await self._condense_observations()
            except asyncio.CancelledError:
                log.info("memory condenser cancelled")
                return
            except Exception:
                log.exception("memory condenser error")

    async def _condense_observations(self) -> None:
        """Condense the last 20 observations into a timestamped structured summary."""
        import json as _json
        recent = list(self._memory._observations)[-20:]
        if not recent:
            return

        # Include timestamp_us so the LLM can echo them back in the JSON output.
        obs_text = "\n".join(
            f"  [{time.strftime('%H:%M:%S', time.localtime(o.timestamp_us / 1e6))}|{o.timestamp_us}]  "
            f"{o.description}"
            for o in recent
        )

        body = {
            "model": "llm",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a scene context summarizer for smart glasses. "
                        "Given a timeline of camera observations (each tagged with "
                        "[HH:MM:SS|timestamp_us]), output ONLY valid JSON in this shape:\n"
                        '{"overview":"1-2 sentence scene summary","events":['
                        '{"timestamp_us":<int>,"time":"HH:MM:SS","description":"brief event"}]}\n'
                        "Include only the 3-6 most significant events. "
                        "Use the exact timestamp_us values from the input."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Observations:\n{obs_text}",
                },
            ],
            "max_tokens": 256,
            "temperature": 0.1,
        }
        try:
            resp = await asyncio.wait_for(
                self._http.post(
                    self._cfg.llm_server.rstrip("/") + "/v1/chat/completions",
                    json=body,
                ),
                timeout=20.0,
            )
            if resp.is_error:
                log.warning("condenser LLM %d: %s", resp.status_code, resp.text[:200])
                return

            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if the model wrapped the JSON.
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()

            try:
                structured = _json.loads(raw)
            except Exception:
                # Fallback: treat as plain text summary (no timestamps).
                log.warning("condenser output not JSON — storing as plain text")
                structured = {"overview": raw, "events": []}

            overview = structured.get("overview", "").strip()
            events   = structured.get("events", [])

            if not overview:
                return

            # Build the formatted text that goes into the LLM context.
            lines = [overview]
            for ev in events:
                ts_us = ev.get("timestamp_us", 0)
                hms   = ev.get("time", "")
                desc  = ev.get("description", "")
                lines.append(f"  [{hms} | {ts_us} µs] {desc}")
            summary_text = "\n".join(lines)

            self._memory.update_scene_summary(summary_text)
            _trace_log.info("CONDENSER  %s", summary_text.replace("\n", " | "))

            # Persist the full structured JSON so video-mcp can be queried by timestamp.
            ts = _now_us()
            asyncio.create_task(
                self._transcript.add_entry(
                    self._cfg.transcript_source + ":scene_summary",
                    ts,
                    _json.dumps(structured),
                )
            )

            log.info("[scene summary]\n%s", summary_text)
        except Exception:
            log.exception("condenser LLM call failed")

    # ── response helpers ──────────────────────────────────────────────────────

    async def send_text(self, pid: str, text: str, topic: str) -> None:
        """Send a text data message to *pid*."""
        try:
            await self._ep.send_return_data(DataMessage(
                participant_id = pid,
                topic          = topic,
                pts_us         = _now_us(),
                data           = text.encode(),
            ))
        except Exception:
            log.exception("send_text failed  pid=%r  topic=%s", pid, topic)

    async def say(self, pid: str, text: str) -> None:
        """TTS → send audio back to participant."""
        try:
            resp = await self._http.post(
                self._tts_url + "/v1/audio/speech",
                json={"input": text, "response_format": "wav"},
                timeout=60.0,
            )
            if resp.is_error:
                log.error("tts %s: %s", resp.status_code, resp.text[:200])
                return
            wav = resp.content
            for chunk in _wav_to_chunks(wav, pid):
                await self._ep.send_return_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("tts error pid=%r: %s", pid, exc, exc_info=True)
