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
import json as _json
import logging
import os
import re as _re
import shutil as _shutil
import time
import wave

import httpx
import numpy as np

from xr_ai_agent import (
    AudioChunk, DataMessage, ParticipantEvent, ProcessorEndpoint
)

from config import WorkerConfig
from intent import is_real_assistant_request, is_shape_noise
from memory import AgentMemory, Observation, RecordedFrame, TranscriptClient
from nat_runtime import NatRuntime
from processors import QueryProcessor
from vad import VadDetector

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"

log        = logging.getLogger("glasses_agent_nat.agent")
_trace_log = logging.getLogger("glasses_agent_nat.trace")  # shared with glasses_agent_nat_worker

_DEFAULT_PID = "web-client"


def _now_us() -> int:
    return time.time_ns() // 1_000


def _chunks_to_wav(int16_pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw int16 PCM bytes in a WAV container for the STT server.

    ``xr_ai_vad`` emits int16 PCM directly via on_utterance, so the bytes
    can be wrapped as-is.
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
    nat_runtime:       Shared NAT workflow/runtime for MCP-backed tools.
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
        nat_runtime:  NatRuntime,
    ) -> None:
        self._cfg               = cfg
        self._memory            = memory
        self._transcript        = transcript_client
        self._qproc             = query_processor
        self._stt_url           = stt_url.rstrip("/")
        self._tts_url           = tts_url.rstrip("/")
        self._nat_runtime       = nat_runtime

        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

        # Per-participant VAD detectors.
        self._vad: dict[str, VadDetector] = {}

        # Track connected participants.
        self._participants: set[str] = set()

        # Background tasks.
        self._bg_task:        asyncio.Task | None = None
        self._condenser_task: asyncio.Task | None = None

        # In-flight query task + dispatch lock per pid. The lock serialises
        # cancel-await-flush-launch so a rapid second utterance can't race
        # past an in-flight cancellation and clobber generation bookkeeping.
        self._query_tasks:    dict[str, asyncio.Task] = {}
        self._dispatch_locks: dict[str, asyncio.Lock] = {}

        # Per-pid TTS generation token. flush_audio() bumps it; say() checks
        # it inside its chunk-send loop and stops emitting once the value
        # has moved on. Without this, an in-progress say() keeps queuing
        # chunks after the hub queue has been flushed, so flush + new
        # response still bleed through.
        self._speech_generation: dict[str, int] = {}

        # Background VLM loop pauses while a user query is being dispatched
        # so we don't compete with the user-facing path for VLM bandwidth.
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
        """Feed inbound mic audio through xr-ai-vad.

        Note: we do NOT pre-filter audio based on "assistant talking" flags.
        Pre-filtering breaks barge-in. The two-stage gate (speech_start
        flushes queued TTS, then post-STT noise/intent classifier decides
        whether to cancel the in-flight task) lives in
        ``_handle_speech_start`` and ``_handle_utterance``.
        """
        pid = chunk.participant_id or _DEFAULT_PID
        vad = self._get_vad(pid)
        # xr_ai_vad expects float32 → int16 conversion. Hub AudioChunks
        # are float32, so convert here.
        f32 = np.frombuffer(chunk.data, dtype=np.float32)
        i16 = (np.clip(f32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        await vad.feed(i16, chunk.sample_rate)

    def _get_vad(self, pid: str) -> VadDetector:
        if pid not in self._vad:
            self._vad[pid] = VadDetector(
                on_utterance     = lambda pcm, sr: self._handle_utterance(pid, pcm, sr),
                on_speech_start  = lambda: self._handle_speech_start(pid),
                silence_duration = self._cfg.silence_duration,
                min_speech       = self._cfg.min_speech,
                silero_threshold = self._cfg.silero_threshold,
            )
        return self._vad[pid]

    async def _handle_speech_start(self, pid: str) -> None:
        """Stage 1 of barge-in: flush queued TTS the moment speech is detected.

        We deliberately do NOT cancel the in-flight query here — VAD fires
        on background noise too, and cancelling on every silero spike would
        kill legitimate responses. Cancellation happens after STT + the
        noise gate confirm a real utterance, in ``_handle_utterance``.
        """
        await self.flush_audio(pid)

    async def _handle_utterance(self, pid: str, pcm_bytes: bytes, sample_rate: int) -> None:
        """STT-transcribe a finalized utterance, then dispatch to QueryProcessor.

        Voice path only — this is where the layered noise/intent gate runs.
        Data-channel text bypasses both layers (deliberate: the client
        explicitly sent it, so the user means it).
        """
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

        # ── Layer 1: shape gate — cheap, synchronous. ─────────────────────
        if is_shape_noise(text):
            log.info("noise-gate L1 dropped  pid=%r  %r", pid, text[:80])
            _trace_log.info("NOISE_L1_DROP  pid=%s  %r", pid, text)
            return

        # ── Layer 2: LLM intent classifier. ───────────────────────────────
        # Skip when worker is recording or guiding (those modes have their
        # own narration policy) and when a fresh demo is on the table
        # (post-demo utterances are almost always guidance requests, even
        # if mangled). Layer 2 is fail-open: classifier error → accept.
        skip_l2 = (
            self._memory.recording is not None
            or self._qproc.is_guiding(pid)
            or self._memory.demo_is_fresh(self._cfg.guidance_freshness_window_s)
        )
        if not skip_l2:
            try:
                accepted = await is_real_assistant_request(
                    self._http, self._cfg.llm_server, text
                )
            except Exception:
                log.exception("intent classifier raised — fail-open")
                accepted = True
            if not accepted:
                log.info("noise-gate L2 dropped  pid=%r  %r", pid, text[:80])
                _trace_log.info("NOISE_L2_DROP  pid=%s  %r", pid, text)
                return

        await self._dispatch_query(pid, text, ref_us=ref_us, source="voice")

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
        await self._dispatch_query(pid, text, ref_us=ref_us, source="data")

    # ── interruption / dispatch ──────────────────────────────────────────────

    async def _dispatch_query(
        self,
        pid: str,
        text: str,
        *,
        ref_us: int,
        source: str,
    ) -> None:
        """Cancel any in-flight query for *pid*, flush queued audio, dispatch a new one.

        The per-pid lock serialises cancel→await→flush→launch so a rapid
        second utterance can't race past an in-flight cancellation and
        clobber the bookkeeping.
        """
        lock = self._dispatch_locks.setdefault(pid, asyncio.Lock())
        async with lock:
            old = self._query_tasks.get(pid)
            current = asyncio.current_task()
            if old is not None and not old.done() and old is not current:
                log.info("interrupt pid=%r — cancelling in-flight response", pid)
                old.cancel()
                try:
                    await old
                except (asyncio.CancelledError, Exception):
                    pass
            await self.flush_audio(pid)
            task = asyncio.create_task(
                self._run_query(pid, text, ref_us=ref_us, source=source),
                name=f"glasses-nat-query-{pid}",
            )
            self._query_tasks[pid] = task

    async def _run_query(
        self,
        pid: str,
        text: str,
        *,
        ref_us: int,
        source: str,
    ) -> None:
        self._user_query_active = True
        try:
            await self._qproc.handle(text, pid, ref_us)
        except asyncio.CancelledError:
            log.info("query cancelled  pid=%r  source=%s", pid, source)
            raise
        except Exception:
            log.exception("query failed pid=%r source=%s", pid, source)
        finally:
            self._user_query_active = False

    async def flush_audio(self, pid: str) -> None:
        """Drop any TTS audio still queued for *pid* and signal a generation bump.

        Bumping ``_speech_generation`` makes the in-progress ``say()`` loop
        stop emitting further chunks (it checks the generation on every
        iteration). Without that guard, ``say()`` keeps queueing chunks
        after the hub queue has been flushed and stale audio bleeds through.
        """
        self._speech_generation[pid] = self._speech_generation.get(pid, 0) + 1
        try:
            await self._ep.flush_return_audio(pid)
        except Exception:
            log.exception("flush_return_audio failed  pid=%r", pid)

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

    def _active_pid(self) -> str:
        """Return the most recently active participant, or the default."""
        if self._participants:
            return next(iter(self._participants))
        return _DEFAULT_PID

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
                pid          = self._active_pid()
                is_recording = self._memory.recording is not None

                if is_recording and not was_recording:
                    self._rec_last_ts    = 0
                    self._rec_warmup_end = _now_us() + 2_000_000  # 2s warmup
                was_recording = is_recording

                # Recording capture is not gated by _user_query_active —
                # voice notes count as part of the demonstration and must not
                # create gaps in the recorded-frame timeline.
                if is_recording:
                    if _now_us() < self._rec_warmup_end:
                        pass  # skip first 2 s — user is still positioning
                    else:
                        frame = await self._capture_recording_frame(
                            pid, skip_ts=self._rec_last_ts
                        )
                        if frame:
                            self._rec_last_ts = frame.timestamp_us
                            self._memory.add_recorded_frame(frame)
                            _trace_log.info("REC_FRAME  %d  %s",
                                            frame.frame_idx + 1, frame.description[:80])
                elif not self._user_query_active and not self._qproc.is_guiding(pid):
                    # Normal change-detection observations are bandwidth-
                    # sensitive — yield while user-facing guidance owns VLM checks.
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
        frame_idx_now = len(recording.recorded_frames)
        run_dir_now   = os.environ.get("XR_RUN_DIR", "/tmp")
        safe_now      = _re.sub(r"[^a-zA-Z0-9_-]", "_", recording.name)[:40]
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

        # ask_image returns a str, but the NAT runtime may JSON-decode it into
        # {"result": "..."} if the content happens to be valid JSON.
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

    async def _call_vlm(self, tool: str, args: dict) -> dict | str | None:
        """Call a vlm-mcp tool through NAT."""
        try:
            return await self._nat_runtime.call_tool("vlm_mcp", tool, args)
        except Exception as exc:
            log.error("vlm-mcp %s failed: %s", tool, exc)
            return None

    async def _call_video(self, tool: str, args: dict) -> dict | str | None:
        """Call a video-mcp tool through NAT."""
        try:
            return await self._nat_runtime.call_tool("video_mcp", tool, args)
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
        """Condense the last 20 observations through the NAT worker task group."""
        recent = list(self._memory._observations)[-20:]
        if not recent:
            return

        try:
            structured = await asyncio.wait_for(
                self._nat_runtime.call_tool(
                    "glasses_worker_tasks",
                    "condense_observations",
                    {
                        "observations": [
                            {
                                "timestamp_us": o.timestamp_us,
                                "description": o.description,
                            }
                            for o in recent
                        ]
                    },
                ),
                timeout=20.0,
            )
            if not isinstance(structured, dict):
                return

            overview = structured.get("overview", "").strip()
            events   = structured.get("events", [])
            summary_text = structured.get("summary_text", "").strip()

            if not overview and not summary_text:
                return

            if not summary_text:
                lines = [overview]
                for ev in events:
                    ts_us = ev.get("timestamp_us", 0)
                    hms   = ev.get("time", "")
                    desc  = ev.get("description", "")
                    lines.append(f"  [{hms} | {ts_us} us] {desc}")
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
        """TTS → send audio back to *pid*.

        Snapshots the per-pid generation token before the chunk-send loop;
        ``flush_audio()`` bumps the token, and the loop bails out the moment
        it observes a newer value. This is what makes barge-in actually
        feel instant — the hub queue is flushed, AND the producer stops
        producing.
        """
        my_gen = self._speech_generation.get(pid, 0)
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
                if self._speech_generation.get(pid, 0) != my_gen:
                    log.debug("say pid=%r — generation bumped, stopping chunk send", pid)
                    return
                await self._ep.send_return_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("tts error pid=%r: %s", pid, exc, exc_info=True)
