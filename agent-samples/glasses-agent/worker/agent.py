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
from memory import AgentMemory, Observation, TranscriptClient
from processors import QueryProcessor
from vad import VadDetector

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"

log = logging.getLogger("glasses_agent.agent")

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
        stt_url:       str,
        tts_url:       str,
        vlm_mcp_url:   str,
        video_mcp_url: str,
    ) -> None:
        self._cfg               = cfg
        self._memory            = memory
        self._transcript        = transcript_client
        self._qproc             = query_processor
        self._stt_url           = stt_url.rstrip("/")
        self._tts_url           = tts_url.rstrip("/")
        self._vlm_mcp_url       = vlm_mcp_url.rstrip("/")
        self._video_mcp_url     = video_mcp_url.rstrip("/")

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
        """Continuously caption frames, focusing on what changed since last caption."""
        import os
        log.info("background VLM loop started  gap=%.1fs", self._cfg.vlm_interval_s)
        last_description: str = ""
        last_frame_ts:    int = 0
        skipped: int = 0
        while True:
            try:
                if not self._user_query_active and self._participants:
                    pid = self._active_pid()
                    obs = await self._observe_frame(pid, previous=last_description,
                                                    skip_ts=last_frame_ts)
                    if obs:
                        last_description = obs.description
                        last_frame_ts    = obs.timestamp_us
                        skipped          = 0
                        self._memory.add_observation(obs)
                        log.info("\n  ┌─ observation ─────────────────────────────\n"
                                 "  │ %s\n"
                                 "  └───────────────────────────────────────────",
                                 obs.description)
                        asyncio.create_task(
                            self._transcript.add_entry(
                                self._cfg.transcript_source + ":observations",
                                obs.timestamp_us,
                                obs.description,
                            )
                        )
                    else:
                        skipped += 1
                        if skipped % 30 == 0:  # every ~30s at 1s gap
                            log.info("[vlm] observing — no change in last %ds",
                                     skipped)
                # Short gap — VLM inference already takes ~1 s; this just prevents
                # a hot loop when get_latest_frame returns quickly (no participant).
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
        try:
            frame_data = await self._call_mcp_http(
                self._video_mcp_url + "/mcp",
                "get_latest_frame",
                {"participant_id": pid},
            )
        except Exception as exc:
            log.error("get_latest_frame failed: %s", exc)
            return None

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
            # Strip any prior comparison language so the context stays clean.
            prev_clean = previous.split("compared to")[0].strip().rstrip(",")
            question = (
                f"Last action: {prev_clean}\n\n"
                "You are logging procedure steps. Look at this new image and describe "
                "what the person just DID — focus on the ACTION, not the scene.\n"
                "Use the format: [verb] [object] [location/direction]. Examples:\n"
                "  'picks up red cup from center of desk'\n"
                "  'places cup on scale to the left'\n"
                "  'presses blue button on device'\n"
                "  'rotates object clockwise'\n"
                "  'opens drawer on right side'\n"
                "Look for: hands moving, objects picked up/put down/moved/pressed/turned, "
                "posture changes, gaze shifts, new objects entering view.\n"
                "• If there is a new action: one short action phrase.\n"
                "• If nothing changed: respond with exactly: unchanged"
            )
        else:
            question = (
                "Describe what the person is doing as a procedure step. "
                "Format: [verb] [object] [location]. "
                "Example: 'sits at desk facing monitor'. "
                "Focus on their action, not the background."
            )

        try:
            description = await self._call_mcp_http(
                self._vlm_mcp_url + "/mcp",
                "ask_image",
                {"question": question, "image_path": frame_path},
            )
        except Exception as exc:
            log.error("ask_image failed: %s", exc)
            return None

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

    async def _call_mcp_http(
        self,
        mcp_url: str,
        tool:    str,
        args:    dict,
    ) -> dict | str | None:
        """Call a FastMCP tool via StreamableHTTP transport."""
        from fastmcp import Client as McpClient
        try:
            async with McpClient(mcp_url) as client:
                result = await client.call_tool(tool, args)
            items = getattr(result, "content", None) or []
            if items and hasattr(items[0], "text"):
                try:
                    return json.loads(items[0].text)
                except Exception:
                    return items[0].text
            return None
        except Exception as exc:
            log.error("mcp %s failed: %s", tool, exc)
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
