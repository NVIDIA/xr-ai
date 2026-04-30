# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
McpAgent — continuous STT → MCP transcript ingest.

Detects speech boundaries via VAD, runs final STT on each complete utterance,
and writes it to the transcript-mcp server via the ``transcript_add_transcript``
tool.  On any data-channel message, fetches transcript + video stats from MCP
and sends a summary back on topic ``mcp.stats``.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
from fastmcp import Client as McpClient

from xr_ai_agent import AudioChunk, DataMessage, ParticipantEvent, ProcessorEndpoint

from audio import chunks_to_wav, now_us, rms
from services import SttClient
from voice import VoiceState

log = logging.getLogger("mcp_agent")


class McpAgent:

    def __init__(
        self,
        ep:      ProcessorEndpoint,
        stt:     SttClient,
        mcp_url: str,
        *,
        silence_threshold: float = 0.01,
        silence_duration:  float = 0.8,
        min_speech:        float = 0.3,
    ) -> None:
        self._ep = ep
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

        self._stt     = stt
        self._mcp_url = mcp_url

        self._vad_threshold = silence_threshold
        self._vad_silence_s = silence_duration
        self._vad_min_s     = min_speech

        self._voice: dict[str, VoiceState] = {}

    # ── voice state ───────────────────────────────────────────────────────────

    def _get_voice(self, pid: str, sample_rate: int = 16000, channels: int = 1) -> VoiceState:
        if pid not in self._voice:
            self._voice[pid] = VoiceState(sample_rate=sample_rate, channels=channels)
        return self._voice[pid]

    # ── audio pipeline ────────────────────────────────────────────────────────

    async def _on_audio(self, chunk: AudioChunk) -> None:
        pid = chunk.participant_id
        vs  = self._get_voice(pid, chunk.sample_rate, chunk.channels)

        chunk_s = chunk.samples / max(chunk.sample_rate, 1)
        if rms(chunk.data) >= self._vad_threshold:
            vs.chunks.append(chunk)
            vs.speech_s += chunk_s
            vs.silent_s  = 0.0
        else:
            if vs.chunks:
                vs.chunks.append(chunk)
            vs.silent_s += chunk_s

        if (
            vs.silent_s  >= self._vad_silence_s
            and vs.speech_s >= self._vad_min_s
            and not vs.processing
        ):
            utterance   = vs.chunks[:]
            vs.chunks   = []
            vs.speech_s = vs.silent_s = 0.0
            vs.processing = True
            ts = now_us()
            asyncio.create_task(self._finalize_utterance(pid, utterance, ts, vs))

    async def _finalize_utterance(
        self, pid: str, chunks: list[AudioChunk], start_us: int, vs: VoiceState,
    ) -> None:
        try:
            transcript = await self._stt.transcribe(chunks_to_wav(chunks))
            if not transcript.strip():
                return

            log.info("transcript  pid=%r  %r", pid, transcript[:120])
            await self._post_transcript(pid, start_us, transcript)
        except httpx.HTTPError as exc:
            log.error("finalize error pid=%r: %s", pid, exc)
        finally:
            vs.processing = False

    # ── data channel — stats summary on any incoming message ──────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        pid = msg.participant_id
        async with McpClient(self._mcp_url) as mcp:
            t_res, v_res = await asyncio.gather(
                mcp.call_tool("transcript_get_transcript_stats", {"source_id":      pid}),
                mcp.call_tool("video_get_video_stats",           {"participant_id": pid}),
                return_exceptions=True,
            )

        report = _format_stats(pid, t_res, v_res)
        log.info("%s", report)
        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="mcp.stats",
            pts_us=now_us(),
            data=report.encode(),
        ))

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            self._voice.pop(event.participant_id, None)

    # ── MCP write ─────────────────────────────────────────────────────────────

    async def _post_transcript(self, pid: str, timestamp_us: int, text: str) -> None:
        try:
            async with McpClient(self._mcp_url) as mcp:
                await mcp.call_tool(
                    "transcript_add_transcript",
                    {"source_id": pid, "timestamp_us": timestamp_us, "text": text},
                )
        except Exception as exc:
            log.error("transcript add failed: %s", exc)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()


def _payload(r):
    if isinstance(r, Exception):
        return None
    return r.data if hasattr(r, "data") else r.structured_content


def _format_stats(pid: str, t_res, v_res) -> str:
    t = _payload(t_res)
    v = _payload(v_res)
    lines = [f"=== stats for {pid} ==="]
    if t is None or "error" in t:
        lines.append("transcripts: unavailable")
    else:
        lines.append(
            f"transcripts: {t.get('count', 0)} utterances  "
            f"{t.get('total_chars', 0)} chars  "
            f"earliest={t.get('earliest_us', 0)}  latest={t.get('latest_us', 0)}"
        )
    if v is None or "error" in v:
        lines.append("video: unavailable")
    else:
        lines.append(
            f"video: {v.get('num_chunks', 0)} chunks  "
            f"{v.get('total_bytes', 0) // 1024} KB total  "
            f"avg {v.get('avg_chunk_bytes', 0) // 1024} KB/chunk  "
            f"earliest={v.get('earliest_us', 0)}  latest={v.get('latest_us', 0)}"
        )
    return "\n".join(lines)
