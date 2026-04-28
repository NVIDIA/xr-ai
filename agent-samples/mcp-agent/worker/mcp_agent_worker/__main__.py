"""
MCP agent worker — continuous STT → transcript ingest pipeline.

Launched as a subprocess by ``uv run mcp_agent`` (the orchestrator).
Do not run this directly.

What it does
------------
1. Listens for audio from XR clients via the hub IPC.
2. Runs VAD to detect speech boundaries (same logic as echo-agent).
3. At end of each utterance, runs STT on the full audio buffer.
4. POSTs the resulting transcript (with participant ID and timestamp)
   to the transcript-mcp-server HTTP ingest endpoint.
5. On any data-channel message, queries the transcript-mcp-server and
   video-mcp-server for stats and sends them back on topic "mcp.stats".

The transcript-mcp-server and video-mcp-server are separate FastMCP
processes started by the orchestrator.  An LLM agent can connect to
either at their /mcp endpoints to call tools directly.

Config (mcp_agent_worker.yaml)
-------------------------------
    stt_server:        http://localhost:8103
    mcp_server:        http://localhost:8200
    silence_threshold: 0.01
    silence_duration:  0.8
    min_speech:        0.3
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import pathlib
import signal
import time
import wave
from dataclasses import dataclass, field

import httpx
import numpy as np
import yaml

from xr_ai_agent import AudioChunk, DataMessage, ParticipantEvent, ProcessorEndpoint

log = logging.getLogger("mcp_agent")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"


def _now_us() -> int:
    return time.time_ns() // 1_000


def _rms(data: bytes) -> float:
    arr = np.frombuffer(data, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0


def _chunks_to_wav(chunks: list) -> bytes:
    raw = b"".join(c.data for c in chunks)
    arr = np.frombuffer(raw, dtype=np.float32)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(chunks[0].channels)
        wf.setsampwidth(2)
        wf.setframerate(chunks[0].sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@dataclass
class _VoiceState:
    chunks:      list  = field(default_factory=list)
    speech_s:    float = 0.0
    silent_s:    float = 0.0
    sample_rate: int   = 16000
    channels:    int   = 1
    processing:  bool  = False


class McpAgent:
    """
    Continuous STT → transcript ingest.

    Detects speech boundaries via VAD, runs final STT on each complete
    utterance, and POSTs the result to the transcript-mcp-server.
    """

    def __init__(self, cfg: dict) -> None:
        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

        mcp_base = cfg.get("mcp_server", "http://localhost:8200").rstrip("/")
        self._stt_url              = cfg.get("stt_server", "http://localhost:8103").rstrip("/") + "/v1/audio/transcriptions"
        self._transcript_url       = mcp_base + "/ingest"
        self._transcript_stats_url = mcp_base + "/transcript/stats"
        self._video_stats_url      = mcp_base + "/video/stats"

        self._vad_threshold = float(cfg.get("silence_threshold", 0.01))
        self._vad_silence_s = float(cfg.get("silence_duration",  0.8))
        self._vad_min_s     = float(cfg.get("min_speech",        0.3))

        self._voice: dict[str, _VoiceState] = {}

    # ── voice state ───────────────────────────────────────────────────────────

    def _get_voice(self, pid: str, sample_rate: int = 16000, channels: int = 1) -> _VoiceState:
        if pid not in self._voice:
            self._voice[pid] = _VoiceState(sample_rate=sample_rate, channels=channels)
        return self._voice[pid]

    # ── audio pipeline ────────────────────────────────────────────────────────

    async def _on_audio(self, chunk: AudioChunk) -> None:
        pid = chunk.participant_id
        vs  = self._get_voice(pid, chunk.sample_rate, chunk.channels)

        chunk_s = chunk.samples / max(chunk.sample_rate, 1)
        if _rms(chunk.data) >= self._vad_threshold:
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
            ts = _now_us()
            asyncio.create_task(self._finalize_utterance(pid, utterance, ts, vs))

    async def _finalize_utterance(
        self, pid: str, chunks: list, start_us: int, vs: _VoiceState,
    ) -> None:
        try:
            transcript = await self._transcribe(_chunks_to_wav(chunks))
            if not transcript.strip():
                return

            log.info("transcript  pid=%r  %r", pid, transcript[:120])

            await self._post_transcript(pid, start_us, transcript)
        except httpx.HTTPError as exc:
            log.error("finalize error pid=%r: %s", pid, exc)
        finally:
            vs.processing = False

    async def _on_data(self, msg: DataMessage) -> None:
        pid = msg.participant_id
        async with httpx.AsyncClient(timeout=5.0) as client:
            t_resp, v_resp = await asyncio.gather(
                client.get(f"{self._transcript_stats_url}/{pid}"),
                client.get(f"{self._video_stats_url}/{pid}"),
                return_exceptions=True,
            )

        lines = [f"=== stats for {pid} ==="]

        if isinstance(t_resp, Exception) or t_resp.is_error:
            lines.append("transcripts: unavailable")
        else:
            t = t_resp.json()
            lines.append(
                f"transcripts: {t.get('count', 0)} utterances  "
                f"{t.get('total_chars', 0)} chars  "
                f"earliest={t.get('earliest_us', 0)}  latest={t.get('latest_us', 0)}"
            )

        if isinstance(v_resp, Exception) or v_resp.is_error:
            lines.append("video: unavailable")
        else:
            v = v_resp.json()
            if "error" in v:
                lines.append(f"video: {v['error']}")
            else:
                lines.append(
                    f"video: {v.get('num_chunks', 0)} chunks  "
                    f"{v.get('total_bytes', 0) // 1024} KB total  "
                    f"avg {v.get('avg_chunk_bytes', 0) // 1024} KB/chunk  "
                    f"earliest={v.get('earliest_us', 0)}  latest={v.get('latest_us', 0)}"
                )

        report = "\n".join(lines)
        log.info("%s", report)
        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="mcp.stats",
            pts_us=_now_us(),
            data=report.encode(),
        ))

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            self._voice.pop(event.participant_id, None)

    # ── HTTP calls ────────────────────────────────────────────────────────────

    async def _transcribe(self, wav_bytes: bytes) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self._stt_url,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"response_format": "json"},
            )
            if resp.is_error:
                log.error("stt %s: %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            return resp.json().get("text", "")

    async def _post_transcript(self, pid: str, timestamp_us: int, text: str) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                self._transcript_url,
                json={"participant_id": pid, "timestamp_us": timestamp_us, "text": text},
            )
            if resp.is_error:
                log.error("transcript ingest %s: %s", resp.status_code, resp.text[:200])

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()


# ── entry point ───────────────────────────────────────────────────────────────

async def _wait_for_services(cfg: dict) -> None:
    services = {
        "STT": cfg.get("stt_server",  "http://localhost:8103").rstrip("/") + "/health",
        "MCP": cfg.get("mcp_server",  "http://localhost:8200").rstrip("/") + "/health",
    }
    pending = set(services)
    while pending:
        for name in list(pending):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    r = await client.get(services[name])
                    if r.is_success:
                        log.info("%s ready", name)
                        pending.discard(name)
            except httpx.ConnectError:
                pass
        if pending:
            log.info("still waiting for: %s", ", ".join(sorted(pending)))
            await asyncio.sleep(5.0)


async def main(cfg: dict) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    await _wait_for_services(cfg)

    agent = McpAgent(cfg)
    loop  = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("mcp-agent connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()
    log.info("mcp-agent stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(main(cfg))


if __name__ == "__main__":
    run()
