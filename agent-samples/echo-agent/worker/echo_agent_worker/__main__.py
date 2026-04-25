"""
Echo agent worker — real-time STT → TTS echo pipeline.

Audio path:
  - Streaming: every `stream_interval` seconds during speech, STT runs on the
    growing audio buffer; newly recognised words are queued for TTS word-by-word.
  - End of speech: final STT on the full utterance; full transcript sent to the
    web client as a data message on topic "stt.transcript".

Text path: typed text → word-by-word TTS → speaker audio.

Config (echo_agent_worker.yaml)
--------------------------------
    stt_server:        http://localhost:8103
    tts_server:        http://localhost:8104
    silence_threshold: 0.01   # float32 RMS below which audio is silence
    silence_duration:  0.8    # seconds of silence that ends an utterance
    min_speech:        0.3    # minimum seconds of speech before STT fires
    stream_interval:   0.8    # seconds between streaming STT calls during speech
"""
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

log = logging.getLogger("echo_agent")

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


def _wav_to_chunks(wav_bytes: bytes, participant_id: str) -> list:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_frames = max(1, sr // 50)  # 20 ms
    pts = _now_us()
    out = []
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


@dataclass
class _VoiceState:
    chunks:      list  = field(default_factory=list)
    speech_s:    float = 0.0
    silent_s:    float = 0.0
    sample_rate: int   = 16000
    channels:    int   = 1
    processing:  bool  = False  # end-of-speech pipeline gate
    stt_busy:    bool  = False  # only one STT call at a time
    prev_words:  list  = field(default_factory=list)  # for word diff
    last_stream: float = field(default_factory=time.monotonic)
    tts_queue:   object = field(default_factory=asyncio.Queue)  # asyncio.Queue[str|None]
    tts_task:    object = None  # asyncio.Task


class EchoAgent:
    """
    Real-time STT→TTS echo pipeline.

    During speech: runs STT on a growing audio window every `stream_interval`
    seconds; newly recognised words are sent to TTS immediately.
    At end of speech: final STT run, full transcript sent as data message.
    """

    def __init__(self, cfg: dict) -> None:
        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_audio(self._on_audio)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

        self._stt_url = cfg.get("stt_server", "http://localhost:8103").rstrip("/") + "/v1/audio/transcriptions"
        self._tts_url = cfg.get("tts_server", "http://localhost:8104").rstrip("/") + "/v1/audio/speech"

        self._vad_threshold   = float(cfg.get("silence_threshold", 0.01))
        self._vad_silence_s   = float(cfg.get("silence_duration",  0.8))
        self._vad_min_s       = float(cfg.get("min_speech",        0.3))
        self._stream_interval = float(cfg.get("stream_interval",   0.8))

        self._voice: dict = {}

    # ── voice state ───────────────────────────────────────────────────────────

    def _get_voice(self, pid: str, sample_rate: int = 16000, channels: int = 1):
        if pid not in self._voice:
            vs = _VoiceState(sample_rate=sample_rate, channels=channels)
            vs.tts_task = asyncio.create_task(self._tts_consumer(pid, vs))
            self._voice[pid] = vs
        return self._voice[pid]

    # ── audio path ────────────────────────────────────────────────────────────

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

        # Streaming STT: fire periodically while speech is accumulating
        if (vs.speech_s >= self._vad_min_s
                and not vs.stt_busy
                and not vs.processing
                and time.monotonic() - vs.last_stream >= self._stream_interval):
            vs.stt_busy    = True   # set before yielding to prevent double-fire
            vs.last_stream = time.monotonic()
            asyncio.create_task(self._stream_stt(pid, vs))

        # End-of-speech detection
        if (vs.silent_s  >= self._vad_silence_s
                and vs.speech_s >= self._vad_min_s
                and not vs.processing):
            utterance   = vs.chunks[:]
            vs.chunks   = []
            vs.speech_s = vs.silent_s = 0.0
            vs.processing = True
            asyncio.create_task(self._finalize_utterance(pid, utterance, vs))

    async def _stream_stt(self, pid: str, vs: _VoiceState) -> None:
        """STT on growing audio window → enqueue new words for TTS."""
        try:
            if not vs.chunks:
                return
            transcript = await self._transcribe(_chunks_to_wav(vs.chunks[:]))
            words      = transcript.split()

            # Regression: STT returned fewer words than before — re-anchor and wait.
            if len(words) <= len(vs.prev_words):
                vs.prev_words = words
                return

            # Hold back the last word — it sits at the audio boundary where batch
            # STT hallucinations are most likely on incomplete utterances.
            stable    = words[:-1]
            new_words = stable[len(vs.prev_words):]
            if new_words:
                log.info("stream pid=%r  +%s", pid, " ".join(new_words))
                vs.prev_words = stable
                for word in new_words:
                    await vs.tts_queue.put(word)
        except httpx.HTTPError as exc:
            log.error("stream stt error pid=%r: %s", pid, exc)
        finally:
            vs.stt_busy = False

    async def _finalize_utterance(
        self, pid: str, chunks: list, vs: _VoiceState,
    ) -> None:
        """Final STT on complete utterance; send full transcript as data message."""
        try:
            transcript = await self._transcribe(_chunks_to_wav(chunks))
            if not transcript.strip():
                return

            words     = transcript.split()
            new_words = words[len(vs.prev_words):]
            if new_words:
                log.info("final  pid=%r  +%s", pid, " ".join(new_words))
                for word in new_words:
                    await vs.tts_queue.put(word)

            log.info("transcript pid=%r  %r", pid, transcript[:80])
            await self._ep.send_return_data(DataMessage(
                participant_id=pid,
                topic="stt.transcript",
                pts_us=_now_us(),
                data=transcript.encode(),
            ))
        except httpx.HTTPError as exc:
            log.error("finalize error pid=%r: %s", pid, exc)
        finally:
            vs.prev_words = []
            vs.processing = False

    # ── TTS consumer (per participant, sequential) ────────────────────────────

    async def _tts_consumer(self, pid: str, vs: _VoiceState) -> None:
        """Drain the TTS word queue sequentially to avoid overlapping audio."""
        while True:
            word = await vs.tts_queue.get()
            if word is None:
                break
            try:
                audio_wav = await self._synthesize(word)
                for out_chunk in _wav_to_chunks(audio_wav, pid):
                    await self._ep.send_return_audio(out_chunk)
            except httpx.HTTPError as exc:
                log.error("tts error pid=%r: %s", pid, exc)

    # ── text path ─────────────────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        log.info("data pid=%r  %r", msg.participant_id, text[:80])
        vs = self._get_voice(msg.participant_id)
        for word in text.split():
            await vs.tts_queue.put(word)

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            vs = self._voice.pop(event.participant_id, None)
            if vs:
                await vs.tts_queue.put(None)  # stop consumer

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

    async def _synthesize(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self._tts_url,
                json={"input": text, "response_format": "wav"},
            )
            if resp.is_error:
                log.error("tts %s: %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            return resp.content

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()


# ── entry point ───────────────────────────────────────────────────────────────

async def _wait_for_services(cfg: dict) -> None:
    services = {
        "STT": cfg.get("stt_server", "http://localhost:8103").rstrip("/") + "/health",
        "TTS": cfg.get("tts_server", "http://localhost:8104").rstrip("/") + "/health",
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

    agent = EchoAgent(cfg)
    loop  = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("echo-agent connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()
    log.info("echo-agent stopped")


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
