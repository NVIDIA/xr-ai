"""
VLM agent worker — connects to the hub via IPC and answers VLM queries.

Launched as a subprocess by ``uv run vlm_agent`` (the orchestrator).
Do not run this directly.

Protocol
--------
Client → agent  (LiveKit data channel, any topic):
    Raw UTF-8 text  OR  JSON  {"query": "…", "track_id": "optional"}

Agent → client  (topic "vlm.response"):
    Raw UTF-8 text — the model's answer

The response is also spoken aloud via TTS, sentence by sentence.
TTS synthesis tasks are launched in parallel as each sentence completes,
so audio starts playing shortly after the first sentence is generated.

Config (vlm_agent_worker.yaml in the sample root, auto-passed by launcher)
---------------------------------------------------------------------------
    vlm_server:  http://localhost:8100   # base URL of the vlm-server HTTP API
    tts_server:  http://localhost:8104   # base URL of the tts-server HTTP API
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import pathlib
import re
import signal
import time
import wave

import httpx
import numpy as np
import yaml
from PIL import Image

from xr_ai_agent import (AudioChunk, DataMessage, FrameData, FrameSignal,
                          ParticipantEvent, PixelFormat, ProcessorEndpoint)

log = logging.getLogger("vlm_agent")

_HUB_PUB  = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH = "ipc:///tmp/xr_hub_in"

_MAX_IMAGE_PIXELS = 1280 * 28 * 28   # ~1 MP — matches vlm-server's pixel cap


def _load_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _now_us() -> int:
    return time.time_ns() // 1_000


def _wav_to_chunks(wav_bytes: bytes, participant_id: str) -> list:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_frames = max(1, sr // 50)  # 20 ms chunks
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


# ── pixel conversion ──────────────────────────────────────────────────────────

def _yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> Image.Image:
    """BT.601 limited-range YCbCr → RGB. U/V must already be full-size (upsampled)."""
    Y = Y.astype(np.float32) - 16.0
    U = U.astype(np.float32) - 128.0
    V = V.astype(np.float32) - 128.0
    R = np.clip(1.164 * Y               + 1.596 * V, 0, 255)
    G = np.clip(1.164 * Y - 0.392 * U  - 0.813 * V, 0, 255)
    B = np.clip(1.164 * Y + 2.017 * U,              0, 255)
    return Image.fromarray(np.stack([R, G, B], axis=-1).astype(np.uint8), "RGB")


def _frame_to_pil(frame: FrameData) -> Image.Image:
    w, h = frame.width, frame.height
    arr  = np.frombuffer(frame.data, dtype=np.uint8)

    if frame.fmt == PixelFormat.RGB24:
        return Image.fromarray(arr.reshape(h, w, 3), "RGB")

    if frame.fmt == PixelFormat.RGBA:
        return Image.fromarray(arr.reshape(h, w, 4), "RGBA").convert("RGB")

    if frame.fmt == PixelFormat.BGRA:
        a = arr.reshape(h, w, 4)
        return Image.fromarray(a[:, :, [2, 1, 0]], "RGB")

    if frame.fmt == PixelFormat.I420:
        y_end = w * h
        uv_sz = (w // 2) * (h // 2)
        Y = arr[:y_end].reshape(h, w)
        U = arr[y_end : y_end + uv_sz].reshape(h // 2, w // 2).repeat(2, 0).repeat(2, 1)
        V = arr[y_end + uv_sz :].reshape(h // 2, w // 2).repeat(2, 0).repeat(2, 1)
        return _yuv_to_rgb(Y, U, V)

    if frame.fmt == PixelFormat.NV12:
        y_end = w * h
        Y  = arr[:y_end].reshape(h, w)
        uv = arr[y_end:].reshape(h // 2, w)
        U  = uv[:, 0::2].repeat(2, 0).repeat(2, 1)
        V  = uv[:, 1::2].repeat(2, 0).repeat(2, 1)
        return _yuv_to_rgb(Y, U, V)

    raise ValueError(f"Unsupported pixel format: {frame.fmt!r}")


def _encode_image(image: Image.Image) -> str:
    """PIL Image → JPEG data URL for the vlm-server API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ── agent ─────────────────────────────────────────────────────────────────────

class VlmAgent:
    """
    Receives live video signals and on-demand VLM queries from XR clients.

    Flow
    ----
    1. on_frame() keeps track of the latest FrameSignal per (participant, track).
    2. on_data() — any data message is treated as a query (raw text or JSON):
       a. request_frame(latest_signal)      — pixel copy from hub SHM
       b. _frame_to_pil / _encode_image     — pixel format → JPEG data URL
       c. POST /v1/chat/completions         — vlm-server HTTP API (streaming)
       d. Each complete sentence → asyncio.create_task(_synthesize) in parallel
       e. send_return_data("vlm.response")  → client data channel
       f. Await TTS tasks in order, send audio chunks as each finishes
    """

    def __init__(self, vlm_server: str, tts_server: str) -> None:
        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_frame(self._on_frame)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

        self._vlm_url = vlm_server.rstrip("/") + "/v1/chat/completions"
        self._tts_url = tts_server.rstrip("/") + "/v1/audio/speech"
        self._latest: dict[tuple[str, str], FrameSignal] = {}

    # ── callbacks ─────────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        self._latest[(sig.participant_id, sig.track_id)] = sig

    async def _on_data(self, msg: DataMessage) -> None:
        query    = ""
        track_id = None
        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict):
                query    = payload.get("query", "")
                track_id = payload.get("track_id")
            else:
                query = str(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            query = msg.data.decode(errors="replace")

        if not query:
            return

        pid = msg.participant_id
        sig = self._pick_signal(pid, track_id)
        if sig is None:
            log.warning("vlm from %r — no video frame yet", pid)
            await self._reply(pid, "No video frame available yet.", msg.pts_us)
            return

        frame = await self._ep.request_frame(sig)
        if frame is None:
            await self._reply(pid, "Frame data unavailable — please retry.", msg.pts_us)
            return

        image     = _frame_to_pil(frame)
        image_url = _encode_image(image)
        log.info("vlm  pid=%r  %dx%d  query=%r", pid, frame.width, frame.height, query[:60])

        await self._ep.set_status("processing", pid)
        full_response = ""
        sentence_buf  = ""
        # Queue of synthesis tasks in sentence order. None signals the sender to stop.
        tts_queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()

        async def _audio_sender() -> None:
            while True:
                task = await tts_queue.get()
                if task is None:
                    break
                try:
                    wav = await task
                    for chunk in _wav_to_chunks(wav, pid):
                        await self._ep.send_return_audio(chunk)
                except Exception as exc:
                    log.error("tts audio error pid=%r: %s", pid, exc, exc_info=True)

        sender = asyncio.create_task(_audio_sender())

        try:
            async for token in self._call_vlm_stream(image_url, query):
                full_response += token
                sentence_buf  += token
                while True:
                    m = re.search(r'(?<=[.!?])\s+', sentence_buf)
                    if not m:
                        break
                    sentence     = sentence_buf[:m.start() + 1].strip()
                    sentence_buf = sentence_buf[m.end():]
                    if sentence:
                        await tts_queue.put(asyncio.create_task(self._synthesize(sentence)))
        except httpx.HTTPError as exc:
            log.error("vlm-server error: %s", exc)
            await tts_queue.put(None)
            await sender
            await self._reply(pid, "VLM server unavailable — please retry.", frame.pts_us)
            await self._ep.set_status("idle", pid)
            return

        if sentence_buf.strip():
            await tts_queue.put(asyncio.create_task(self._synthesize(sentence_buf.strip())))
        await tts_queue.put(None)

        full_response = full_response.strip()
        log.info("vlm response  pid=%r  %d chars", pid, len(full_response))
        await self._reply(pid, full_response, frame.pts_us)
        await sender
        await self._ep.set_status("idle", pid)

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            pid  = event.participant_id
            keys = [k for k in self._latest if k[0] == pid]
            for k in keys:
                del self._latest[k]

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _call_vlm_stream(self, image_url: str, query: str):
        """Async generator that yields text tokens from the VLM server via SSE."""
        payload = {
            "model": "vlm",
            "stream": True,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text",      "text": query},
            ]}],
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", self._vlm_url, json=payload) as resp:
                if resp.is_error:
                    log.error("vlm-server %s", resp.status_code)
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        chunk   = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

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

    def _pick_signal(self, pid: str, track_id: str | None) -> FrameSignal | None:
        if track_id:
            return self._latest.get((pid, track_id))
        candidates = [(k, v) for k, v in self._latest.items() if k[0] == pid]
        if not candidates:
            return None
        return max(candidates, key=lambda kv: kv[1].seq)[1]

    async def _reply(self, pid: str, text: str, pts_us: int) -> None:
        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="vlm.response",
            pts_us=pts_us,
            data=text.encode(),
        ))

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()


# ── entry point ───────────────────────────────────────────────────────────────

async def main(vlm_server: str, tts_server: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log.info("vlm-agent  vlm=%s  tts=%s", vlm_server, tts_server)

    agent = VlmAgent(vlm_server, tts_server)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    log.info("vlm-agent connecting  sub=%s  push=%s", _HUB_PUB, _HUB_PUSH)
    try:
        await agent.run()
    finally:
        agent.shutdown()

    log.info("vlm-agent stopped")


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg = _load_config(ns.config) if ns.config else {}
    vlm_server = cfg.get("vlm_server", "http://localhost:8100").strip()
    tts_server = cfg.get("tts_server", "http://localhost:8104").strip()

    asyncio.run(main(vlm_server, tts_server))


if __name__ == "__main__":
    run()
