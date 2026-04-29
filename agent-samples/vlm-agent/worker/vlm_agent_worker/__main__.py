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

All client communication (VLM response text, camera control, audio) goes
out via this worker's own ProcessorEndpoint — no MCP indirection.

Config (vlm_agent_worker.yaml in the sample root, auto-passed by launcher)
---------------------------------------------------------------------------
    vlm_server:  http://localhost:8100   # vlm-server HTTP API
    tts_server:  http://localhost:8104   # tts-server HTTP API
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
_FRAME_STALE_S = 2.0                 # signals older than this are treated as no frame
_CAMERA_ON_TIMEOUT_S = 10.0          # generous: client chains stop+start serially
_CAMERA_GRACE_S = 5.0                # keep camera on this long after each query so
                                     # rapid follow-up queries don't re-cycle it


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
        self._latest: dict[tuple[str, str], tuple[FrameSignal, float]] = {}  # signal + monotonic timestamp
        self._frame_events: dict[str, asyncio.Event] = {}
        self._inflight: dict[str, asyncio.Task] = {}  # current query task per participant
        # Camera state at the worker. We keep the camera on for a grace period
        # after each query so a follow-up doesn't pay a full start/stop cycle.
        self._camera_on:   dict[str, bool] = {}
        self._stop_timers: dict[str, asyncio.Task] = {}

    # ── callbacks ─────────────────────────────────────────────────────────────

    async def _on_frame(self, sig: FrameSignal) -> None:
        self._latest[(sig.participant_id, sig.track_id)] = (sig, time.monotonic())
        ev = self._frame_events.get(sig.participant_id)
        if ev:
            ev.set()

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

        # Cancel any in-flight task for this participant; its finally blocks
        # release the camera and tear down sub-tasks.
        prev = self._inflight.get(pid)
        if prev is not None and not prev.done():
            log.info("vlm  pid=%r — interrupting previous query", pid)
            prev.cancel()
            try:
                await prev
            except (asyncio.CancelledError, Exception):
                pass

        # Always flush queued return audio. The previous task usually completes
        # well before its TTS finishes playing on the client (sender drains the
        # queue in ms; playback takes seconds), so even when there is no live
        # task to cancel the hub still has audio queued from the prior query.
        # Chunks already on the wire to the client may play out for ~100 ms.
        await self._ep.flush_return_audio(pid)

        me = asyncio.current_task()
        self._inflight[pid] = me
        try:
            await self._handle_query(pid, query, track_id, msg.pts_us)
        finally:
            if self._inflight.get(pid) is me:
                self._inflight.pop(pid, None)

    async def _handle_query(
        self,
        pid: str,
        query: str,
        track_id: str | None,
        pts_us: int,
    ) -> None:
        sig = self._pick_signal(pid, track_id)
        need_camera = sig is None

        if need_camera:
            log.info("vlm  pid=%r — no frame, requesting camera on demand", pid)
            await self._ensure_camera_on(pid)
            ev = asyncio.Event()
            self._frame_events[pid] = ev
            try:
                await asyncio.wait_for(ev.wait(), timeout=_CAMERA_ON_TIMEOUT_S)
            except asyncio.TimeoutError:
                self._schedule_camera_off(pid)
                await self._reply(pid, "Camera unavailable — enable camera or Camera On Demand.", pts_us)
                return
            finally:
                self._frame_events.pop(pid, None)
            sig = self._pick_signal(pid, track_id)

        if sig is None:
            self._schedule_camera_off(pid)
            await self._reply(pid, "Frame data unavailable — please retry.", pts_us)
            return

        frame = await self._ep.request_frame(sig)
        if frame is None:
            self._schedule_camera_off(pid)
            await self._reply(pid, "Frame data unavailable — please retry.", pts_us)
            return

        # Camera will stop after the grace period unless another query renews it.
        self._schedule_camera_off(pid)

        image     = _frame_to_pil(frame)
        image_url = _encode_image(image)
        log.info("vlm  pid=%r  %dx%d  query=%r", pid, frame.width, frame.height, query[:60])

        await self._ep.set_status("processing", pid)
        full_response = ""
        sentence_buf  = ""
        # Queue of synthesis tasks in sentence order. None signals the sender to stop.
        tts_queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()
        synth_tasks: list[asyncio.Task] = []

        async def _audio_sender() -> None:
            # Lead-time pacing: keep ~250 ms of audio ahead of real-time playback,
            # then sleep so we don't outpace it. Bursts to fill the lead first; once
            # caught up, paces at audio rate. Bounded buffer keeps the worker → hub
            # → connector pipeline shallow so flush_return_audio actually interrupts
            # quickly. 250 ms is enough headroom that scheduling jitter never
            # underruns playback.
            target_lead_s = 0.25
            start = None
            audio_sent_s = 0.0
            while True:
                task = await tts_queue.get()
                if task is None:
                    break
                try:
                    wav = await task
                    for chunk in _wav_to_chunks(wav, pid):
                        if start is None:
                            start = time.monotonic()
                        await self._ep.send_return_audio(chunk)
                        audio_sent_s += chunk.samples / chunk.sample_rate
                        lead = audio_sent_s - (time.monotonic() - start)
                        if lead > target_lead_s:
                            await asyncio.sleep(lead - target_lead_s)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.error("tts audio error pid=%r: %s", pid, exc, exc_info=True)

        sender = asyncio.create_task(_audio_sender())

        try:
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
                            synth = asyncio.create_task(self._synthesize(sentence))
                            synth_tasks.append(synth)
                            await tts_queue.put(synth)
            except httpx.HTTPError as exc:
                log.error("vlm-server error: %s", exc)
                await tts_queue.put(None)
                await sender
                await self._reply(pid, "VLM server unavailable — please retry.", frame.pts_us)
                return

            if sentence_buf.strip():
                synth = asyncio.create_task(self._synthesize(sentence_buf.strip()))
                synth_tasks.append(synth)
                await tts_queue.put(synth)
            await tts_queue.put(None)

            full_response = full_response.strip()
            log.info("vlm response  pid=%r  %d chars", pid, len(full_response))
            await self._reply(pid, full_response, frame.pts_us)
            await sender
        finally:
            if not sender.done():
                sender.cancel()
            for t in synth_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(sender, *synth_tasks, return_exceptions=True)
            await self._ep.set_status("idle", pid)

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            pid  = event.participant_id
            keys = [k for k in self._latest if k[0] == pid]
            for k in keys:
                del self._latest[k]
            self._frame_events.pop(pid, None)
            self._camera_on.pop(pid, None)
            timer = self._stop_timers.pop(pid, None)
            if timer is not None and not timer.done():
                timer.cancel()
            task = self._inflight.pop(pid, None)
            if task is not None and not task.done():
                task.cancel()

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
        now = time.monotonic()
        fresh = [(k, sig, ts) for k, (sig, ts) in self._latest.items()
                 if k[0] == pid and now - ts < _FRAME_STALE_S]
        if not fresh:
            return None
        if track_id:
            for k, sig, _ in fresh:
                if k[1] == track_id:
                    return sig
            return None
        return max(fresh, key=lambda x: x[1].seq)[1]

    async def _client_control(self, pid: str, action: str) -> None:
        await self._ep.send_return_data(DataMessage(
            participant_id=pid,
            topic="clientControl",
            pts_us=_now_us(),
            data=json.dumps({"action": action}).encode(),
        ))

    async def _ensure_camera_on(self, pid: str) -> None:
        """Cancel any pending stop and send startCamera if not already on."""
        timer = self._stop_timers.pop(pid, None)
        if timer is not None and not timer.done():
            timer.cancel()
        if not self._camera_on.get(pid, False):
            await self._client_control(pid, "startCamera")
            self._camera_on[pid] = True

    def _schedule_camera_off(self, pid: str, delay: float = _CAMERA_GRACE_S) -> None:
        """Replace any pending stop with one that fires after *delay* seconds."""
        existing = self._stop_timers.pop(pid, None)
        if existing is not None and not existing.done():
            existing.cancel()
        if not self._camera_on.get(pid, False):
            return
        async def _stop_after_delay() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            await self._client_control(pid, "stopCamera")
            self._camera_on[pid] = False
        self._stop_timers[pid] = asyncio.create_task(_stop_after_delay())

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
