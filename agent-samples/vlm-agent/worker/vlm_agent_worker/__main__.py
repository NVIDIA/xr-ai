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

Config (vlm_agent_worker.yaml in the sample root, auto-passed by launcher)
---------------------------------------------------------------------------
    model:     nvidia/Cosmos-Reason2-8B   # HuggingFace model ID
    hf_token:  hf_xxxx                    # token for gated models — do NOT commit

    # hf_token can also be provided via the HF_TOKEN environment variable.

First-query note: the model is loaded on demand. The first query after
startup will block for ~30–60 s while weights load from disk.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import signal
import threading

# Must be set before huggingface_hub is imported — enables the Rust-based
# parallel downloader (hf-transfer) which is ~5-10x faster than urllib.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np
import yaml
from PIL import Image

from xr_ai_agent import (DataMessage, FrameData, FrameSignal, ParticipantEvent,
                          PixelFormat, ProcessorEndpoint)

log = logging.getLogger("vlm_agent")

# Cache under vlm-agent/ (two levels up from this file: vlm_agent_worker/ → worker/ → vlm-agent/)
_MODEL_CACHE = pathlib.Path(__file__).resolve().parents[2] / "models"


def _load_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_config(cfg: dict) -> tuple[str, str]:
    """Apply config values to environment. Returns (model_id, system_prompt)."""
    if token := cfg.get("hf_token", "").strip():
        os.environ["HF_TOKEN"] = token
    model = cfg.get("model", "").strip() or os.environ.get("VLM_MODEL", "nvidia/Cosmos-Reason2-8B")
    system_prompt = cfg.get("system_prompt", "").strip()
    return model, system_prompt

# Qwen2.5-VL image token budget: 1 token per 28×28 px patch.
# Large frames (e.g. 1920×1080) produce ~2500 tokens and push the model past
# its context limit, triggering a CUDA device-side assert.  Cap to ~1 MP.
_MAX_IMAGE_PIXELS = 1280 * 28 * 28   # ≈ 1 003 520 px  (~1002×1002)
_HUB_PUB     = "ipc:///tmp/xr_hub_pub"
_HUB_PUSH    = "ipc:///tmp/xr_hub_in"


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


# ── VLM inference ─────────────────────────────────────────────────────────────

class _VlmBackend:
    """
    Thread-safe lazy loader and inference runner for any VLM supported by AutoModelForImageTextToText.

    Model is loaded on the first call to infer() and reused for all subsequent
    calls. Loading blocks the calling thread; use run_in_executor from asyncio.
    """

    def __init__(self, model_id: str, system_prompt: str = "") -> None:
        self._model_id     = model_id
        self._system_prompt = system_prompt
        self._model     = None
        self._processor = None
        self._lock      = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            _MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            log.info("Loading VLM %s  cache=%s", self._model_id, _MODEL_CACHE)
            kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=str(_MODEL_CACHE),
            )
            proc_kwargs = dict(
                cache_dir=str(_MODEL_CACHE),
                min_pixels=256 * 28 * 28,
                max_pixels=_MAX_IMAGE_PIXELS,
            )
            # Try offline first — avoids a network round-trip when weights are cached.
            try:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id, local_files_only=True, **kwargs,
                ).eval()
                self._processor = AutoProcessor.from_pretrained(
                    self._model_id, local_files_only=True, **proc_kwargs,
                )
            except OSError:
                log.info("Model not in cache — downloading %s", self._model_id)
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id, **kwargs,
                ).eval()
                self._processor = AutoProcessor.from_pretrained(
                    self._model_id, **proc_kwargs,
                )
            log.info("VLM ready on %s", next(self._model.parameters()).device)

    def infer(self, image: Image.Image, query: str) -> str:
        """Synchronous inference. Call from a thread pool, not the event loop."""
        self._ensure_loaded()
        import torch
        from qwen_vl_utils import process_vision_info

        # Resize before tokenisation — large frames (e.g. 1920×1080) exceed the
        # model's context window and cause a CUDA device-side assert at runtime.
        if image.width * image.height > _MAX_IMAGE_PIXELS:
            scale = (_MAX_IMAGE_PIXELS / (image.width * image.height)) ** 0.5
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.LANCZOS,
            )

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  query},
        ]})
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=4096,   # reasoning models emit <think> blocks before answering
                repetition_penalty=1.05,
            )
        trimmed = out_ids[:, inputs.input_ids.shape[1]:]
        raw = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        # Strip <think>…</think> reasoning block emitted by Cosmos-Reason and similar models.
        import re
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ── agent ─────────────────────────────────────────────────────────────────────

class VlmAgent:
    """
    Receives live video signals and on-demand VLM queries from XR clients.

    Flow
    ----
    1. on_frame() keeps track of the latest FrameSignal per (participant, track).
    2. on_data() — any data message is treated as a query (raw text or JSON):
       a. request_frame(latest_signal)   — pixel copy from hub SHM
       b. _frame_to_pil(frame)           — pixel format → PIL Image
       c. _VlmBackend.infer()            — VLM inference in thread pool
       d. send_return_data("vlm.response") → client data channel
    """

    def __init__(self, model_id: str, system_prompt: str = "") -> None:
        self._ep = ProcessorEndpoint(sub_addr=_HUB_PUB, push_addr=_HUB_PUSH)
        self._ep.on_frame(self._on_frame)
        self._ep.on_data(self._on_data)
        self._ep.on_participant(self._on_participant)

        self._latest: dict[tuple[str, str], FrameSignal] = {}
        self._vlm = _VlmBackend(model_id, system_prompt)

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

        image = _frame_to_pil(frame)
        log.info("vlm  pid=%r  %dx%d  query=%r", pid, frame.width, frame.height, query[:60])

        await self._ep.set_status("processing", pid)
        loop   = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, self._vlm.infer, image, query)
        log.info("vlm response  pid=%r  %d chars", pid, len(answer))
        await self._reply(pid, answer, frame.pts_us)
        await self._ep.set_status("idle", pid)

    async def _on_participant(self, event: ParticipantEvent) -> None:
        if not event.joined:
            keys = [k for k in self._latest if k[0] == event.participant_id]
            for k in keys:
                del self._latest[k]

    # ── helpers ───────────────────────────────────────────────────────────────

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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._vlm._ensure_loaded)
        await self._ep.run()

    def shutdown(self) -> None:
        self._ep.stop()
        self._ep.close()


# ── entry point ───────────────────────────────────────────────────────────────

async def main(model_id: str, system_prompt: str = "") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log.info("vlm-agent  model=%s", model_id)

    agent = VlmAgent(model_id, system_prompt)

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
    model_id, system_prompt = _apply_config(cfg)

    asyncio.run(main(model_id, system_prompt))


if __name__ == "__main__":
    run()
