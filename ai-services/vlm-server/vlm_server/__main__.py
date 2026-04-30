# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vlm_server — OpenAI-compatible Vision-Language Model HTTP server.

Loads nvidia/Cosmos-Reason1-7B (or any Qwen2.5-VL-compatible model) in-process
and serves an OpenAI-compatible API:

    POST /v1/chat/completions   (messages may include image_url blocks)
    GET  /v1/models

Images are accepted as base64 data URLs: "data:image/jpeg;base64,<b64>"

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:          str    HuggingFace model ID (required)
    port:           int    HTTP port (default: 8100)
    host:           str    Bind address (default: "0.0.0.0")
    hf_token:       str    HuggingFace token for gated models
    system_prompt:  str    Optional system prompt prepended to every request
    model_cache:    str    HuggingFace weight cache path.  Resolved relative to
                           this YAML file's directory.  Default: ../models
    max_new_tokens: int    Max tokens to generate (default: 4096)
"""
import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import threading
import uuid
from pathlib import Path

import yaml

_DEFAULT_PORT   = 8100


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../models")
    p   = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
_MAX_PIXELS     = 1280 * 28 * 28   # ~1 MP — matches vlm-agent pixel budget
_DEFAULT_TOKENS = 4096


# ── model backend ─────────────────────────────────────────────────────────────

class _VlmBackend:
    """Thread-safe lazy loader for any AutoModelForImageTextToText model."""

    def __init__(self, model_id: str, system_prompt: str, model_cache: Path) -> None:
        self._model_id     = model_id
        self._system_prompt = system_prompt
        self._model_cache  = model_cache
        self._model        = None
        self._processor    = None
        self._lock         = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._model_cache.mkdir(parents=True, exist_ok=True)
            cache = str(self._model_cache)
            kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache)
            proc_kw = dict(
                cache_dir=cache,
                min_pixels=256 * 28 * 28,
                max_pixels=_MAX_PIXELS,
            )
            print(f"[vlm_server] Loading {self._model_id}  cache={cache}")
            try:
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id, local_files_only=True, **kwargs,
                ).eval()
                self._processor = AutoProcessor.from_pretrained(
                    self._model_id, local_files_only=True, **proc_kw,
                )
            except OSError:
                print(f"[vlm_server] Downloading {self._model_id}…")
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id, **kwargs,
                ).eval()
                self._processor = AutoProcessor.from_pretrained(
                    self._model_id, **proc_kw,
                )
            dev = next(self._model.parameters()).device
            print(f"[vlm_server] Model ready on {dev}")

    def infer(self, images: list, text: str, max_new_tokens: int) -> str:
        """Synchronous. Call from a thread pool."""
        from PIL import Image
        self._ensure_loaded()
        import torch
        from qwen_vl_utils import process_vision_info

        # Resize images that exceed the pixel budget
        pil_images = []
        for img in images:
            if img.width * img.height > _MAX_PIXELS:
                scale = (_MAX_PIXELS / (img.width * img.height)) ** 0.5
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)), Image.LANCZOS,
                )
            pil_images.append(img)

        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": content})

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.05,
            )
        trimmed = out_ids[:, inputs.input_ids.shape[1]:]
        raw = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    def infer_stream(self, images: list, text: str, max_new_tokens: int):
        """Synchronous generator that yields answer tokens as they're produced.

        Buffers and discards any <think>…</think> preamble so only the answer
        reaches callers. Call from a background thread — NOT the event loop.
        """
        from transformers import TextIteratorStreamer
        self._ensure_loaded()
        import torch
        from qwen_vl_utils import process_vision_info
        from PIL import Image

        pil_images = []
        for img in images:
            if img.width * img.height > _MAX_PIXELS:
                scale = (_MAX_PIXELS / (img.width * img.height)) ** 0.5
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)), Image.LANCZOS,
                )
            pil_images.append(img)

        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": text})
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": content})

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=120.0,
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,
        )
        t = threading.Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
        t.start()

        # Buffer tokens until the <think> preamble (if any) is complete.
        # Once past it, yield tokens directly for minimum latency.
        preamble = ""
        past_think = False
        with torch.inference_mode():
            for token in streamer:
                if past_think:
                    yield token
                    continue
                preamble += token
                if "</think>" in preamble:
                    past_think = True
                    tail = preamble.split("</think>", 1)[1].lstrip()
                    if tail:
                        yield tail
                elif not preamble.lstrip().startswith("<") and len(preamble) >= 8:
                    # No think block — emit accumulated preamble and stream directly.
                    past_think = True
                    yield preamble
        t.join()


# ── image decoding ─────────────────────────────────────────────────────────────

def _decode_image(url: str):
    """Decode a data: URL or HTTP URL into a PIL Image."""
    from PIL import Image
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    # Plain URL — callers should pre-download; reject here to keep deps minimal.
    raise ValueError(f"Only data: URLs are supported; got: {url[:64]!r}")


# ── FastAPI app ───────────────────────────────────────────────────────────────

def _build_app(cfg: dict, model_cache: Path):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    model_id      = cfg["model"]
    system_prompt = cfg.get("system_prompt", "").strip()
    max_new_tokens = int(cfg.get("max_new_tokens", _DEFAULT_TOKENS))

    backend = _VlmBackend(model_id, system_prompt, model_cache)

    app = FastAPI(title="VLM Server", version="0.1.0")

    class ImageUrl(BaseModel):
        url: str

    class ContentBlock(BaseModel):
        type:      str
        text:      str | None       = None
        image_url: ImageUrl | None  = None

    class Message(BaseModel):
        role:    str
        content: str | list[ContentBlock]

    class ChatRequest(BaseModel):
        model:          str          = model_id
        messages:       list[Message]
        max_tokens:     int          = max_new_tokens
        temperature:    float        = 0.0
        stream:         bool         = False

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        images, text_parts = [], []
        for msg in req.messages:
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            else:
                for block in msg.content:
                    if block.type == "image_url" and block.image_url:
                        images.append(_decode_image(block.image_url.url))
                    elif block.type == "text" and block.text:
                        text_parts.append(block.text)

        query  = "\n".join(text_parts)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if req.stream:
            async def _sse():
                loop  = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def _run_stream():
                    try:
                        for token in backend.infer_stream(images, query, req.max_tokens):
                            loop.call_soon_threadsafe(queue.put_nowait, token)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                threading.Thread(target=_run_stream, daemon=True).start()

                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk",
                        "model": model_id,
                        "choices": [{"index": 0,
                                     "delta": {"content": token},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                done = {
                    "id": req_id, "object": "chat.completion.chunk",
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")

        loop   = asyncio.get_running_loop()
        answer = await loop.run_in_executor(
            None, backend.infer, images, query, req.max_tokens,
        )
        return {
            "id":      req_id,
            "object":  "chat.completion",
            "model":   model_id,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app, backend


# ── entry point ───────────────────────────────────────────────────────────────

async def _run(cfg: dict, yaml_dir: Path) -> None:
    import uvicorn

    if not cfg.get("model"):
        print("[vlm_server] 'model' is required in config", file=sys.stderr)
        sys.exit(1)

    model_cache = _resolve_model_cache(cfg, yaml_dir)

    if hf_token := (cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")):
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    port = int(cfg.get("port", _DEFAULT_PORT))
    host = cfg.get("host", "0.0.0.0")

    app, backend = _build_app(cfg, model_cache)

    # Warm up — load weights now so first request is fast.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend._ensure_loaded)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    print(f"[vlm_server] Ready  →  http://localhost:{port}/v1")
    await server.serve()
    print("[vlm_server] Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    yaml_dir  = Path.cwd()
    if ns.config and ns.config.exists():
        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(_run(cfg, yaml_dir))


if __name__ == "__main__":
    run()
