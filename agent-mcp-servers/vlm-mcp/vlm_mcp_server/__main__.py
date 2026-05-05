# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VLM MCP server.

Pure FastMCP — one tool at /mcp on port 8220. There are no REST endpoints,
no hub IPC subscription, and no `xr-ai-agent` runtime dependency.

The single tool ``ask_image(question, image_path)`` reads a local PNG path,
encodes it as a JPEG data URL, and POSTs to vlm-server's OpenAI-compatible
``/v1/chat/completions`` endpoint. The model's answer is returned verbatim.

Typical two-step agent flow
───────────────────────────
1. Call ``video_mcp.get_frame_from_time(participant_id, second_ago=0)``
   (or ``second_ago=N`` for a frame from N seconds ago) to obtain a PNG
   path on the local filesystem.
2. Pass that path straight into ``ask_image`` along with your question.

vlm-mcp itself knows nothing about participants, the hub, or the frame
source — it just reads a file and forwards it to the VLM.

Tool (FastMCP, mounted at /mcp)
────────────────────────────────
  ask_image(question, image_path) → str
      Send the local image at *image_path* and *question* to vlm-server
      and return the answer text. Reads the file synchronously inside an
      executor; the asyncio loop is never blocked.

Config (vlm_mcp_server.yaml)
────────────────────────────
    host:                 0.0.0.0
    port:                 8220
    vlm_server:           http://localhost:8100
    vlm_request_timeout_s: 60.0
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import pathlib
import re
from contextlib import asynccontextmanager

import httpx
import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger
from PIL import Image

from xr_ai_logging import setup_logging


# ── VLM HTTP client ───────────────────────────────────────────────────────────

class VlmClient:
    """Thin async client for vlm-server's OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, timeout: float, enable_thinking: bool = False) -> None:
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._enable_thinking = enable_thinking
        self._client = httpx.AsyncClient(timeout=timeout)

    async def ask(self, image_data_url: str, question: str) -> str:
        payload: dict = {
            "model": "vlm",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": question},
                ],
            }],
        }
        if not self._enable_thinking:
            # Suppresses <think>…</think> generation entirely for lower latency.
            # Qwen2.5-VL (Cosmos-Reason1-7B) honours this template argument.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        resp = await self._client.post(self._url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        # Belt-and-suspenders: strip any <think> that leaked through anyway.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def close(self) -> None:
        await self._client.aclose()


# ── image helpers ─────────────────────────────────────────────────────────────

def _load_jpeg_data_url(image_path: str, quality: int = 85) -> str:
    """Open *image_path*, convert to RGB, encode as a JPEG data URL.

    Runs synchronously — caller is expected to invoke via
    ``loop.run_in_executor`` to keep the asyncio loop responsive.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ── FastMCP build ─────────────────────────────────────────────────────────────

def build_mcp(vlm: VlmClient) -> FastMCP:
    """Return a FastMCP server with the single ``ask_image`` tool bound."""
    mcp = FastMCP("vlm-mcp")

    @mcp.tool()
    async def ask_image(question: str, image_path: str) -> str:
        """
        Ask the vision-language model a question about a local image file.

        This is the primary tool for any task that requires visual understanding
        of the XR scene: describing what the user sees, reading text on screen,
        identifying objects, checking UI state, answering user questions about
        their environment, and so on.

        Typical usage pattern
        ---------------------
        Step 1 — acquire a frame::

            frame = video_mcp.get_frame_from_time(
                participant_id="alice",
                second_ago=0,            # live frame (what the user sees right now)
                # second_ago=3,          # frame from 3 seconds ago
                # reference_time_us=..., # pass the user's speech timestamp to avoid
                #                        # LLM-thinking delay shifting the frame
            )
            # frame["path"] is an absolute path to a PNG on the local filesystem

        Step 2 — ask the VLM::

            answer = vlm_mcp.ask_image(
                question="What objects are on the table?",
                image_path=frame["path"],
            )

        Good ``question`` values
        -------------------------
        - The user's exact words:  "what is that?"
        - A rephrasing:            "Describe what the user is looking at."
        - A specific sub-question: "What text appears on the whiteboard?"
        - A follow-up:             "List every distinct color visible in the scene."
        - Counting:                "How many people are in the frame?"
        - Spatial:                 "Is there anything in the top-left corner?"

        Parameters
        ----------
        question
            Free-form natural-language question or instruction for the VLM.
            The model receives both the image and this text in the same turn.
        image_path
            Absolute path to a local image file (PNG or JPEG). Typically the
            ``path`` value returned by ``video_mcp.get_frame_from_time``.
            The file is read from disk and sent to vlm-server as a
            base64-encoded JPEG (quality 85).

        Returns
        -------
        str
            The VLM's answer, trimmed of leading/trailing whitespace.
            On error (file not found, server unreachable, etc.) returns a
            human-readable error string starting with ``"ask_image: ..."``.

        Notes
        -----
        - ``image_path`` MUST be the ``path`` value returned by a prior call to
          ``get_frame_from_time`` or ``get_latest_frame``.  Never invent or guess
          a path — the tool will return an error and the task will fail.
        - Image I/O runs in a thread pool so the asyncio event loop is never
          blocked even for large frames.
        - The tool does NOT maintain conversation history. Each call is
          independent; pass relevant context in ``question`` if needed.
        - vlm-server must be running at the ``vlm_server`` URL configured in
          ``vlm_mcp_server.yaml`` (default: http://localhost:8100).
        """
        if not image_path:
            return "ask_image: image_path is empty — call video_mcp.get_frame_from_time first."
        path = pathlib.Path(image_path)
        if not path.exists():
            return f"ask_image: file not found at {image_path!r}."

        loop = asyncio.get_running_loop()
        try:
            data_url = await loop.run_in_executor(None, _load_jpeg_data_url, str(path))
        except Exception as exc:
            logger.exception("ask_image: failed to load {}", image_path)
            return f"ask_image: failed to read image at {image_path!r}: {exc}"

        try:
            answer = await vlm.ask(data_url, question)
        except httpx.HTTPError as exc:
            logger.exception("ask_image: vlm-server HTTP error")
            return f"ask_image: vlm-server request failed: {exc}"

        logger.debug(
            "ask_image  q={!r}  image={}  -> {} chars",
            question[:80], path.name, len(answer),
        )
        return answer.strip()

    return mcp


# ── server ────────────────────────────────────────────────────────────────────

async def _serve(cfg: dict, ready_file: pathlib.Path | None = None) -> None:
    host                  = cfg.get("host", "0.0.0.0")
    port                  = int(cfg.get("port", 8220))
    vlm_server            = cfg.get("vlm_server", "http://localhost:8100")
    vlm_request_timeout_s = float(cfg.get("vlm_request_timeout_s", 60.0))
    enable_thinking       = bool(cfg.get("enable_thinking", False))

    vlm = VlmClient(vlm_server, timeout=vlm_request_timeout_s, enable_thinking=enable_thinking)
    mcp = build_mcp(vlm)
    app = mcp.http_app(path="/mcp")

    @asynccontextmanager
    async def _lifespan(_app):
        try:
            yield
        finally:
            await vlm.close()

    # FastMCP installs its own lifespan on the returned ASGI app; chain ours
    # so the VLM client is closed cleanly when uvicorn shuts down.
    base_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _combined(_app):
        async with base_lifespan(_app):
            async with _lifespan(_app):
                yield

    app.router.lifespan_context = _combined

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info(
        "vlm-mcp-server  port={}  vlm_server={}  timeout={:.1f}s",
        port, vlm_server, vlm_request_timeout_s,
    )
    if ready_file:
        ready_file.touch()
    await server.serve()


def run() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=pathlib.Path, default=None)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    if ns.config and ns.config.exists():
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    setup_logging("vlm-mcp")
    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
