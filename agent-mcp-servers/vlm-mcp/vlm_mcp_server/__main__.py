# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VLM MCP server.

Pure FastMCP — one tool at /mcp on port 8240. There are no REST endpoints,
no hub IPC subscription, and no `xr-ai-agent` runtime dependency.

The single tool ``ask_image(question, image_path)`` reads a local PNG path,
encodes it as a JPEG data URL, and calls the VLM via ``xr-ai-models``
``OpenAICompatVLM``. The model's answer is returned verbatim.

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
    port:                 8240
    models:
      vlm:
        kind:     preset:cosmos_vlm
        base_url: http://localhost:8100
    vlm_request_timeout_s: 60.0
    enable_thinking: false

Legacy config (still accepted; emits a deprecation warning):
    vlm_server:           http://localhost:8100
    vlm_request_timeout_s: 60.0
    enable_thinking: false
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import pathlib
import re
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
import yaml
from fastmcp import FastMCP
from loguru import logger
from PIL import Image

from xr_ai_logging import setup_logging
from xr_ai_models import (
    ModelsConfig,
    VLMSpec,
    load_models_config_from_dict,
    make_vlm,
)
from xr_ai_models.config import KIND_OPENAI_COMPAT
from xr_ai_models.protocols import VLMService


# ── VLM factory ──────────────────────────────────────────────────────────────

def _make_vlm_from_cfg(cfg: dict[str, Any]) -> tuple[VLMService, float]:
    """Construct a VLMService from the server config dict.

    Accepts either the new ``models:`` block (forwarded to the SDK config
    loader) or the legacy ``vlm_server:`` URL key (back-compat; synthesises a
    ``cosmos_vlm``-equivalent spec so existing deployments need no changes).

    Returns ``(vlm, request_timeout_s)`` so callers can surface the timeout
    that was actually wired into the spec without re-reading ``cfg``.
    """
    models_block: dict[str, Any] | None = cfg.get("models")
    vlm_server: str | None = cfg.get("vlm_server")
    vlm_request_timeout_s = float(cfg.get("vlm_request_timeout_s", 60.0))
    enable_thinking = bool(cfg.get("enable_thinking", False))

    if models_block:
        vlm_entry = dict(models_block.get("vlm") or {})
        if not vlm_entry:
            raise ValueError("models.vlm is missing or empty in vlm_mcp_server.yaml")

        if "timeout" not in vlm_entry:
            vlm_entry["timeout"] = vlm_request_timeout_s

        # The cosmos_vlm preset defaults enable_thinking to False; an explicit
        # top-level true must reach the wire by overriding default_extras.
        if enable_thinking:
            extras = dict(vlm_entry.get("default_extras") or {})
            ctk = dict(extras.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = True
            extras["chat_template_kwargs"] = ctk
            vlm_entry["default_extras"] = extras

        config = load_models_config_from_dict(
            {"vlm": vlm_entry}, source="vlm_mcp_server.yaml:models"
        )

    elif vlm_server:
        logger.warning(
            "vlm_mcp_server.yaml: 'vlm_server' key is deprecated — "
            "migrate to a 'models:' block with kind: preset:cosmos_vlm"
        )
        chat_template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
        spec = VLMSpec(
            kind=KIND_OPENAI_COMPAT,
            base_url=vlm_server,
            model_name="vlm",
            capabilities={"streaming": True, "vision": True},
            default_extras={"chat_template_kwargs": chat_template_kwargs},
            timeout=vlm_request_timeout_s,
        )
        config = ModelsConfig(entries={"vlm": spec})

    else:
        raise ValueError(
            "vlm_mcp_server.yaml must specify either a 'models:' block "
            "or the legacy 'vlm_server:' key"
        )

    return make_vlm(config, "vlm"), vlm_request_timeout_s


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

def build_mcp(vlm: VLMService) -> FastMCP:
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
        - vlm-server must be running at the ``base_url`` configured under
          ``models.vlm`` in ``vlm_mcp_server.yaml`` (default: http://localhost:8100).
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
            response = await vlm.ask_image(data_url, question)
            content = response.content
        except httpx.HTTPError as exc:
            logger.exception("ask_image: vlm-server HTTP error")
            return f"ask_image: vlm-server request failed: {exc}"

        # Belt-and-suspenders: strip any <think> that leaked through despite
        # enable_thinking=False — cosmos_vlm preset sets this to False but
        # some model revisions may still emit reasoning tokens.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        logger.debug(
            "ask_image  q={!r}  image={}  -> {} chars",
            question[:80], path.name, len(content),
        )
        return content

    return mcp


# ── server ────────────────────────────────────────────────────────────────────

async def _serve(cfg: dict, ready_file: pathlib.Path | None = None) -> None:
    host = cfg.get("host", "0.0.0.0")
    port = int(cfg.get("port", 8240))

    vlm, vlm_request_timeout_s = _make_vlm_from_cfg(cfg)
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
        "vlm-mcp-server  port={}  timeout={:.1f}s",
        port, vlm_request_timeout_s,
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
