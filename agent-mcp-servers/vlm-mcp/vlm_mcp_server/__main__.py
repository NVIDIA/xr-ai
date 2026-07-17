# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VLM MCP server.

Thin MCP compatibility process — one tool at /mcp on port 8240. There are no
REST endpoints, hub IPC subscriptions, or `xr-ai-agent` runtime dependencies.

The single tool ``ask_image(question, image_path)`` reads a local PNG path,
republishes the native ``xr_vision`` image-question function. Image
normalization and the VLM call stay in the native function.

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
import pathlib
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
import yaml
from loguru import logger
from nat.builder.workflow_builder import WorkflowBuilder

from xr_ai_logging import setup_logging
from xr_ai_nat.adapters.mcp import create_mcp_server
from xr_ai_nat.functions.vision import VisionFunctionsConfig
from xr_ai_nat.functions.vision._images import load_jpeg_data_url as _load_jpeg_data_url  # noqa: F401
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


# ── FastMCP build ─────────────────────────────────────────────────────────────

async def build_mcp(vlm: VLMService):
    """Republish the native vision function under the existing MCP tool name."""

    async with WorkflowBuilder() as builder:
        await builder.add_function_group("vision", VisionFunctionsConfig(vlm=vlm))
        group = await builder.get_function_group("vision")
        functions = await group.get_all_functions()

    function = functions["vision__ask_image"]
    return create_mcp_server(
        "vlm-mcp",
        [function],
        tool_names={function.instance_name: "ask_image"},
    )


# ── server ────────────────────────────────────────────────────────────────────

async def _serve(cfg: dict, ready_file: pathlib.Path | None = None) -> None:
    host = cfg.get("host", "0.0.0.0")
    port = int(cfg.get("port", 8240))

    vlm, vlm_request_timeout_s = _make_vlm_from_cfg(cfg)
    mcp = await build_mcp(vlm)
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

    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            log_config=None)
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

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    setup_logging("vlm-mcp")
    asyncio.run(_serve(cfg, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
