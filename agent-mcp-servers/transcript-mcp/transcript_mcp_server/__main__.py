# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP compatibility process for the native text-memory functions.

Wire-shape note: republishing over the typed native surface (without the former
``untyped_outputs`` unwrapping) means ``query_transcripts`` and ``list_sources``
now return typed objects — ``{"segments": [...]}`` and ``{"sources": [...]}`` —
rather than the bare lists the legacy shim emitted. This is a deliberate change
that keeps the MCP outputs aligned with the typed native API. This process is a
transitional compatibility shim (removed in a later phase) and has no in-tree
consumer, so the typed shape is adopted rather than reproducing the legacy one.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import uvicorn
import yaml
from loguru import logger
from nat.builder.workflow_builder import WorkflowBuilder
from xr_ai_logging import setup_logging
from xr_ai_nat.functions.text_memory import TextMemoryFunctionsConfig
from xr_ai_nat.mcp import create_mcp_server


async def build_mcp(directory: str | Path):
    """Republish the four native text-memory functions under legacy MCP names."""

    async with WorkflowBuilder() as builder:
        await builder.add_function_group(
            "text_memory",
            TextMemoryFunctionsConfig(directory=directory),
        )
        group = await builder.get_function_group("text_memory")
        functions = await group.get_all_functions()

    exports = [
        functions["text_memory__query_transcripts"],
        functions["text_memory__add_transcript"],
        functions["text_memory__list_sources"],
        functions["text_memory__get_transcript_stats"],
    ]
    aliases = {
        function.instance_name: function.instance_name.removeprefix("text_memory__")
        for function in exports
    }
    return create_mcp_server(
        "transcript-mcp",
        exports,
        tool_names=aliases,
    )


async def build_app(directory: str | Path):
    """Return the ASGI app serving the compatibility transport at `/mcp`."""

    return (await build_mcp(directory)).http_app(path="/mcp")


def run() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ready-file", type=Path, default=None)
    args, _ = parser.parse_known_args()

    config: dict = {}
    if args.config and args.config.exists():
        with args.config.open() as config_file:
            config = yaml.safe_load(config_file) or {}

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    setup_logging("transcript-mcp")

    run_dir = os.environ.get("XR_RUN_DIR")
    default_directory = str(Path(run_dir) / "transcripts") if run_dir else "/tmp/xr_transcripts"
    directory = config.get("transcripts_dir") or default_directory
    host = config.get("host", "0.0.0.0")
    port = int(config.get("port", 8200))

    async def serve() -> None:
        app = await build_app(directory)
        uvicorn_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            log_config=None,
        )
        server = uvicorn.Server(uvicorn_config)
        logger.info("transcript-mcp-server  mcp=/mcp  port={}  dir={}", port, directory)
        if args.ready_file:
            args.ready_file.touch()
        await server.serve()

    asyncio.run(serve())


if __name__ == "__main__":
    run()
