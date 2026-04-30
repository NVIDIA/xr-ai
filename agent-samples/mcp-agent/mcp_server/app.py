# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Composed FastMCP app builder.

Mounts the transcript and video sub-servers into a single FastMCP instance
at /mcp.  The video sub-server connects to the hub as a ``ProcessorEndpoint``
to serve live latest-frame queries; the caller is responsible for managing
the endpoint task in the same asyncio loop as the uvicorn server.
"""
from __future__ import annotations

import pathlib

from fastmcp import FastMCP

from transcript_mcp_server import TranscriptStore, build_mcp as build_transcript_mcp
from video_mcp_server      import (ChunkStore, FrameProvider,
                                   build_mcp as build_video_mcp)
from xr_ai_agent           import ProcessorEndpoint, Subscribe


def build(cfg: dict) -> tuple[object, ProcessorEndpoint]:
    """Build the composed FastMCP ASGI app.

    Returns the Starlette ASGI app **and** the ProcessorEndpoint that backs
    live frame queries — the caller is responsible for running and shutting
    down the endpoint in the same event loop.
    """
    transcript_cfg = cfg.get("transcript", {})
    video_cfg      = cfg.get("video",      {})

    transcripts_dir = pathlib.Path(transcript_cfg.get("transcripts_dir", "/tmp/xr_transcripts"))
    recordings_dir  = pathlib.Path(video_cfg.get("recordings_dir",  "/dev/shm/xr-ai/recordings"))
    out_dir         = pathlib.Path(video_cfg.get("out_dir",         "/tmp/xr_video_queries"))
    hub_pub         = video_cfg.get("hub_pub",  "ipc:///tmp/xr_hub_pub")
    hub_push        = video_cfg.get("hub_push", "ipc:///tmp/xr_hub_in")
    gpu_id          = int(video_cfg.get("gpu_id", 0))
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = TranscriptStore(str(transcripts_dir))
    chunks      = ChunkStore(recordings_dir)

    ep       = ProcessorEndpoint(
        sub_addr=hub_pub, push_addr=hub_push,
        filter=Subscribe.VIDEO,
    )
    provider = FrameProvider(ep)

    mcp = FastMCP("xr-mcp")
    mcp.mount(build_transcript_mcp(transcripts),                          namespace="transcript")
    mcp.mount(build_video_mcp(chunks, out_dir, provider, gpu_id=gpu_id),  namespace="video")

    return mcp.http_app(path="/mcp"), ep


def build_app(cfg: dict):
    """Backwards-compatible entry — returns just the ASGI app.

    Used by tests; production should call ``build`` and manage the endpoint
    lifecycle alongside the uvicorn server (see mcp_server.py).
    """
    app, _ep = build(cfg)
    return app
