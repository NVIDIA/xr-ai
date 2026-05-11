# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Token server — browser-facing HTTPS entry point.

Serves:
  GET  /token            — signed LiveKit JWT for browser clients
  GET  /rtc[/*]/validate — proxied to LiveKit HTTP (token pre-check)
  WS   /rtc[/*]          — proxied to LiveKit WebSocket (signaling)
  GET  /                 — optional browser static files
"""
from __future__ import annotations

import asyncio

import httpx
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from . import _lk_proxy
from ._token import make_client_token
from .config import LiveKitConnectorConfig


def build_app(cfg: LiveKitConnectorConfig) -> FastAPI:
    lk_internal_http = f"http://127.0.0.1:{cfg.lk_port_ws}"
    lk_internal_ws   = f"ws://127.0.0.1:{cfg.lk_port_ws}"

    app = FastAPI(title="XR-Media-Hub Token Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    proxy_client = httpx.AsyncClient(timeout=5.0)

    @app.on_event("shutdown")
    async def _close_proxy_client() -> None:
        await proxy_client.aclose()

    @app.get("/token")
    async def get_token(identity: str = Query(default="browser-user")) -> dict:
        token = make_client_token(cfg, identity=identity, ttl=None)
        return {"token": token, "room": cfg.room_name, "url": cfg.token_server_url}

    _lk_proxy.mount_rtc_proxy(
        app,
        client=proxy_client,
        lk_internal_http=lk_internal_http,
        lk_internal_ws=lk_internal_ws,
    )

    if cfg.browser_dir:
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=cfg.browser_dir, html=True), name="static")

    return app


class TokenServer:
    def __init__(self, cfg: LiveKitConnectorConfig) -> None:
        self._cfg = cfg
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        app = build_app(self._cfg)
        uv_cfg: dict = dict(
            app=app,
            host=self._cfg.token_server_host,
            port=self._cfg.token_server_port,
            log_level="warning",
        )
        if self._cfg.cert_file and self._cfg.key_file:
            uv_cfg["ssl_certfile"] = self._cfg.cert_file
            uv_cfg["ssl_keyfile"]  = self._cfg.key_file
            scheme = "https"
        else:
            scheme = "http"

        self._server = uvicorn.Server(uvicorn.Config(**uv_cfg))
        self._task = asyncio.create_task(self._server.serve())
        logger.info(
            "Token server → {}://{}:{}  room={!r}",
            scheme, self._cfg.token_server_host, self._cfg.token_server_port,
            self._cfg.room_name,
        )

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task
            self._task = None
        self._server = None
