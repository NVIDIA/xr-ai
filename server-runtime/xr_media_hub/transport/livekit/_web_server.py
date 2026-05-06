# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Web server — serves the standalone web client and a token endpoint.

Serves:
  GET  /token           — signed LiveKit JWT; returns {token, url, room}
  GET  /rtc/validate    — proxied to LiveKit HTTP (token pre-check)
  WS   /rtc             — proxied to LiveKit WebSocket signaling
  GET  /*               — static files from web_client_dir (SPA fallback)

Runs on web_server_port (default 8080) so it does not conflict with the
optional token server (default 8000) or LiveKit (7880).

When ``web_server_tls`` is enabled the /token endpoint returns a same-origin
``wss://<host>:<web_server_port>/rtc`` URL and the /rtc* routes proxy to the
internal plaintext LiveKit signaling port. This avoids the HTTPS-page-vs-ws://
mixed-content problem for browser clients (including XR headsets) without
requiring LiveKit itself to terminate TLS.
"""
from __future__ import annotations

import asyncio

import httpx
import uvicorn
from fastapi import FastAPI, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from . import _lk_proxy
from ._tls import ensure_self_signed_cert
from ._token import make_client_token
from .config import LiveKitConnectorConfig


def _build_app(cfg: LiveKitConnectorConfig) -> FastAPI:
    app = FastAPI(title="XR-Media-Hub Web Server", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    lk_internal_http = f"http://127.0.0.1:{cfg.lk_port_ws}"
    lk_internal_ws   = f"ws://127.0.0.1:{cfg.lk_port_ws}"

    # Reused across every /rtc/validate hit so we don't pay TCP+TLS startup
    # on every token check. Same lifecycle as the FastAPI app.
    proxy_client = httpx.AsyncClient(timeout=5.0)

    @app.on_event("shutdown")
    async def _close_proxy_client() -> None:
        await proxy_client.aclose()

    @app.get("/token")
    async def get_token(request: Request, identity: str = Query(default="web-user")) -> dict:
        # Derive the LiveKit host from the incoming request so the URL works
        # whether the browser is on localhost or a remote machine.
        host = request.headers.get("host", "localhost").split(":")[0]
        if cfg.web_server_tls:
            # Same-origin WSS proxy — avoids mixed-content on HTTPS pages.
            lk_url = f"wss://{host}:{cfg.web_server_port}"
        else:
            lk_url = f"ws://{host}:{cfg.lk_port_ws}"
        token = make_client_token(cfg, identity=identity, ttl=None)
        return {"token": token, "room": cfg.room_name, "url": lk_url}

    # ── LiveKit signaling proxy (only useful when web_server_tls is true) ────
    # Always register — it's harmless on plain HTTP and keeps the code simpler.

    @app.get("/rtc/validate")
    async def rtc_validate(request: Request) -> Response:
        return await _lk_proxy.proxy_validate(proxy_client, lk_internal_http, request)

    @app.websocket("/rtc")
    async def rtc_ws_proxy(client_ws: WebSocket) -> None:
        await _lk_proxy.pump_rtc_ws(client_ws, lk_internal_ws)

    # StaticFiles asserts scope["type"] == "http" and crashes on WebSocket upgrades.
    # Catch any remaining WebSocket paths and close them before the mount sees them.
    @app.websocket("/{path:path}")
    async def _close_ws(ws: WebSocket, path: str = "") -> None:
        await ws.close(1001)

    if cfg.web_client_dir:
        app.mount("/", StaticFiles(directory=cfg.web_client_dir, html=True, follow_symlink=True), name="static")

    return app


class WebServer:
    def __init__(self, cfg: LiveKitConnectorConfig) -> None:
        self._cfg = cfg
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        app = _build_app(self._cfg)

        ssl_kwargs: dict = {}
        scheme = "http"
        if self._cfg.web_server_tls:
            cert = self._cfg.cert_file or None
            key  = self._cfg.key_file  or None
            if not cert or not key:
                cert, key = ensure_self_signed_cert()
                logger.info("TLS: using auto-generated self-signed cert  {}", cert)
            ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
            scheme = "https"

        uv_cfg = uvicorn.Config(
            app=app,
            host=self._cfg.web_server_host,
            port=self._cfg.web_server_port,
            log_level="warning",
            **ssl_kwargs,
        )
        self._server = uvicorn.Server(uv_cfg)
        self._task = asyncio.create_task(self._serve_safe())
        logger.info(
            "Web server → {}://{}:{}  client={!r}",
            scheme, self._cfg.web_server_host, self._cfg.web_server_port,
            self._cfg.web_client_dir or "<no static dir>",
        )

    async def _serve_safe(self) -> None:
        # uvicorn calls sys.exit(1) on bind failure; SystemExit is a BaseException
        # and asyncio would re-raise it in the event loop, crashing the process.
        try:
            await self._server.serve()
        except SystemExit as exc:
            logger.error(
                "Web server failed to start on port {} — is it already in use? (exit code {})",
                self._cfg.web_server_port, exc.code,
            )

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task
            self._task = None
        self._server = None
