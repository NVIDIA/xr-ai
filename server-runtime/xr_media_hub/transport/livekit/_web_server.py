# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Web server — serves the standalone web client and a token endpoint.

Serves:
  GET  /token           — signed LiveKit JWT; returns {token, url, room}
  GET  /cert            — active TLS cert as an installable iOS profile
  GET  /rtc[/*]/validate — proxied to LiveKit HTTP (token pre-check)
  WS   /rtc[/*]         — proxied to LiveKit WebSocket signaling
  GET  /*               — static files from web_client_dir (SPA fallback)

When ``web_server_tls`` is enabled the /token endpoint returns a same-origin
``wss://<host>:<web_server_port>/rtc`` URL and the /rtc* routes proxy to the
internal plaintext LiveKit signaling port.
"""
from __future__ import annotations

import asyncio

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from . import _lk_proxy
from ._tls import ensure_self_signed_cert
from ._token import make_client_token
from ._token_server import _proxy_client_lifespan, serve_safe, wait_until_bound
from .config import LiveKitConnectorConfig


def _build_app(cfg: LiveKitConnectorConfig, cert_bytes: bytes | None) -> FastAPI:
    lk_internal_http = f"http://127.0.0.1:{cfg.lk_port_ws}"
    lk_internal_ws   = f"ws://127.0.0.1:{cfg.lk_port_ws}"

    # Shared so /rtc/validate hits don't pay TCP+TLS startup per request.
    proxy_client = httpx.AsyncClient(timeout=5.0)

    app = FastAPI(
        title="XR-Media-Hub Web Server", docs_url=None, redoc_url=None,
        lifespan=_proxy_client_lifespan(proxy_client),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/cert")
    async def get_cert() -> Response:
        """Serve the self-signed cert as an installable iOS profile."""
        if cert_bytes is None:
            raise HTTPException(status_code=404, detail="TLS disabled — no cert to serve")
        return Response(
            content=cert_bytes,
            media_type="application/x-x509-ca-cert",
            headers={"Content-Disposition": 'attachment; filename="xr-ai-hub.crt"'},
        )

    @app.get("/token")
    async def get_token(request: Request, identity: str = Query(default="web-user")) -> dict:
        # Use the request's Host header so the URL works for both localhost
        # and remote clients without per-deployment config.
        host = request.headers.get("host", "localhost").split(":")[0]
        if cfg.web_server_tls:
            lk_url = f"wss://{host}:{cfg.web_server_port}"
        else:
            lk_url = f"ws://{host}:{cfg.lk_port_ws}"
        token = make_client_token(cfg, identity=identity, ttl=None)
        return {"token": token, "room": cfg.room_name, "url": lk_url}

    _lk_proxy.mount_rtc_proxy(
        app,
        client=proxy_client,
        lk_internal_http=lk_internal_http,
        lk_internal_ws=lk_internal_ws,
    )

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
        # Startup failure captured by _serve_safe so start() can surface the
        # real cause (and so the serve task's exception is always retrieved).
        self._serve_error: BaseException | None = None

    async def start(self) -> None:
        ssl_kwargs: dict = {}
        cert_bytes: bytes | None = None
        scheme = "http"
        if self._cfg.web_server_tls:
            cert = self._cfg.cert_file or None
            key  = self._cfg.key_file  or None
            if not cert or not key:
                cert, key = ensure_self_signed_cert()
                logger.info("TLS: using auto-generated self-signed cert  {}", cert)
            ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
            scheme = "https"
            # Read once at startup so /cert serves from memory.
            try:
                with open(cert, "rb") as f:
                    cert_bytes = f.read()
            except OSError as exc:
                logger.warning("TLS: cannot read cert at {} for /cert endpoint: {}", cert, exc)

        app = _build_app(self._cfg, cert_bytes)

        uv_cfg = uvicorn.Config(
            app=app,
            host=self._cfg.web_server_host,
            port=self._cfg.web_server_port,
            log_level="warning",
            **ssl_kwargs,
        )
        port = self._cfg.web_server_port
        self._serve_error = None
        self._server = uvicorn.Server(uv_cfg)
        self._task = asyncio.create_task(self._serve_safe(port))

        # A port conflict must fail fast: a "started" log on a dead web server
        # leaves every browser client silently unable to reach the client/token
        # endpoint. Mirror TokenServer — poll for bind, raise on failure.
        await wait_until_bound(self._server, self._task)
        if not self._server.started:
            self._task = None
            self._server = None
            raise RuntimeError(
                f"Web server failed to start on {self._cfg.web_server_host}:{port} "
                "— port already in use, or startup timed out."
            ) from self._serve_error
        logger.info(
            "Web server → {}://{}:{}  client={!r}",
            scheme, self._cfg.web_server_host, port,
            self._cfg.web_client_dir or "<no static dir>",
        )

    async def _serve_safe(self, port: int) -> None:
        self._serve_error = await serve_safe(self._server, port, "Web server")

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task
            self._task = None
        self._server = None
