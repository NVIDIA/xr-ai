# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Web server — serves the standalone web client and a token endpoint.

Serves:
  GET  /token           — signed LiveKit JWT; returns {token, url, room}
  GET  /cert            — development root CA as an installable profile
  GET  /rtc[/*]/validate — proxied to LiveKit HTTP (token pre-check)
  WS   /rtc[/*]         — proxied to LiveKit WebSocket signaling
  GET  /*               — static files from web_client_dir (SPA fallback)

When ``web_server_tls`` is enabled the /token endpoint returns a same-origin
``wss://<host>:<web_server_port>/rtc`` URL and the /rtc* routes proxy to the
internal plaintext LiveKit signaling port.
"""
from __future__ import annotations

import asyncio
import pathlib

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from . import _lk_proxy
from ._tls import _read_public_root_ca, ensure_development_certificates
from ._token import make_client_token
from ._token_server import _proxy_client_lifespan, serve_safe, wait_until_bound
from .config import LiveKitConnectorConfig

# The generated leaf is renewed 30 days before expiry. A daily check leaves
# ample margin without putting certificate and interface probes on a hot path.
_DEVELOPMENT_CERTIFICATE_CHECK_INTERVAL_S = 24 * 60 * 60


def _build_app(cfg: LiveKitConnectorConfig, cert_bytes: bytes | None) -> FastAPI:
    lk_internal_http = f"http://127.0.0.1:{cfg.lk_port_ws}"
    lk_internal_ws   = f"ws://127.0.0.1:{cfg.lk_port_ws}"

    # Shared so /rtc/validate hits don't pay TCP+TLS startup per request.
    proxy_client = httpx.AsyncClient(timeout=5.0)

    app = FastAPI(
        title="DeviceIOHub Web Server", docs_url=None, redoc_url=None,
        lifespan=_proxy_client_lifespan(proxy_client),
    )
    app.state.development_root_ca = cert_bytes
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/cert")
    async def get_cert() -> Response:
        """Serve only the public root CA as an installable profile."""
        root_ca = app.state.development_root_ca
        if root_ca is None:
            raise HTTPException(status_code=404, detail="No root CA available")
        return Response(
            content=root_ca,
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
        self._certificate_task: asyncio.Task | None = None
        self._app: FastAPI | None = None
        self._served_development_leaf: bytes | None = None
        # Startup failure captured by _serve_safe so start() can surface the
        # real cause (and so the serve task's exception is always retrieved).
        self._serve_error: BaseException | None = None

    async def start(self) -> None:
        ssl_kwargs: dict = {}
        cert_bytes: bytes | None = None
        root_ca: str | None = None
        scheme = "http"
        if self._cfg.web_server_tls:
            cert = self._cfg.cert_file or None
            key  = self._cfg.key_file  or None
            if not cert or not key:
                # Off-loop: cert generation does RSA keygen and network probes.
                cert, key, root_ca = await asyncio.to_thread(
                    ensure_development_certificates, self._cfg.web_server_extra_sans
                )
                self._served_development_leaf = pathlib.Path(cert).read_bytes()
                logger.info("TLS: using auto-generated server leaf {}", cert)
            else:
                logger.info(
                    "TLS: using externally supplied server certificate {}; /cert is disabled "
                    "because DeviceIOHub does not own its root CA",
                    cert,
                )
            ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
            scheme = "https"
            # Serve from memory; the renewal task refreshes this state if the
            # development root ever has to be replaced.
            if root_ca is not None:
                try:
                    cert_bytes = _read_public_root_ca(root_ca)
                except ValueError as exc:
                    logger.warning(
                        "TLS: cannot expose root CA at {} through /cert: {}", root_ca, exc
                    )

        app = _build_app(self._cfg, cert_bytes)
        self._app = app

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
        if root_ca is not None:
            self._certificate_task = asyncio.create_task(
                self._monitor_development_certificates(),
                name="web-server-certificate-renewal",
            )

    async def _serve_safe(self, port: int) -> None:
        self._serve_error = await serve_safe(self._server, port, "Web server")

    async def _monitor_development_certificates(self) -> None:
        while True:
            await asyncio.sleep(_DEVELOPMENT_CERTIFICATE_CHECK_INTERVAL_S)
            try:
                await self._refresh_development_certificates()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "TLS: failed to check or reload development certificates: {!r}", exc
                )

    async def _refresh_development_certificates(self) -> None:
        cert, key, root_ca = await asyncio.to_thread(
            ensure_development_certificates, self._cfg.web_server_extra_sans
        )
        leaf_bytes = pathlib.Path(cert).read_bytes()
        root_bytes = _read_public_root_ca(root_ca)
        if leaf_bytes != self._served_development_leaf:
            ssl_context = self._server.config.ssl if self._server is not None else None
            if ssl_context is None:
                raise RuntimeError("web server TLS context is unavailable")
            ssl_context.load_cert_chain(cert, key)
            self._served_development_leaf = leaf_bytes
            logger.info("TLS: reloaded renewed development server leaf {}", cert)
        if self._app is not None:
            self._app.state.development_root_ca = root_bytes

    async def stop(self) -> None:
        if self._certificate_task:
            self._certificate_task.cancel()
            try:
                await self._certificate_task
            except asyncio.CancelledError:
                pass
            self._certificate_task = None
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task
            self._task = None
        self._server = None
        self._app = None
        self._served_development_leaf = None
