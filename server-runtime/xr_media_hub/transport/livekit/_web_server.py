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
import logging

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from livekit.api import AccessToken, VideoGrants

from ._tls import ensure_self_signed_cert
from .config import LiveKitConnectorConfig

log = logging.getLogger(__name__)


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
        token = (
            AccessToken(cfg.api_key, cfg.api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=cfg.room_name))
            .to_jwt()
        )
        return {"token": token, "room": cfg.room_name, "url": lk_url}

    # ── LiveKit signaling proxy (only useful when web_server_tls is true) ────
    # Always register — it's harmless on plain HTTP and keeps the code simpler.

    @app.get("/rtc/validate")
    async def rtc_validate(request: Request) -> Response:
        qs = str(request.url.query)
        r = await proxy_client.get(f"{lk_internal_http}/rtc/validate?{qs}")
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "text/plain"),
        )

    @app.websocket("/rtc")
    async def rtc_ws_proxy(client_ws: WebSocket) -> None:
        qs = client_ws.scope.get("query_string", b"").decode()
        target = f"{lk_internal_ws}/rtc?{qs}"
        await client_ws.accept()
        try:
            async with websockets.connect(target, max_size=None) as lk_ws:

                async def c2l() -> None:
                    try:
                        while True:
                            msg = await client_ws.receive()
                            msg_type = msg.get("type")
                            if msg_type == "websocket.disconnect":
                                break
                            # Only websocket.receive carries data — any other
                            # event type (lifespan ack, ping/pong) shouldn't
                            # land here, but skip explicitly so a future ASGI
                            # change doesn't accidentally forward control
                            # frames upstream as bogus payload.
                            if msg_type != "websocket.receive":
                                continue
                            if msg.get("bytes"):
                                await lk_ws.send(msg["bytes"])
                            elif msg.get("text"):
                                await lk_ws.send(msg["text"])
                    except Exception:
                        pass
                    finally:
                        await lk_ws.close()

                async def l2c() -> None:
                    try:
                        async for frame in lk_ws:
                            if isinstance(frame, bytes):
                                await client_ws.send_bytes(frame)
                            else:
                                await client_ws.send_text(frame)
                    except Exception:
                        pass

                await asyncio.gather(c2l(), l2c(), return_exceptions=True)
        except Exception as exc:
            log.debug("WS proxy /rtc error: %s", exc)
        finally:
            try:
                await client_ws.close()
            except Exception:
                pass

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
                log.info("TLS: using auto-generated self-signed cert  %s", cert)
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
        log.info(
            "Web server → %s://%s:%d  client=%r",
            scheme, self._cfg.web_server_host, self._cfg.web_server_port,
            self._cfg.web_client_dir or "<no static dir>",
        )

    async def _serve_safe(self) -> None:
        # uvicorn calls sys.exit(1) on bind failure; SystemExit is a BaseException
        # and asyncio would re-raise it in the event loop, crashing the process.
        try:
            await self._server.serve()
        except SystemExit as exc:
            log.error(
                "Web server failed to start on port %d — is it already in use? (exit code %s)",
                self._cfg.web_server_port, exc.code,
            )

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task
            self._task = None
        self._server = None
