"""
Token server — browser-facing HTTPS entry point.

Serves:
  GET  /token           — signed LiveKit JWT for browser clients
  GET  /rtc/validate    — proxied to LiveKit HTTP (token pre-check)
  WS   /rtc             — proxied to LiveKit WebSocket (signaling)
  GET  /                — optional browser static files

Runs programmatically via uvicorn so the caller can await start()/stop().
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
from livekit.api import AccessToken, VideoGrants

from .config import LiveKitConnectorConfig

log = logging.getLogger(__name__)


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

    @app.get("/token")
    async def get_token(identity: str = Query(default="browser-user")) -> dict:
        token = (
            AccessToken(cfg.api_key, cfg.api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=cfg.room_name))
            .to_jwt()
        )
        return {"token": token, "room": cfg.room_name, "url": cfg.token_server_url}

    @app.get("/rtc/validate")
    async def rtc_validate(request: Request) -> Response:
        qs = str(request.url.query)
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{lk_internal_http}/rtc/validate?{qs}")
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
                            if msg["type"] == "websocket.disconnect":
                                break
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
        log.info(
            "Token server → %s://%s:%d  room=%r",
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
