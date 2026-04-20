"""
Web server — serves the standalone web client and a token endpoint.

Serves:
  GET  /token           — signed LiveKit JWT; returns {token, url, room}
  GET  /*               — static files from web_client_dir (SPA fallback)

Runs on web_server_port (default 8080) so it does not conflict with the
optional token server (default 8000) or LiveKit (7880).
"""
from __future__ import annotations

import asyncio
import logging

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from livekit.api import AccessToken, VideoGrants

from .config import LiveKitConnectorConfig

log = logging.getLogger(__name__)


def _build_app(cfg: LiveKitConnectorConfig) -> FastAPI:
    lk_url = f"ws://{cfg.web_server_host}:{cfg.lk_port_ws}"

    app = FastAPI(title="XR-Media-Hub Web Server", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/token")
    async def get_token(identity: str = Query(default="web-user")) -> dict:
        token = (
            AccessToken(cfg.api_key, cfg.api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=cfg.room_name))
            .to_jwt()
        )
        return {"token": token, "room": cfg.room_name, "url": lk_url}

    if cfg.web_client_dir:
        app.mount("/", StaticFiles(directory=cfg.web_client_dir, html=True), name="static")

    return app


class WebServer:
    def __init__(self, cfg: LiveKitConnectorConfig) -> None:
        self._cfg = cfg
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        app = _build_app(self._cfg)
        uv_cfg = uvicorn.Config(
            app=app,
            host=self._cfg.web_server_host,
            port=self._cfg.web_server_port,
            log_level="warning",
        )
        self._server = uvicorn.Server(uv_cfg)
        self._task = asyncio.create_task(self._server.serve())
        log.info(
            "Web server → http://%s:%d  client=%r",
            self._cfg.web_server_host, self._cfg.web_server_port,
            self._cfg.web_client_dir or "<no static dir>",
        )

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task
            self._task = None
        self._server = None
