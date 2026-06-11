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
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from . import _lk_proxy
from ._token import make_client_token
from .config import LiveKitConnectorConfig

# Max seconds start() waits for uvicorn to bind before treating startup as
# failed. Binding the port is near-instant (no model load); the bound is a
# backstop for a startup that neither binds nor exits.
_STARTUP_TIMEOUT_S = 10.0


async def serve_safe(
    server: uvicorn.Server, port: int, label: str,
) -> BaseException | None:
    """Run ``server.serve()``, swallowing uvicorn's bind-failure SystemExit.

    uvicorn calls sys.exit(1) on bind failure; SystemExit is a BaseException
    that ``await task`` in stop() would re-raise into the caller, aborting
    graceful shutdown. Swallow it here and return the captured error so the
    caller can surface the real cause (and the task's exception is retrieved).

    CancelledError is intentionally NOT caught: start()'s timeout path cancels
    this task and awaits it.
    """
    try:
        await server.serve()
    except SystemExit as exc:
        logger.error(
            "{} failed to start on port {} — is it already in use? (exit code {})",
            label, port, exc.code,
        )
        return exc
    except Exception as exc:
        logger.error("{} crashed on port {}: {!r}", label, port, exc)
        return exc
    return None


async def wait_until_bound(server: uvicorn.Server, task: asyncio.Task) -> None:
    """Poll until *server* binds or *task* dies trying.

    uvicorn sets ``started`` True only after a successful bind; on bind failure
    startup() sys.exit(1)s first, so the task finishes with ``started`` False.
    On timeout the task is cancelled and awaited so we don't orphan it.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STARTUP_TIMEOUT_S
    while not server.started and not task.done():
        if loop.time() >= deadline:
            break
        await asyncio.sleep(0.05)
    if not server.started and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Expected: we just cancelled `task` ourselves and await it only to
            # let the cancellation propagate and the server unwind cleanly.
            pass


def _proxy_client_lifespan(client: httpx.AsyncClient):
    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        await client.aclose()
    return _lifespan


def build_app(cfg: LiveKitConnectorConfig) -> FastAPI:
    lk_internal_http = f"http://127.0.0.1:{cfg.lk_port_ws}"
    lk_internal_ws   = f"ws://127.0.0.1:{cfg.lk_port_ws}"

    proxy_client = httpx.AsyncClient(timeout=5.0)

    app = FastAPI(
        title="XR-Media-Hub Token Server",
        lifespan=_proxy_client_lifespan(proxy_client),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

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
        # Startup failure captured by _serve_safe so start() can surface the
        # real cause (and so the serve task's exception is always retrieved).
        self._serve_error: BaseException | None = None

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

        port = self._cfg.token_server_port
        self._serve_error = None
        self._server = uvicorn.Server(uvicorn.Config(**uv_cfg))
        self._task = asyncio.create_task(self._serve_safe(port))

        # A bind failure must NOT look healthy: the token server is the
        # browser-facing auth/signaling entry point, so a non-bind has to abort
        # connector startup loudly instead of leaving a dead endpoint that every
        # browser client silently fails to reach.
        await wait_until_bound(self._server, self._task)
        if not self._server.started:
            # Drop the references so a later stop() doesn't re-await (and
            # re-raise) the cancelled task. Chain the captured cause.
            self._task = None
            self._server = None
            raise RuntimeError(
                f"Token server failed to start on "
                f"{self._cfg.token_server_host}:{port} "
                "— port already in use, or startup timed out."
            ) from self._serve_error
        logger.info(
            "Token server → {}://{}:{}  room={!r}",
            scheme, self._cfg.token_server_host, port, self._cfg.room_name,
        )

    async def _serve_safe(self, port: int) -> None:
        self._serve_error = await serve_safe(self._server, port, "Token server")

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task
            self._task = None
        self._server = None
