# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-origin reverse proxy from the hub's web/token server to LiveKit."""
from __future__ import annotations

import asyncio
from wsgiref.util import is_hop_by_hop

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger

# Headers set by the proxy/server itself in the upstream handshake.
# wsgiref.util.is_hop_by_hop covers RFC 7230 Connection/Upgrade/Keep-Alive/etc.
_WS_FRAMING_HEADERS = frozenset({
    "host",
    "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
    "sec-websocket-protocol", "sec-websocket-accept",
})


def _forward_headers(items) -> dict[str, str]:
    return {
        k: v for k, v in items
        if not is_hop_by_hop(k) and k.lower() not in _WS_FRAMING_HEADERS
    }


def _rtc_path(tail: str) -> str:
    return f"/rtc/{tail}" if tail else "/rtc"


async def proxy_validate(
    client: httpx.AsyncClient,
    lk_internal_http: str,
    request: Request,
    tail: str = "",
) -> Response:
    qs = str(request.url.query)
    headers = _forward_headers(request.headers.items())
    r = await client.get(f"{lk_internal_http}{_rtc_path(tail)}/validate?{qs}", headers=headers)
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "text/plain"),
    )


async def pump_rtc_ws(
    client_ws: WebSocket,
    lk_internal_ws: str,
    tail: str = "",
) -> None:
    qs = client_ws.scope.get("query_string", b"").decode()
    target = f"{lk_internal_ws}{_rtc_path(tail)}?{qs}"
    fwd_headers = _forward_headers(client_ws.headers.items())
    await client_ws.accept()
    try:
        async with websockets.connect(
            target, additional_headers=fwd_headers, max_size=None,
        ) as lk_ws:

            async def c2l() -> None:
                try:
                    while True:
                        msg = await client_ws.receive()
                        msg_type = msg.get("type")
                        if msg_type == "websocket.disconnect":
                            break
                        # Guard against future ASGI event types — only
                        # websocket.receive carries payload to forward.
                        if msg_type != "websocket.receive":
                            continue
                        if msg.get("bytes"):
                            await lk_ws.send(msg["bytes"])
                        elif msg.get("text"):
                            await lk_ws.send(msg["text"])
                except Exception as exc:
                    logger.debug("WS proxy /rtc c2l ended: {}", exc)
                finally:
                    await lk_ws.close()

            async def l2c() -> None:
                try:
                    async for frame in lk_ws:
                        if isinstance(frame, bytes):
                            await client_ws.send_bytes(frame)
                        else:
                            await client_ws.send_text(frame)
                except Exception as exc:
                    logger.debug("WS proxy /rtc l2c ended: {}", exc)

            await asyncio.gather(c2l(), l2c(), return_exceptions=True)
    except Exception as exc:
        logger.debug("WS proxy /rtc error: {}", exc)
    finally:
        try:
            await client_ws.close()
        except Exception:
            pass  # already closed


def mount_rtc_proxy(
    app: FastAPI,
    *,
    client: httpx.AsyncClient,
    lk_internal_http: str,
    lk_internal_ws: str,
) -> None:
    """Register the four /rtc[/<version>] routes on *app*.

    The ``{tail:path}`` segment carries the SDK's protocol version (``v1``
    for livekit-client v2.x); the empty-tail routes preserve the pre-v2
    plain ``/rtc`` form. Both pairs forward end-to-end headers so the
    LiveKit Swift SDK's ``Authorization: Bearer`` reaches the server.
    """
    @app.get("/rtc/validate")
    async def _rtc_validate_root(request: Request) -> Response:
        return await proxy_validate(client, lk_internal_http, request)

    @app.get("/rtc/{tail:path}/validate")
    async def _rtc_validate_versioned(tail: str, request: Request) -> Response:
        return await proxy_validate(client, lk_internal_http, request, tail)

    @app.websocket("/rtc")
    async def _rtc_ws_root(client_ws: WebSocket) -> None:
        await pump_rtc_ws(client_ws, lk_internal_ws)

    @app.websocket("/rtc/{tail:path}")
    async def _rtc_ws_versioned(client_ws: WebSocket, tail: str) -> None:
        await pump_rtc_ws(client_ws, lk_internal_ws, tail)
