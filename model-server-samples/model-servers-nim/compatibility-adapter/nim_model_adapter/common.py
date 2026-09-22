# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared readiness, identity, and client lifetime for compatibility endpoints."""
from contextlib import asynccontextmanager

import grpc
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


def identity(config: dict) -> dict:
    settings = {key: value for key, value in config.items() if key not in ("host", "port")}
    settings.setdefault("kind", "embedding")
    if "base_url" in settings:
        settings["base_url"] = settings["base_url"].rstrip("/")
    return {"status": "ok", "service": "nim-model-adapter", "configuration": settings}


def create_app(config: dict, backends: list) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            for backend in backends:
                await backend.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/health/ready")
    @app.get("/health")
    async def health():
        for backend in backends:
            if not await backend.health():
                raise HTTPException(503, "NIM is not ready")
        if url := config.get("health_url"):
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
                    response = await client.get(url)
                    response.raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(503, "NIM is not ready") from exc
        return identity(config)

    @app.get("/v1/models")
    async def models():
        alias = config.get("alias", "embed")
        return {"object": "list", "data": [{"id": alias, "object": "model"}]}

    @app.exception_handler(httpx.HTTPStatusError)
    async def upstream_status(_request, exc):
        return JSONResponse({"detail": "NIM rejected the request"}, status_code=exc.response.status_code)

    @app.exception_handler(httpx.RequestError)
    async def upstream_connection(_request, _exc):
        return JSONResponse({"detail": "NIM connection failed"}, status_code=502)

    @app.exception_handler(httpx.TimeoutException)
    @app.exception_handler(TimeoutError)
    async def upstream_timeout(_request, _exc):
        return JSONResponse({"detail": "NIM request timed out"}, status_code=504)

    @app.exception_handler(grpc.RpcError)
    async def upstream_speech(_request, _exc):
        return JSONResponse({"detail": "Speech NIM request failed"}, status_code=502)

    return app
