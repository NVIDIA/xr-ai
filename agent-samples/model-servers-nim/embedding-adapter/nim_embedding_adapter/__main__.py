# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt XR AI's prefixed embedding inputs to NIM's asymmetric model API."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xr_ai_logging import setup_logging
from xr_ai_models import AdapterSpec, EmbeddingSpec, EndpointSpec, ModelsConfig, make_embedding


class EmbeddingRequest(BaseModel):
    model: str = "embed"
    input: str | list[str]
    input_type: Literal["query", "passage"] | None = None
    encoding_format: Literal["float"] = "float"


def _split_input(text: str, input_type: str | None) -> tuple[str, str]:
    # The repository's RAG service already labels each string this way. Strip
    # the label before NIM adds the model's own prefix, avoiding double prompts.
    for kind in ("query", "passage"):
        prefix = f"{kind}: "
        if text.startswith(prefix):
            if input_type is not None and kind != input_type:
                raise ValueError("input_type conflicts with the input's query/passage prefix")
            return kind, text[len(prefix):]
    return input_type or "passage", text


def build_app(base_url: str, model_name: str, *, clients=None) -> FastAPI:
    """Create a prefix-aware adapter using the shared typed embedding client."""
    backends = clients if clients is not None else {
        kind: make_embedding(ModelsConfig({kind: EmbeddingSpec(
            adapter=AdapterSpec(model_name=f"{model_name}-{kind}"),
            endpoint=EndpointSpec(base_url=base_url, health_path="/v1/health/ready"),
        )}), kind)
        for kind in ("query", "passage")
    }

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            for backend in backends.values():
                await backend.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not await backends["passage"].health():
            raise HTTPException(503, "embedding NIM is not ready")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "embed", "object": "model"}]}

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingRequest):
        if request.model not in ("embed", model_name):
            raise HTTPException(404, "unknown embedding model")
        texts = [request.input] if isinstance(request.input, str) else request.input
        groups: dict[str, list[tuple[int, str]]] = {"query": [], "passage": []}
        try:
            for index, text in enumerate(texts):
                kind, content = _split_input(text, request.input_type)
                groups[kind].append((index, content))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        rows = []
        try:
            # Serial groups keep NIM memory use bounded on a shared GPU.
            for kind, group in groups.items():
                if not group:
                    continue
                vectors = await backends[kind].embed([text for _, text in group])
                rows.extend({"object": "embedding", "index": index, "embedding": vector}
                            for (index, _), vector in zip(group, vectors, strict=True))
        except httpx.HTTPStatusError as exc:
            raise HTTPException(exc.response.status_code, "embedding NIM rejected the request") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise HTTPException(502, "embedding NIM request failed") from exc
        return {"object": "list", "model": request.model, "data": sorted(rows, key=lambda row: row["index"])}

    return app


async def _serve(config: dict, ready_file: Path | None) -> None:
    app = build_app(config["base_url"], config["model"])
    server = uvicorn.Server(uvicorn.Config(
        app, host=config.get("host", "0.0.0.0"), port=int(config["port"]), log_level="warning",
    ))
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("embedding adapter stopped before listening")
            await asyncio.sleep(0.05)
        if ready_file:
            ready_file.touch()
        await task
    finally:
        server.should_exit = True
        if not task.done():
            await task


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    setup_logging("nim-embedding-adapter")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    asyncio.run(_serve(config, args.ready_file))


if __name__ == "__main__":
    run()
